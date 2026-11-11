"""Train/evaluate an edge-level rigidity classifier on CPU.

Usage:
    python scripts/train.py --data data/generated/edge_small.json --model gin --seeds 0 1 2
    python scripts/train.py --data data/generated/edge_small.json --model features   # shortcut check

Writes results/<run-name>/{config.json, seed<k>/model.pt, seed<k>/metrics.json, summary.json}.
The best-validation-accuracy checkpoint per seed is the one evaluated on test
and the one explainers should load.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import NUM_NODE_FEATURES, load_split_datasets  # noqa: E402
from src.models import EdgeGIN, FeatureMLP, shortcut_features  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(args):
    if args.model == "gin":
        return EdgeGIN(NUM_NODE_FEATURES, args.hidden, args.layers, args.dropout)
    return FeatureMLP(args.hidden, args.dropout)


def logits_for(model, batch, kind):
    if kind == "gin":
        return model(batch.x, batch.edge_index, batch.batch, batch.candidate)
    return model(batch.feats)


@torch.no_grad()
def evaluate(model, loader, kind):
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    tp = fp = fn = 0
    for batch in loader:
        out = logits_for(model, batch, kind)
        loss_sum += F.cross_entropy(out, batch.y, reduction="sum").item()
        pred = out.argmax(-1)
        correct += (pred == batch.y).sum().item()
        total += batch.y.numel()
        tp += ((pred == 1) & (batch.y == 1)).sum().item()
        fp += ((pred == 1) & (batch.y == 0)).sum().item()
        fn += ((pred == 0) & (batch.y == 1)).sum().item()
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    return {"loss": loss_sum / max(total, 1), "acc": correct / max(total, 1), "f1_redundant": f1, "n": total}


def run_seed(args, splits, seed: int, out_dir: Path) -> dict:
    set_seed(seed)
    loaders = {
        s: DataLoader(splits[s], batch_size=args.batch_size, shuffle=(s == "train"))
        for s in ("train", "val", "test")
    }
    model = build_model(args)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val, best_state, best_epoch, history = -1.0, None, 0, []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in loaders["train"]:
            opt.zero_grad()
            loss = F.cross_entropy(logits_for(model, batch, args.model), batch.y)
            loss.backward()
            opt.step()
        tr = evaluate(model, loaders["train"], args.model)
        va = evaluate(model, loaders["val"], args.model)
        history.append({"epoch": epoch, "train_acc": tr["acc"], "train_loss": tr["loss"],
                        "val_acc": va["acc"], "val_loss": va["loss"]})
        if va["acc"] > best_val:
            best_val, best_epoch = va["acc"], epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % args.log_every == 0 or epoch == 1:
            print(f"  seed {seed} ep {epoch:3d}  train {tr['acc']:.3f} (loss {tr['loss']:.3f})  val {va['acc']:.3f}")

    model.load_state_dict(best_state)
    metrics = {
        "seed": seed,
        "best_epoch": best_epoch,
        "train": evaluate(model, loaders["train"], args.model),
        "val": evaluate(model, loaders["val"], args.model),
        "test": evaluate(model, loaders["test"], args.model),
        "seconds": round(time.time() - t0, 1),
        "history": history,
    }
    seed_dir = out_dir / f"seed{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "config": vars_jsonable(args)}, seed_dir / "model.pt")
    (seed_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def vars_jsonable(args):
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=ROOT / "data/generated/edge_small.json")
    p.add_argument("--model", choices=["gin", "features"], default="gin")
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--name", type=str, default=None)
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    splits = load_split_datasets(args.data)
    if args.model == "features":
        for s in splits.values():
            for d in s:
                d.feats = shortcut_features(d)  # [1, F]; concatenated per batch
    print({s: len(v) for s, v in splits.items()})

    name = args.name or f"{args.model}_{args.data.stem}"
    out_dir = ROOT / "results" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars_jsonable(args), indent=2), encoding="utf-8")

    all_metrics = [run_seed(args, splits, s, out_dir) for s in args.seeds]

    def agg(split, key):
        vals = [m[split][key] for m in all_metrics]
        return {"mean": statistics.mean(vals), "std": statistics.pstdev(vals), "values": vals}

    summary = {split: {k: agg(split, k) for k in ("acc", "f1_redundant")} for split in ("train", "val", "test")}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n== {name}  ({len(args.seeds)} seeds)")
    for split in ("train", "val", "test"):
        a = summary[split]["acc"]
        print(f"  {split:5s} acc {a['mean']:.3f} ± {a['std']:.3f}")
    print(f"saved to {out_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

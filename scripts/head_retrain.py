"""Milestone 1A: is the readout (classifier head) the correctable bottleneck?

The encoder is FROZEN. We capture the exact vector the original head receives
and train new heads on it, then evaluate once on the held-out deep part of the
v2 composed-circuit benchmark (and on stress sets, to check nothing breaks).

Heads:
  original         the trained model's own head
  linear/orig      logistic regression on the encoder's original training data
  mlp/orig         same architecture as the original head, re-initialized, same data
  linear/circ      logistic regression on FRESH shallow composed circuits (k = 2..4,
                   independent generation seed, 0 overlap with the benchmark)
  mlp/circ         same, original head architecture
  linear/both      original data + fresh shallow circuits
Model selection (L2 strength, early stopping) uses validation splits only.

Usage: python scripts/head_retrain.py --runs maskaug_mpnn maskaug_gin
"""

from __future__ import annotations

import argparse
import collections
import copy
import json
import random
import statistics as st
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.probe_by_depth import fit_probe  # noqa: E402
from src.data import NUM_NODE_FEATURES, record_to_data  # noqa: E402
from src.models import model_from_config  # noqa: E402

GEN = ROOT / "data_new/generated"


@torch.no_grad()
def head_inputs(model, recs):
    """Batched capture of the head's input (frozen encoder)."""
    out = []
    captured = {}

    def grab(module, inputs, output):  # returns None: does not alter the forward pass
        captured["x"] = inputs[0].detach()

    hook = model.head[0].register_forward_hook(grab)
    for b in DataLoader([record_to_data(r) for r in recs], batch_size=256):
        model(b.x, b.edge_index, b.batch, b.candidate)
        out.append(captured["x"])
    hook.remove()
    return torch.cat(out)


def train_linear(Xtr, ytr, Xva, yva):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    best = None
    for l2 in (1e-3, 1e-2, 1e-1, 1.0):
        w, b = fit_probe((Xtr - mu) / sd, ytr, l2=l2)
        acc = ((((Xva - mu) / sd) @ w + b > 0).float() == yva).float().mean().item()
        if best is None or acc > best[0]:
            best = (acc, w, b)
    _, w, b = best
    return lambda X: ((X - mu) / sd) @ w + b


def train_mlp(template, Xtr, ytr, Xva, yva, seed):
    torch.manual_seed(seed)
    head = copy.deepcopy(template)
    for m in head.modules():
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    best, best_state, bad = -1.0, None, 0
    g = torch.Generator().manual_seed(seed)
    for _ in range(300):
        head.train()
        for idx in torch.randperm(len(Xtr), generator=g).split(256):
            opt.zero_grad()
            F.cross_entropy(head(Xtr[idx]), ytr[idx].long()).backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            acc = (head(Xva).argmax(-1).float() == yva).float().mean().item()
        if acc > best:
            best, best_state, bad = acc, copy.deepcopy(head.state_dict()), 0
        else:
            bad += 1
            if bad >= 25:
                break
    head.load_state_dict(best_state)
    head.eval()
    return lambda X: (lambda o: o[:, 1] - o[:, 0])(head(X).detach())


def scores(score_fn, X, recs):
    """acc, pair acc, ordering, mean score on negatives, per group."""
    s = score_fn(X).tolist()
    out = {}
    groups = collections.defaultdict(list)
    for i, r in enumerate(recs):
        groups[r["_group"]].append(i)
    for g, idx in groups.items():
        out[(g, "acc")] = st.mean(int(s[i] > 0) == recs[i]["label"] for i in idx)
        pairs = collections.defaultdict(dict)
        for i in idx:
            if recs[i].get("pair_id"):
                pairs[recs[i]["pair_id"]][recs[i]["label"]] = i
        full = [p for p in pairs.values() if len(p) == 2]
        if full:
            out[(g, "pair")] = st.mean(s[p[0]] <= 0 < s[p[1]] for p in full)
            out[(g, "order")] = st.mean(s[p[1]] > s[p[0]] for p in full)
            out[(g, "neg")] = st.mean(s[p[0]] for p in full)
    return out


def load(name, split=None, group=None):
    recs = json.loads((GEN / name).read_text())["records"]
    if split:
        recs = [r for r in recs if r.get("split") == split]
    for r in recs:
        r["_group"] = group(r) if callable(group) else group
    return recs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--train-data", default="mixed_masked_train.json")
    p.add_argument("--out", type=Path, default=ROOT / "results/head_retrain.md")
    args = p.parse_args()
    torch.set_num_threads(16)

    orig_tr = load(args.train_data, "train", "orig")
    orig_va = load(args.train_data, "val", "orig")
    circ = load("circuit_shallow_train_v2.json", group="circ")
    pids = sorted({r["pair_id"] for r in circ})
    random.Random(0).shuffle(pids)
    val_p = set(pids[: len(pids) // 5])
    circ_tr = [r for r in circ if r["pair_id"] not in val_p]
    circ_va = [r for r in circ if r["pair_id"] in val_p]
    bench = load("circuit_bench_v2.json", group=lambda r: "bench shallow" if r["n_components"] <= 4
                 else "deep " + r["family"].split("_")[1])
    stress = (load("chain_matched_eval.json", group="chains")
              + load("ood_suite.json", group=lambda r: r["family"])
              + load("../../data/generated/edge_medium.json", "test", "medium test"))
    evalset = bench + stress

    res = collections.defaultdict(lambda: collections.defaultdict(list))
    for run in args.runs:
        for ck_path in sorted((ROOT / "results" / run).glob("seed*/model.pt")):
            seed = int(ck_path.parent.name[4:])
            ck = torch.load(ck_path, weights_only=False)
            model = model_from_config(ck["config"], NUM_NODE_FEATURES)
            model.load_state_dict(ck["state_dict"])
            model.eval()
            X = {k: head_inputs(model, v) for k, v in
                 (("otr", orig_tr), ("ova", orig_va), ("ctr", circ_tr), ("cva", circ_va), ("ev", evalset))}
            y = {k: torch.tensor([float(r["label"]) for r in v]) for k, v in
                 (("otr", orig_tr), ("ova", orig_va), ("ctr", circ_tr), ("cva", circ_va))}
            heads = {
                "original": lambda Z, m=model: (lambda o: o[:, 1] - o[:, 0])(m.head(Z).detach()),
                "linear/orig": train_linear(X["otr"], y["otr"], X["ova"], y["ova"]),
                "mlp/orig": train_mlp(model.head, X["otr"], y["otr"], X["ova"], y["ova"], seed),
                "linear/circ": train_linear(X["ctr"], y["ctr"], X["cva"], y["cva"]),
                "mlp/circ": train_mlp(model.head, X["ctr"], y["ctr"], X["cva"], y["cva"], seed),
                "linear/both": train_linear(torch.cat([X["otr"], X["ctr"]]), torch.cat([y["otr"], y["ctr"]]),
                                            torch.cat([X["ova"], X["cva"]]), torch.cat([y["ova"], y["cva"]])),
            }
            for name, fn in heads.items():
                with torch.no_grad():
                    for k, v in scores(fn, X["ev"], evalset).items():
                        res[(run, name)][k].append(v)

    def cell(d, k):
        v = d.get(k, [])
        return f"{st.mean(v):.2f}±{st.pstdev(v):.2f}" if v else "–"

    lines = ["# Milestone 1A: retrained heads on frozen encoders\n",
             "Mean ± std over encoder seeds. pair = both members of a matched pair right (chance 0.25); "
             "order = redundant member scored higher. Test sets evaluated once.\n",
             "## Composed-circuit benchmark v2 (held out)\n",
             "| run | head | deep chain pair | deep chain order | deep chain neg score | deep tree pair | deep tree order "
             "| bench shallow pair |", "|---|---|---|---|---|---|---|---|"]
    for (run, name), d in res.items():
        lines.append(f"| {run} | {name} | {cell(d, ('deep chain', 'pair'))} | {cell(d, ('deep chain', 'order'))} | "
                     f"{cell(d, ('deep chain', 'neg'))} | {cell(d, ('deep tree', 'pair'))} | {cell(d, ('deep tree', 'order'))} | "
                     f"{cell(d, ('bench shallow', 'pair'))} |")
    lines += ["\n## Stress sets (accuracy): does the new head break anything?\n",
              "| run | head | medium test | chains | ood_chain | ood_large | lattice |", "|---|---|---|---|---|---|---|"]
    for (run, name), d in res.items():
        lines.append(f"| {run} | {name} | " + " | ".join(cell(d, (g, "acc")) for g in
                     ("medium test", "chains", "ood_chain", "ood_large", "lattice")) + " |")
    text = "\n".join(lines) + "\n"
    args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

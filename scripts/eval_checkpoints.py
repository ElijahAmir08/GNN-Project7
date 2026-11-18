"""Score saved checkpoints on evaluation-only datasets (no retraining).

For each run directory and each seed checkpoint, report on every --data file:
  acc            overall accuracy
  acc_indep      accuracy on label 0 (independent)
  acc_redund     accuracy on label 1 (redundant)
  pair_acc       fraction of base graphs whose candidates are ALL classified
                 correctly (matched-pair files put one candidate of each label
                 on the same base, so this is the strict shortcut-proof score)
  acc_no_leak    accuracy excluding examples whose base graph is isomorphic
                 to a base graph in the training split of --train-data
Files whose records carry several "family" values are reported per family.

Usage:
  python scripts/eval_checkpoints.py \
      --data data/generated/edge_medium.json data_new/generated/laman_matched_eval.json \
             data_new/generated/chain_matched_eval.json
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics as st
import sys
from pathlib import Path

import networkx as nx
import torch
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import NUM_NODE_FEATURES, record_to_data  # noqa: E402
from src.models import build_model, shortcut_features  # noqa: E402

DEFAULT_RUNS = ["features_edge_medium", "ladder_gcn_medium", "ladder_mpnn_medium",
                "gin_edge_medium", "ladder_gin_medium", "ladder_dsgin_medium"]


def graph_of(r) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(range(r["num_nodes"]))
    g.add_edges_from(map(tuple, r["edges"]))
    return g


def leaked_ids(records, train_records) -> set[str]:
    by_hash = collections.defaultdict(list)
    for r in {t["base_id"]: t for t in train_records}.values():
        g = graph_of(r)
        by_hash[nx.weisfeiler_lehman_graph_hash(g)].append(g)
    out = set()
    for r in records:
        g = graph_of(r)
        if any(nx.is_isomorphic(g, h) for h in by_hash.get(nx.weisfeiler_lehman_graph_hash(g), [])):
            out.add(r["graph_id"])
    return out


@torch.no_grad()
def predict(model, kind, records) -> list[int]:
    datas = [record_to_data(r) for r in records]
    if kind == "features":
        for d in datas:
            d.feats = shortcut_features(d)
    preds = []
    for b in DataLoader(datas, batch_size=256):
        out = model(b.feats) if kind == "features" else model(b.x, b.edge_index, b.batch, b.candidate)
        preds += out.argmax(-1).tolist()
    return preds


def scores(records, preds, leaks) -> dict:
    ok = [p == r["label"] for r, p in zip(records, preds)]
    by_base = collections.defaultdict(list)
    for r, c in zip(records, ok):
        by_base[r["base_id"]].append(c)
    clean = [c for r, c in zip(records, ok) if r["graph_id"] not in leaks]

    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    return {
        "acc": mean(ok),
        "acc_indep": mean([c for r, c in zip(records, ok) if r["label"] == 0]),
        "acc_redund": mean([c for r, c in zip(records, ok) if r["label"] == 1]),
        "pair_acc": mean([all(v) for v in by_base.values()]),
        "acc_no_leak": mean(clean),
        "n": len(records), "n_no_leak": len(clean),
    }


def fmt(vals):
    vals = [v for v in vals if v == v]
    if not vals:
        return "–"
    return f"{st.mean(vals):.3f}" if len(vals) == 1 else f"{st.mean(vals):.3f} ± {st.pstdev(vals):.3f}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, nargs="+", required=True)
    p.add_argument("--runs", nargs="+", default=DEFAULT_RUNS)
    p.add_argument("--train-data", type=Path, default=ROOT / "data/generated/edge_medium.json")
    p.add_argument("--out", type=Path, default=ROOT / "results/eval_checkpoints.md")
    args = p.parse_args()

    train_records = [r for r in json.loads(args.train_data.read_text())["records"] if r["split"] == "train"]
    datasets = {}
    for path in args.data:
        recs = [r for r in json.loads(path.read_text())["records"] if r.get("split", "test") == "test"]
        leaks = leaked_ids(recs, train_records)
        families = sorted({r["family"] for r in recs if "family" in r})
        if len(families) > 1:  # report multi-family files per family
            for fam in families:
                datasets[f"{path.stem}:{fam}"] = ([r for r in recs if r.get("family") == fam], leaks)
        else:
            datasets[path.stem] = (recs, leaks)

    lines = ["| model | eval set | n (no-leak n) | acc | acc independent | acc redundant | pair acc | acc excl. leaked |",
             "|---|---|---|---|---|---|---|---|"]
    raw = {}
    for run in args.runs:
        ckpts = sorted((ROOT / "results" / run).glob("seed*/model.pt"))
        for name, (recs, leaks) in datasets.items():
            per_seed = []
            for ck_path in ckpts:
                ck = torch.load(ck_path, weights_only=False)
                cfg = ck["config"]
                model = build_model(cfg["model"], NUM_NODE_FEATURES, cfg["hidden"], cfg["layers"], cfg["dropout"])
                model.load_state_dict(ck["state_dict"])
                model.eval()
                per_seed.append(scores(recs, predict(model, cfg["model"], recs), leaks))
            raw[f"{run}/{name}"] = per_seed
            s0 = per_seed[0]
            lines.append(f"| {run} | {name} | {s0['n']} ({s0['n_no_leak']}) | **{fmt([s['acc'] for s in per_seed])}** "
                         f"| {fmt([s['acc_indep'] for s in per_seed])} | {fmt([s['acc_redund'] for s in per_seed])} "
                         f"| {fmt([s['pair_acc'] for s in per_seed])} | {fmt([s['acc_no_leak'] for s in per_seed])} |")
    text = "\n".join(lines) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    args.out.with_suffix(".json").write_text(json.dumps(raw, indent=2), encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

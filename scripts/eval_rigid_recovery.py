"""Experiment 3: how well does each model's representation encode rigid-component structure?

For each graph, up to --pairs non-edge vertex pairs are sampled (seeded) and labelled
R_ij = 1 iff i, j are rigidly linked (exact oracle). Pair features from the FROZEN
node embeddings of the candidate-marked forward pass: [h_i + h_j, h_i * h_j].
A logistic probe is fit on the fresh shallow circuits (k = 2..4) and evaluated on the
deep v2 benchmark graphs (k >= 6) by AUROC. Models trained with the rigid auxiliary
target (C) also report their own head's AUROC. Mean ± std over seeds.

Usage: python scripts/eval_rigid_recovery.py --runs circ_mpnn exp3B_mpnn exp3C_mpnn exp3D_mpnn
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import statistics as st
import sys
from itertools import combinations
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.probe_by_depth import fit_probe  # noqa: E402
from src.data import NUM_NODE_FEATURES, record_to_data  # noqa: E402
from src.models import model_from_config  # noqa: E402
from src.oracle import rigidity_rank  # noqa: E402

GEN = ROOT / "data_new/generated"


def labelled_pairs(recs, n_pairs, seed):
    out = []
    for idx, r in enumerate(recs):
        rng = random.Random(seed * 1_000_003 + idx)
        n, edges = r["num_nodes"], [tuple(e) for e in r["edges"]]
        present = set(edges)
        non = [p for p in combinations(range(n), 2) if p not in present]
        pairs = rng.sample(non, min(n_pairs, len(non)))
        base = rigidity_rank(n, edges)
        out.append((pairs, [int(rigidity_rank(n, edges + [p]) == base) for p in pairs]))
    return out


def auroc(scores, labels):
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).float().mean()
    eq = (pos[:, None] == neg[None, :]).float().mean()
    return (gt + 0.5 * eq).item()


@torch.no_grad()
def pair_features(model, recs, pairs, use_head=False):
    feats, head_scores = [], []
    for r, (ps, _) in zip(recs, pairs):
        d = record_to_data(r)
        model(d.x, d.edge_index, None, d.candidate)
        h = model._last_h
        if ps:
            idx = torch.tensor(ps).t()
            feats.append(torch.cat([h[idx[0]] + h[idx[1]], h[idx[0]] * h[idx[1]]], dim=-1))
            if use_head:
                head_scores.append(model.aux_logits(idx))
    return torch.cat(feats), (torch.cat(head_scores) if use_head else None)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--pairs", type=int, default=24)
    p.add_argument("--out", type=Path, default=ROOT / "results/rigid_recovery.md")
    args = p.parse_args()
    torch.set_num_threads(16)
    shallow = json.loads((GEN / "circuit_shallow_train_v2.json").read_text())["records"]
    deep = [r for r in json.loads((GEN / "circuit_bench_v2.json").read_text())["records"] if r["n_components"] >= 6]
    sp, dp = labelled_pairs(shallow, args.pairs, 1), labelled_pairs(deep, args.pairs, 2)
    ys = torch.tensor([lab for _, ls in sp for lab in ls], dtype=torch.float)
    yd = torch.tensor([lab for _, ls in dp for lab in ls])

    res = collections.defaultdict(lambda: collections.defaultdict(list))
    for run in args.runs:
        for ck_path in sorted((ROOT / "results" / run).glob("seed*/model.pt")):
            ck = torch.load(ck_path, weights_only=False)
            model = model_from_config(ck["config"], NUM_NODE_FEATURES)
            model.load_state_dict(ck["state_dict"])
            model.eval()
            has_head = hasattr(model, "aux_head") and "exp3C" in run
            Xs, _ = pair_features(model, shallow, sp)
            Xd, head = pair_features(model, deep, dp, use_head=has_head)
            mu, sd = Xs.mean(0), Xs.std(0) + 1e-6
            w, b = fit_probe((Xs - mu) / sd, ys)
            res[run]["probe"].append(auroc((Xd - mu) / sd @ w + b, yd))
            if has_head:
                res[run]["own head"].append(auroc(head, yd))

    lines = ["# Rigid-component recovery (AUROC on deep v2 benchmark pairs; probe fit on fresh shallow circuits)\n",
             f"Positive rate (rigidly linked) on deep pairs: {yd.float().mean():.1%}. Mean ± std over seeds.\n",
             "| run | linear probe AUROC | own auxiliary head AUROC |", "|---|---|---|"]
    for run, d in res.items():
        own = f"{st.mean(d['own head']):.3f}±{st.pstdev(d['own head']):.3f}" if d.get("own head") else "–"
        lines.append(f"| {run} | {st.mean(d['probe']):.3f}±{st.pstdev(d['probe']):.3f} | {own} |")
    text = "\n".join(lines) + "\n"
    args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

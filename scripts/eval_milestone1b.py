"""Milestone 1B evaluation: did retraining the representation pass the gate?

For each run (own trained head, no retraining), on held-out data evaluated once:
  deep v2 benchmark (k >= 6): chain / tree pair accuracy, ordering, negatives' score
  stress sets: medium test, chains, ood_chain, ood_large, lattice (accuracy)
  geometry: cosine between label directions fit (shared standardization) on the
            original training data, the shallow composed circuits, and ood_large
Mean ± std over seeds. 1-sum violations come from scripts/rule_tests.py.

Usage: python scripts/eval_milestone1b.py --runs maskaug_mpnn circ_mpnn circmean_mpnn ...
"""

from __future__ import annotations

import argparse
import collections
import statistics as st
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.head_retrain import head_inputs, load, scores  # noqa: E402
from scripts.probe_by_depth import fit_probe  # noqa: E402
from src.data import NUM_NODE_FEATURES  # noqa: E402
from src.models import model_from_config  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", type=Path, default=ROOT / "results/eval_milestone1b.md")
    args = p.parse_args()
    torch.set_num_threads(16)

    bench = load("circuit_bench_v2.json", group=lambda r: "bench shallow" if r["n_components"] <= 4
                 else "deep " + r["family"].split("_")[1])
    stress = (load("chain_matched_eval.json", group="chains")
              + load("ood_suite.json", group=lambda r: r["family"])
              + load("../../data/generated/edge_medium.json", "test", "medium test"))
    evalset = bench + stress
    orig = load("mixed_masked_train.json", "train", "o")
    circ = load("circuit_shallow_train_v2.json", group="c")
    ood_large = [r for r in stress if r["_group"] == "ood_large"]

    res = collections.defaultdict(lambda: collections.defaultdict(list))
    for run in args.runs:
        for ck_path in sorted((ROOT / "results" / run).glob("seed*/model.pt")):
            ck = torch.load(ck_path, weights_only=False)
            model = model_from_config(ck["config"], NUM_NODE_FEATURES)
            model.load_state_dict(ck["state_dict"])
            model.eval()
            X = head_inputs(model, evalset)
            with torch.no_grad():
                for k, v in scores(lambda Z, m=model: (lambda o: o[:, 1] - o[:, 0])(m.head(Z)), X, evalset).items():
                    res[run][k].append(v)
            Xo, Xc, Xl = head_inputs(model, orig), head_inputs(model, circ), head_inputs(model, ood_large)
            allX = torch.cat([Xo, Xc, Xl])
            mu, sd = allX.mean(0), allX.std(0) + 1e-6
            dirs = []
            for Xg, rs in ((Xo, orig), (Xc, circ), (Xl, ood_large)):
                w, _ = fit_probe((Xg - mu) / sd, torch.tensor([float(r["label"]) for r in rs]))
                dirs.append(w / w.norm())
            res[run]["cos orig-circ"].append((dirs[0] @ dirs[1]).item())
            res[run]["cos orig-ood_large"].append((dirs[0] @ dirs[2]).item())
            res[run]["cos circ-ood_large"].append((dirs[1] @ dirs[2]).item())

    def cell(d, k):
        v = d.get(k, [])
        return f"{st.mean(v):.2f}±{st.pstdev(v):.2f}" if v else "–"

    lines = ["# Milestone 1B: representation interventions (own heads, held-out data, mean ± std over seeds)\n",
             "| run | deep chain pair | deep chain order | deep chain neg score | deep tree pair | medium | chains | "
             "ood_chain | ood_large | lattice | cos orig–circ | cos orig–ood_large | cos circ–ood_large |",
             "|---|" + "---|" * 12]
    for run, d in res.items():
        lines.append(f"| {run} | {cell(d, ('deep chain', 'pair'))} | {cell(d, ('deep chain', 'order'))} | "
                     f"{cell(d, ('deep chain', 'neg'))} | {cell(d, ('deep tree', 'pair'))} | "
                     + " | ".join(cell(d, (g, "acc")) for g in ("medium test", "chains", "ood_chain", "ood_large", "lattice"))
                     + f" | {cell(d, 'cos orig-circ')} | {cell(d, 'cos orig-ood_large')} | {cell(d, 'cos circ-ood_large')} |")
    text = "\n".join(lines) + "\n"
    args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

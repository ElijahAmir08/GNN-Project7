"""Experiment 2b: is the label linearly decodable from frozen representations?

For each (shape, components k) and each trained checkpoint, collect the
classifier-input vectors of all benchmark examples and fit an L2-regularized
logistic-regression probe with 5-fold cross-validation over whole matched
pairs (both members of a pair always in the same fold). Compare probe
accuracy with the model's own accuracy:
  probe high, model low  -> information present, the model's readout misuses it
  probe low              -> information absent from the representation
Features are standardized inside each training fold.

Usage: python scripts/probe_by_depth.py --runs maskaug_mpnn maskaug_gin
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics as st
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.separation_by_depth import readout_vector  # noqa: E402
from src.data import NUM_NODE_FEATURES, record_to_data  # noqa: E402
from src.models import model_from_config  # noqa: E402


def fit_probe(X, y, l2=1e-2, steps=300):
    w = torch.zeros(X.size(1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(X @ w + b, y) + l2 * (w ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach(), b.detach()


def cv_probe(X, y, pair_ids, folds=5):
    uniq = sorted(set(pair_ids))
    g = torch.Generator().manual_seed(0)
    order = [uniq[i] for i in torch.randperm(len(uniq), generator=g).tolist()]
    correct = 0
    for f in range(folds):
        test_pairs = set(order[f::folds])
        te = torch.tensor([p in test_pairs for p in pair_ids])
        mu, sd = X[~te].mean(0), X[~te].std(0) + 1e-6
        w, b = fit_probe((X[~te] - mu) / sd, y[~te])
        pred = (((X[te] - mu) / sd) @ w + b) > 0
        correct += (pred.float() == y[te]).sum().item()
    return correct / len(y)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--data", type=Path, default=ROOT / "data_new/generated/circuit_bench.json")
    p.add_argument("--out", type=Path, default=ROOT / "results/probe_by_depth.md")
    args = p.parse_args()
    torch.set_num_threads(16)
    recs = json.loads(args.data.read_text())["records"]
    groups = collections.defaultdict(list)
    for r in recs:
        groups[(r["n_components"], r["family"].split("_")[1])].append(r)
    keys = sorted(groups)

    res = collections.defaultdict(lambda: collections.defaultdict(list))
    for run in args.runs:
        for ck_path in sorted((ROOT / "results" / run).glob("seed*/model.pt")):
            ck = torch.load(ck_path, weights_only=False)
            model = model_from_config(ck["config"], NUM_NODE_FEATURES)
            model.load_state_dict(ck["state_dict"])
            model.eval()
            for key, rs in groups.items():
                datas = [record_to_data(r) for r in rs]
                X = torch.stack([readout_vector(model, d) for d in datas])
                y = torch.tensor([float(r["label"]) for r in rs])
                with torch.no_grad():
                    own = [model(d.x, d.edge_index, None, d.candidate).argmax(-1).item() for d in datas]
                res[run][key + ("model",)].append(sum(int(a == r["label"]) for a, r in zip(own, rs)) / len(rs))
                res[run][key + ("probe",)].append(cv_probe(X, y, [r["pair_id"] for r in rs]))

    lines = ["# Linear probe vs model accuracy on the composed-circuit benchmark\n",
             "Accuracy (chance 0.50). probe = 5-fold logistic regression on frozen classifier inputs, folds by "
             "matched pair; model = the network's own prediction. Mean over seeds.\n"]
    for shape in ("chain", "tree"):
        ks = [k for k, s in keys if s == shape]
        lines.append(f"\n## {shape}\n")
        lines.append("| run | " + " | ".join(f"k={k} model / probe" for k in ks) + " |")
        lines.append("|---|" + "---|" * len(ks))
        for run in args.runs:
            lines.append(f"| {run} | " + " | ".join(
                f"{st.mean(res[run][(k, shape, 'model')]):.2f} / {st.mean(res[run][(k, shape, 'probe')]):.2f}" for k in ks) + " |")
    text = "\n".join(lines) + "\n"
    args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

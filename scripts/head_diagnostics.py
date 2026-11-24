"""Why does the trained classifier fail where a linear probe succeeds? (no training of GNNs)

All quantities use the exact vector the classifier head receives (its input),
on the composed-circuit benchmark, grouped as shallow (k <= 4), deep chain and
deep tree (k >= 6). Per model (mean over seeds):

D1 pair outcomes        both correct / both called redundant / both called independent /
                        reversed / one wrong
D2 ordering             head: P(L(h+) > L(h-)) where L = logit_redundant - logit_independent;
                        probe: P(w.(h+ - h-) > 0) and mean signed margin, w fit on shallow data
D3 bias and scale       mean head logit difference on positives / negatives; mean ||h||
D4 probe transfer       probe fit on shallow chains vs shallow trees, accuracy on deep chains

If ordering stays high while pair accuracy falls, the head's threshold/calibration
shifts with depth (a fixable readout problem). If ordering also falls, the label
direction itself is lost or rotated.

Usage: python scripts/head_diagnostics.py --runs maskaug_mpnn maskaug_gin ...
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
from scripts.probe_by_depth import fit_probe  # noqa: E402
from scripts.separation_by_depth import readout_vector  # noqa: E402
from src.data import NUM_NODE_FEATURES, record_to_data  # noqa: E402
from src.models import model_from_config  # noqa: E402


def group_of(r):
    if r["n_components"] <= 4:
        return "shallow"
    return "deep " + r["family"].split("_")[1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--data", type=Path, default=ROOT / "data_new/generated/circuit_bench.json")
    p.add_argument("--out", type=Path, default=ROOT / "results/head_diagnostics.md")
    args = p.parse_args()
    torch.set_num_threads(16)
    recs = json.loads(args.data.read_text())["records"]
    datas = [record_to_data(r) for r in recs]
    groups = ["shallow", "deep chain", "deep tree"]

    res = collections.defaultdict(lambda: collections.defaultdict(list))
    for run in args.runs:
        for ck_path in sorted((ROOT / "results" / run).glob("seed*/model.pt")):
            ck = torch.load(ck_path, weights_only=False)
            model = model_from_config(ck["config"], NUM_NODE_FEATURES)
            model.load_state_dict(ck["state_dict"])
            model.eval()
            H = torch.stack([readout_vector(model, d) for d in datas])
            with torch.no_grad():
                logits = model.head(H)
            L = (logits[:, 1] - logits[:, 0]).tolist()
            y = torch.tensor([float(r["label"]) for r in recs])
            sh = torch.tensor([r["n_components"] <= 4 for r in recs])
            mu, sd = H[sh].mean(0), H[sh].std(0) + 1e-6
            Z = (H - mu) / sd
            w, b = fit_probe(Z[sh], y[sh])
            # D4: probes fit on shallow chains only / shallow trees only
            sc = torch.tensor([r["n_components"] <= 4 and "chain" in r["family"] for r in recs])
            st_ = torch.tensor([r["n_components"] <= 4 and "tree" in r["family"] for r in recs])
            wc, bc = fit_probe(Z[sc], y[sc])
            wt, bt = fit_probe(Z[st_], y[st_])
            dc = torch.tensor([group_of(r) == "deep chain" for r in recs])
            res[(run, "deep chain")]["probe_from_shallow_chains"].append(
                (((Z[dc] @ wc + bc) > 0).float() == y[dc]).float().mean().item())
            res[(run, "deep chain")]["probe_from_shallow_trees"].append(
                (((Z[dc] @ wt + bt) > 0).float() == y[dc]).float().mean().item())

            pairs = collections.defaultdict(dict)
            for i, r in enumerate(recs):
                pairs[r["pair_id"]][r["label"]] = i
            for g in groups:
                idx = [i for i, r in enumerate(recs) if group_of(r) == g]
                pid = [pi for pi, pr in pairs.items() if group_of(recs[pr[0]]) == g]
                pred = [int(L[i] > 0) for i in range(len(recs))]
                d = res[(run, g)]
                d["acc"].append(st.mean(pred[i] == recs[i]["label"] for i in idx))
                cats = collections.Counter()
                order_head, order_probe, margin = [], [], []
                for pi in pid:
                    n_, p_ = pairs[pi][0], pairs[pi][1]
                    a, c = pred[n_], pred[p_]
                    cats["both correct" if (a, c) == (0, 1) else "both redundant" if (a, c) == (1, 1)
                         else "both independent" if (a, c) == (0, 0) else "reversed"] += 1
                    order_head.append(L[p_] > L[n_])
                    m = (w @ (Z[p_] - Z[n_])).item()
                    order_probe.append(m > 0)
                    margin.append(m)
                for c in ("both correct", "both redundant", "both independent", "reversed"):
                    d[c].append(cats[c] / len(pid))
                d["head_order"].append(st.mean(order_head))
                d["probe_order"].append(st.mean(order_probe))
                d["probe_margin"].append(st.median(margin))
                d["L_pos"].append(st.mean(L[pairs[pi][1]] for pi in pid))
                d["L_neg"].append(st.mean(L[pairs[pi][0]] for pi in pid))
                d["norm"].append(H[idx].norm(dim=1).mean().item())
                d["probe_acc"].append((((Z[idx] @ w + b) > 0).float() == y[idx]).float().mean().item())

    cols = [("acc", "head acc"), ("both correct", "pair: both right"), ("both redundant", "pair: both 'redundant'"),
            ("both independent", "pair: both 'independent'"), ("reversed", "pair: reversed"),
            ("head_order", "head orders pair right"), ("probe_order", "probe orders pair right"),
            ("probe_acc", "probe acc"), ("L_pos", "head logit diff, positives"), ("L_neg", "head logit diff, negatives"),
            ("norm", "‖h‖")]
    lines = ["# Classifier head vs linear probe on the composed-circuit benchmark\n",
             "Same input for both: the vector the classifier head receives. Probe fit on shallow (k ≤ 4) examples, "
             "fixed L2, never tuned on deep data. Mean over seeds.\n",
             "| run | group | " + " | ".join(c[1] for c in cols) + " |", "|---|---|" + "---|" * len(cols)]
    for run in args.runs:
        for g in groups:
            d = res[(run, g)]
            lines.append(f"| {run} | {g} | " + " | ".join(f"{st.mean(d[c]):.2f}" for c, _ in cols) + " |")
    lines += ["\n## D4: does a shallow-fit probe transfer by orientation or by family features?\n",
              "| run | probe fit on shallow chains → deep chains | probe fit on shallow trees → deep chains |", "|---|---|---|"]
    for run in args.runs:
        d = res[(run, "deep chain")]
        lines.append(f"| {run} | {st.mean(d['probe_from_shallow_chains']):.2f} | {st.mean(d['probe_from_shallow_trees']):.2f} |")
    text = "\n".join(lines) + "\n"
    args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

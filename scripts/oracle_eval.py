"""Faithful-to-the-model vs. correct-by-the-math, graded by an exact oracle.

Reads explain_scores*.json written by scripts/explain_prelim.py and grades
each explanation S (selected edges at budget k) on two independent axes:

  faithful   hard-mask sufficiency: the model, given ONLY the edges in S,
             still makes its original prediction
  valid      ground truth agrees: with only S kept, the true label of the
             candidate equals the model's prediction. For a "redundant"
             prediction this means S is a rigidity certificate (S alone
             proves dependence). For an "independent" prediction validity is
             almost always trivially true, so the 2x2 is reported for
             predictions of "redundant" only.

Examples are split by model correctness:
  TP  truth redundant, predicted redundant
  FP  truth independent, predicted redundant (no valid explanation can exist)
  FN  truth redundant, predicted independent
  TN  truth independent, predicted independent

Additional columns:
  cert_eff        (|C|-1) / shortest prefix of the ranking that is a valid
                  certificate (1.0 = finds a minimal certificate first;
                  truth-redundant examples only)
  model_ok_on_S   model prediction on S equals the true label on S: is the
                  model itself reliable on the masked graphs that fidelity
                  metrics feed it?
  model_ok_on_G-S same for the complement (deletion test)
  truth_nec       removing S changes the true label (exact necessity)

Budget k = |C| - 1 for truth-redundant examples, round(|E| / 2) otherwise.

Usage:
  python scripts/oracle_eval.py                       # all runs, all score files
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import NUM_NODE_FEATURES, record_to_data  # noqa: E402
from src.models import build_model  # noqa: E402
from src.oracle import truth_label  # noqa: E402
from src.orbits import orbit_average, orbit_select  # noqa: E402

RUNS = ["ladder_gcn_medium", "ladder_mpnn_medium", "gin_edge_medium", "ladder_gin_medium", "ladder_dsgin_medium"]
METHODS = ["oracle", "PGExplainer", "GNNExplainer", "random"]
VARIANTS = ["raw", "orbit_avg", "orbit_sel"]


def load_model(ck_path: Path):
    ck = torch.load(ck_path, weights_only=False)
    cfg = ck["config"]
    m = build_model(cfg["model"], NUM_NODE_FEATURES, cfg["hidden"], cfg["layers"], cfg["dropout"])
    m.load_state_dict(ck["state_dict"])
    return m.eval()


@torch.no_grad()
def model_pred(model, d, keep: np.ndarray) -> int:
    ei = d.edge_index[:, torch.from_numpy(np.repeat(keep, 2))]
    return model(d.x, ei, None, d.candidate).argmax(-1).item()


def ranking(scores, orbits, variant, k) -> np.ndarray:
    if variant == "raw":
        return np.argsort(-scores, kind="stable")
    avg = orbit_average(scores, orbits)
    if variant == "orbit_avg":
        return np.argsort(-avg, kind="stable")
    # orbit_sel: whole orbits in order of mean score; prefix of size >= k
    return orbit_select(scores, orbits, len(scores))


def selection(scores, orbits, variant, k) -> np.ndarray:
    if variant == "orbit_sel":
        return orbit_select(scores, orbits, k)
    return ranking(scores, orbits, variant, k)[:k]


def shortest_valid_prefix(n, edges, cand, order) -> int | None:
    """Validity is monotone in the prefix (supersets of certificates are
    certificates), so binary search."""
    if not truth_label(n, [edges[i] for i in order], cand):
        return None
    lo, hi = 1, len(order)
    while lo < hi:
        mid = (lo + hi) // 2
        if truth_label(n, [edges[i] for i in order[:mid]], cand):
            hi = mid
        else:
            lo = mid + 1
    return lo


def grade(model, ex, method, variant) -> dict:
    r = ex["record"]
    n, cand = r["num_nodes"], tuple(r["candidate_edge"])
    edges = [tuple(e) for e in r["edges"]]
    E = len(edges)
    d = record_to_data(r)
    scores = np.asarray(ex["scores"][method])
    orbits = ex["orbits"]
    truth = r["label"]
    pred = model_pred(model, d, np.ones(E, dtype=bool))
    k = len(r["circuit_edges"]) - 1 if truth == 1 else max(1, round(E / 2))

    sel = selection(scores, orbits, variant, k)
    keep = np.zeros(E, dtype=bool)
    keep[sel] = True
    S = [edges[i] for i in sel]
    rest = [edges[i] for i in range(E) if not keep[i]]

    t_S, t_rest = truth_label(n, S, cand), truth_label(n, rest, cand)
    m_S, m_rest = model_pred(model, d, keep), model_pred(model, d, ~keep)
    out = {
        "group": {(1, 1): "TP", (0, 1): "FP", (1, 0): "FN", (0, 0): "TN"}[(truth, pred)],
        "faithful": m_S == pred,
        "valid": t_S == pred,
        "model_ok_on_S": m_S == t_S,
        "model_ok_on_rest": m_rest == t_rest,
        "truth_nec": t_rest != truth,
        "model_nec": m_rest != pred,
        "size": len(sel),
        "k": k,
    }
    if truth == 1:
        j = shortest_valid_prefix(n, edges, cand, ranking(scores, orbits, variant, k))
        out["cert_eff"] = (k / j) if j else 0.0
    return out


def pct(xs):
    return 100 * float(np.mean(xs)) if len(xs) else float("nan")


def fmt(vals, digits=1):
    vals = [v for v in vals if v == v]
    if not vals:
        return "–"
    if len(vals) == 1:
        return f"{vals[0]:.{digits}f}"
    return f"{st.mean(vals):.{digits}f} ± {st.pstdev(vals):.{digits}f}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", default=RUNS)
    p.add_argument("--datasets", nargs="+", default=["", "_chain_matched_eval"],
                   help="score-file suffixes: '' = edge_medium test, '_chain_matched_eval' = chains")
    p.add_argument("--out", type=Path, default=ROOT / "results/oracle_tables.md")
    args = p.parse_args()

    # results[(dataset, run, method, variant)] -> list over seeds of per-seed aggregates
    agg = collections.defaultdict(list)
    counts = collections.defaultdict(list)
    for ds in args.datasets:
        for run in args.runs:
            for seed_dir in sorted((ROOT / "results" / run).glob("seed*")):
                f = seed_dir / f"explain_scores{ds}.json"
                if not f.exists():
                    continue
                model = load_model(seed_dir / "model.pt")
                examples = json.loads(f.read_text())["examples"]
                for method in METHODS:
                    for variant in VARIANTS:
                        rows = [grade(model, ex, method, variant) for ex in examples]
                        g = collections.defaultdict(list)
                        for row in rows:
                            g[row["group"]].append(row)
                        pred_red = g["TP"] + g["FP"]
                        cells = {
                            "FV": pct([x["faithful"] and x["valid"] for x in pred_red]),
                            "F_notV": pct([x["faithful"] and not x["valid"] for x in pred_red]),
                            "V_notF": pct([x["valid"] and not x["faithful"] for x in pred_red]),
                            "neither": pct([not x["faithful"] and not x["valid"] for x in pred_red]),
                        }
                        stats = {
                            **cells,
                            "TP_FV": pct([x["faithful"] and x["valid"] for x in g["TP"]]),
                            "FP_faithful": pct([x["faithful"] for x in g["FP"]]),
                            "FN_faithful": pct([x["faithful"] for x in g["FN"]]),
                            "cert_eff": float(np.mean([x["cert_eff"] for x in rows if "cert_eff" in x] or [np.nan])),
                            "truth_nec": pct([x["truth_nec"] for x in rows if x["group"] in ("TP", "FN")]),
                            "model_nec": pct([x["model_nec"] for x in rows]),
                            "model_ok_on_S": pct([x["model_ok_on_S"] for x in rows]),
                            "model_ok_on_rest": pct([x["model_ok_on_rest"] for x in rows]),
                            "overshoot": float(np.mean([x["size"] - x["k"] for x in rows])),
                        }
                        agg[(ds, run, method, variant)].append(stats)
                        if method == "oracle" and variant == "raw":
                            counts[(ds, run)].append({k: len(v) for k, v in g.items()})

    name = {"": "edge_medium test (n 6–10)", "_chain_matched_eval": "Elijah chains (n 14–16)"}
    out = ["# Oracle-graded explanations: faithful to the model vs. correct by the math\n",
           "Generated by `scripts/oracle_eval.py`; mean ± std over checkpoint seeds. "
           "F = faithful (model keeps its prediction on S alone, hard mask); V = valid (exact rigidity oracle "
           "agrees with the prediction on S alone; for 'redundant' predictions S is a certificate). "
           "2×2 cells are over examples the model predicts redundant (TP + FP). Budget k = |C| − 1 if truly "
           "redundant, else ⌊|E|/2⌉.\n"]
    for ds in args.datasets:
        out.append(f"\n## {name.get(ds, ds)}\n")
        out.append("Prediction outcome counts per seed (TP / FP / FN / TN):\n")
        for run in args.runs:
            cs = counts.get((ds, run))
            if cs:
                out.append(f"- {run}: " + "; ".join(f"{c.get('TP', 0)}/{c.get('FP', 0)}/{c.get('FN', 0)}/{c.get('TN', 0)}" for c in cs))
        out.append("\n### 2×2 (raw explanations)\n")
        out.append("| model | explainer | F∧V % | F∧¬V % | V∧¬F % | neither % | TP: F∧V % | FP: faithful % | FN: faithful % | cert. efficiency | truth-necessary % | model-necessary % | model correct on S % | model correct on G−S % |")
        out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for run in args.runs:
            for method in METHODS:
                s = agg.get((ds, run, method, "raw"))
                if not s:
                    continue
                col = lambda key, d=1: fmt([x[key] for x in s], d)  # noqa: E731
                out.append(f"| {run} | {method} | **{col('FV')}** | {col('F_notV')} | {col('V_notF')} | {col('neither')} "
                           f"| {col('TP_FV')} | {col('FP_faithful')} | {col('FN_faithful')} | {col('cert_eff', 2)} "
                           f"| {col('truth_nec')} | {col('model_nec')} | {col('model_ok_on_S')} | {col('model_ok_on_rest')} |")
        out.append("\n### Symmetry corrections (F∧V % and certificate efficiency)\n")
        out.append("| model | explainer | F∧V raw | F∧V orbit_avg | F∧V orbit_sel | cert. eff. raw | cert. eff. orbit_avg | cert. eff. orbit_sel | orbit_sel overshoot |")
        out.append("|---|---|---|---|---|---|---|---|---|")
        for run in args.runs:
            for method in ("PGExplainer", "GNNExplainer"):
                ss = {v: agg.get((ds, run, method, v)) for v in VARIANTS}
                if not all(ss.values()):
                    continue
                out.append(f"| {run} | {method} | " + " | ".join(fmt([x["FV"] for x in ss[v]]) for v in VARIANTS)
                           + " | " + " | ".join(fmt([x["cert_eff"] for x in ss[v]], 2) for v in VARIANTS)
                           + f" | {fmt([x['overshoot'] for x in ss['orbit_sel']], 2)} |")
    text = "\n".join(out) + "\n"
    args.out.write_text(text, encoding="utf-8")
    args.out.with_suffix(".json").write_text(json.dumps({"|".join(k): v for k, v in agg.items()}, indent=1))
    print(text)


if __name__ == "__main__":
    main()

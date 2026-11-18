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

Graded metrics (truth-redundant examples; partial credit for near misses):
  recall@k        fraction of a known certificate inside the top-k edges
                  (best over record["certificates"] when several exist)
  valid@k+1/+2    a valid certificate exists within 1 / 2 edges of slack
  slack_score     (|E| - j) / (|E| - k), j = shortest valid prefix:
                  1 = minimal certificate first, 0 = needs every edge.
                  Unlike cert_eff it does not reward random rankings when
                  circuits cover most edges (random ~ 1 / (k + 1)).
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
from src.certify import OracleCounter, alternatives, certify, select_faithful, symmetric_set  # noqa: E402
from src.oracle import truth_label  # noqa: E402
from src.orbits import orbit_average, orbit_select  # noqa: E402

RUNS = ["ladder_gcn_medium", "ladder_mpnn_medium", "gin_edge_medium", "ladder_gin_medium", "ladder_dsgin_medium"]
METHODS = ["oracle", "PGExplainer", "GNNExplainer", "EdgeGrad", "random"]  # those present in a dump
VARIANTS = ["raw", "orbit_avg", "orbit_sel", "cert", "cert_faith"]
CERT_VARIANTS = ("cert", "cert_faith")


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


@torch.no_grad()
def model_prob(model, d, keep: np.ndarray, cls: int) -> float:
    ei = d.edge_index[:, torch.from_numpy(np.repeat(keep, 2))]
    return torch.softmax(model(d.x, ei, None, d.candidate), -1)[0, cls].item()


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

    extra = {}
    if variant in CERT_VARIANTS:
        # Certificate-constrained: ranking -> minimal valid certificate; for
        # cert_faith, pick the most model-faithful of several alternatives.
        oc = OracleCounter(n, edges, cand)
        cert = certify(oc, scores)
        if cert is None:  # candidate not redundant: no certificate exists
            sel = np.argsort(-scores, kind="stable")[:k]
            extra["fallback"] = True
        else:
            extra["fallback"] = False
            certs = [cert]
            if variant == "cert_faith":
                def prob(idx):
                    keep_c = np.zeros(E, dtype=bool)
                    keep_c[idx] = True
                    return model_prob(model, d, keep_c, pred)
                certs = alternatives(oc, scores, cert)
                cert, _ = select_faithful(certs, prob)
                if any(len(o) > 1 for o in orbits):
                    sym = symmetric_set(n, edges, cand, cert, certs, orbits)
                    members_faithful = []
                    for mem in sym["members"]:
                        keep_m = np.zeros(E, dtype=bool)
                        keep_m[mem] = True
                        members_faithful.append(model_pred(model, d, keep_m) == pred)
                    extra.update(set_complete=sym["complete"], set_members=len(sym["members"]),
                                 set_members_faithful=float(np.mean(members_faithful)))
            extra["n_certs_found"] = len(certs)
            sel = np.array(cert)
        extra["oracle_calls"] = oc.calls
    else:
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
        **extra,
    }
    if truth == 1:
        if variant in CERT_VARIANTS:  # the certificate first, then the rest by score
            rest_order = [i for i in np.argsort(-scores, kind="stable") if not keep[i]]
            order = np.array(list(sel) + rest_order)
        else:
            order = ranking(scores, orbits, variant, k)
        j = shortest_valid_prefix(n, edges, cand, order)
        out["cert_eff"] = (k / j) if j else 0.0
        out["slack"] = ((E - j) / (E - k) if E > k else 1.0) if j else 0.0
        out["valid_k1"] = bool(j) and j <= k + 1
        out["valid_k2"] = bool(j) and j <= k + 2
        certs = r.get("certificates") or [r["circuit_edges"]]
        top = {edges[i] for i in sel}
        best = 0.0
        for cert in certs:
            cset = {tuple(e) for e in cert} - {cand}
            best = max(best, len(top & cset) / len(cset))
        out["recall"] = best
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


def summarize(model, examples, method, variant) -> tuple[dict, dict]:
    """Grade every example and aggregate one (method, variant) cell."""
    rows = [grade(model, ex, method, variant) for ex in examples]
    g = collections.defaultdict(list)
    for row in rows:
        g[row["group"]].append(row)
    pred_red = g["TP"] + g["FP"]

    def mean_of(key):
        return float(np.mean([float(x[key]) for x in rows if key in x] or [np.nan]))

    stats = {
        "FV": pct([x["faithful"] and x["valid"] for x in pred_red]),
        "F_notV": pct([x["faithful"] and not x["valid"] for x in pred_red]),
        "V_notF": pct([x["valid"] and not x["faithful"] for x in pred_red]),
        "neither": pct([not x["faithful"] and not x["valid"] for x in pred_red]),
        "TP_FV": pct([x["faithful"] and x["valid"] for x in g["TP"]]),
        "FP_faithful": pct([x["faithful"] for x in g["FP"]]),
        "FN_faithful": pct([x["faithful"] for x in g["FN"]]),
        "cert_eff": mean_of("cert_eff"),
        "cert_eff_exact": float(np.mean([float(x["cert_eff"] >= 1.0) for x in rows if "cert_eff" in x] or [np.nan])),
        "truth_nec": pct([x["truth_nec"] for x in rows if x["group"] in ("TP", "FN")]),
        "model_nec": pct([x["model_nec"] for x in rows]),
        "model_ok_on_S": pct([x["model_ok_on_S"] for x in rows]),
        "model_ok_on_rest": pct([x["model_ok_on_rest"] for x in rows]),
        "overshoot": float(np.mean([x["size"] - x["k"] for x in rows])),
        **{key: mean_of(key) for key in ("recall", "slack", "valid_k1", "valid_k2", "oracle_calls",
                                          "n_certs_found", "fallback", "set_complete", "set_members",
                                          "set_members_faithful")},
    }
    return stats, {k: len(v) for k, v in g.items()}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", default=RUNS)
    p.add_argument("--datasets", nargs="+", default=["", "_chain_matched_eval", "_hard_eval"],
                   help="score-file suffixes: '' = edge_medium test, '_chain_matched_eval' = chains, '_hard_eval' = hard set (reported per family)")
    p.add_argument("--out", type=Path, default=ROOT / "results/oracle_tables.md")
    args = p.parse_args()

    # results[(dataset, run, method, variant)] -> list over seeds of per-seed aggregates
    agg = collections.defaultdict(list)
    counts = collections.defaultdict(list)
    ds_labels: list[str] = []  # dataset (and family, when records have one), in order seen
    for ds in args.datasets:
        for run in args.runs:
            for seed_dir in sorted((ROOT / "results" / run).glob("seed*")):
                f = seed_dir / f"explain_scores{ds}.json"
                if not f.exists():
                    continue
                model = load_model(seed_dir / "model.pt")
                by_family = collections.defaultdict(list)
                for ex in json.loads(f.read_text())["examples"]:
                    by_family[ex["record"].get("family")].append(ex)
                for fam in sorted(by_family, key=str):
                    dsl = ds if fam is None else f"{ds}:{fam}"
                    if dsl not in ds_labels:
                        ds_labels.append(dsl)
                    present = by_family[fam][0]["scores"]
                    for method in [m for m in METHODS if m in present]:
                        for variant in VARIANTS:
                            stats, group_counts = summarize(model, by_family[fam], method, variant)
                            agg[(dsl, run, method, variant)].append(stats)
                            if method == "oracle" and variant == "raw":
                                counts[(dsl, run)].append(group_counts)

    name = {"": "edge_medium test (n 6–10)", "_chain_matched_eval": "Elijah chains (n 14–16)",
            "_hard_eval:decoy": "Hard set: decoy family", "_hard_eval:multi": "Hard set: multi-certificate family",
            "_hard_eval:symmetric": "Hard set: symmetric family"}
    out = ["# Oracle-graded explanations: faithful to the model vs. correct by the math\n",
           "Generated by `scripts/oracle_eval.py`; mean ± std over checkpoint seeds. "
           "F = faithful (model keeps its prediction on S alone, hard mask); V = valid (exact rigidity oracle "
           "agrees with the prediction on S alone; for 'redundant' predictions S is a certificate). "
           "2×2 cells are over examples the model predicts redundant (TP + FP). Budget k = |C| − 1 if truly "
           "redundant, else ⌊|E|/2⌉.\n"]
    for ds in ds_labels:
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
                col = lambda key, d=1, s=s: fmt([x[key] for x in s], d)  # noqa: E731
                out.append(f"| {run} | {method} | **{col('FV')}** | {col('F_notV')} | {col('V_notF')} | {col('neither')} "
                           f"| {col('TP_FV')} | {col('FP_faithful')} | {col('FN_faithful')} | {col('cert_eff', 2)} "
                           f"| {col('truth_nec')} | {col('model_nec')} | {col('model_ok_on_S')} | {col('model_ok_on_rest')} |")
        out.append("\n### Graded explanation quality (truly redundant examples; raw explanations)\n")
        out.append("| model | explainer | circuit recall@k | valid@k % | valid@k+1 % | valid@k+2 % | slack score |")
        out.append("|---|---|---|---|---|---|---|")
        for run in args.runs:
            for method in METHODS:
                s = agg.get((ds, run, method, "raw"))
                if not s:
                    continue
                out.append(f"| {run} | {method} | {fmt([x['recall'] for x in s], 2)} "
                           f"| {fmt([100 * x['cert_eff_exact'] for x in s])} "
                           f"| {fmt([100 * x['valid_k1'] for x in s])} | {fmt([100 * x['valid_k2'] for x in s])} "
                           f"| {fmt([x['slack'] for x in s], 2)} |")
        out.append("\n### Symmetry corrections (F∧V % and certificate efficiency)\n")
        out.append("| model | explainer | F∧V raw | F∧V orbit_avg | F∧V orbit_sel | slack raw | slack orbit_avg | slack orbit_sel | orbit_sel overshoot |")
        out.append("|---|---|---|---|---|---|---|---|---|")
        sym_variants = ("raw", "orbit_avg", "orbit_sel")
        for run in args.runs:
            for method in ("PGExplainer", "GNNExplainer", "EdgeGrad"):
                ss = {v: agg.get((ds, run, method, v)) for v in sym_variants}
                if not all(ss.values()):
                    continue
                out.append(f"| {run} | {method} | " + " | ".join(fmt([x["FV"] for x in ss[v]]) for v in sym_variants)
                           + " | " + " | ".join(fmt([x["slack"] for x in ss[v]], 2) for v in sym_variants)
                           + f" | {fmt([x['overshoot'] for x in ss['orbit_sel']], 2)} |")

        out.append("\n### Certificate-constrained explainers (cert = ranking -> minimal valid certificate; "
                   "cert_faith = most model-faithful of up to 8 alternative certificates)\n")
        out.append("F∧V over examples the model predicts redundant. Validity is guaranteed whenever the candidate is "
                   "truly redundant, so the informative columns are faithfulness and the gap to raw top-k. "
                   "fallback = candidate not redundant (no certificate exists; raw top-k used).\n")
        out.append("| model | explainer | F∧V raw | F∧V cert | F∧V cert_faith | TP faithful: cert | TP faithful: cert_faith "
                   "| certificates found | oracle calls | fallback % |")
        out.append("|---|---|---|---|---|---|---|---|---|---|")
        for run in args.runs:
            for method in ("random", "PGExplainer", "GNNExplainer", "EdgeGrad"):
                ss = {v: agg.get((ds, run, method, v)) for v in ("raw", "cert", "cert_faith")}
                if not all(ss.values()):
                    continue
                out.append(f"| {run} | {method} | {fmt([x['FV'] for x in ss['raw']])} | {fmt([x['FV'] for x in ss['cert']])} "
                           f"| **{fmt([x['FV'] for x in ss['cert_faith']])}** | {fmt([x['TP_FV'] for x in ss['cert']])} "
                           f"| {fmt([x['TP_FV'] for x in ss['cert_faith']])} | {fmt([x['n_certs_found'] for x in ss['cert_faith']], 1)} "
                           f"| {fmt([x['oracle_calls'] for x in ss['cert_faith']], 0)} | {fmt([100 * x['fallback'] for x in ss['cert_faith']])} |")

        rows_set = [(run, method, agg.get((ds, run, method, "cert_faith"))) for run in args.runs
                    for method in ("random", "PGExplainer", "GNNExplainer", "EdgeGrad")]
        rows_set = [(r, m, s_) for r, m, s_ in rows_set if s_ and any(x["set_members"] == x["set_members"] for x in s_)]
        if rows_set:
            out.append("\n### Set-valued symmetric explanations (truly redundant examples with nontrivial orbits)\n")
            out.append("Each member is a minimal valid certificate; complete = the members' union equals the union of "
                       "the edge orbits of the chosen certificate (symmetry-closed). Compare with whole-orbit "
                       "selection (orbit_sel), which is symmetric but not minimal.\n")
            out.append("| model | explainer | set complete % | members per set | members faithful % | orbit_sel overshoot | orbit_sel F∧V |")
            out.append("|---|---|---|---|---|---|---|")
            for run, method, s_ in rows_set:
                osel = agg.get((ds, run, method, "orbit_sel")) or []
                out.append(f"| {run} | {method} | {fmt([100 * x['set_complete'] for x in s_])} | {fmt([x['set_members'] for x in s_], 2)} "
                           f"| {fmt([100 * x['set_members_faithful'] for x in s_])} | {fmt([x['overshoot'] for x in osel], 2)} "
                           f"| {fmt([x['FV'] for x in osel])} |")
    text = "\n".join(out) + "\n"
    args.out.write_text(text, encoding="utf-8")
    args.out.with_suffix(".json").write_text(json.dumps({"|".join(k): v for k, v in agg.items()}, indent=1))
    print(text)


if __name__ == "__main__":
    main()

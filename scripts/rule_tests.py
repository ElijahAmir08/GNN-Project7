"""Metamorphic rule tests: does the model obey laws the true rule ALWAYS obeys?

No labels are needed beyond the source examples; each law is a mathematical
fact about generic 2D rigidity (the rigidity matroid):

  add_edges   Adding bars never destroys redundancy. For a truly redundant
              example G, add r random bars (never the candidate): the
              candidate stays redundant. Violation = the model predicted
              redundant on G but independent on G+. (Never trained on: the
              masked augmentation only ever REMOVES edges.)
  delete      Removing bars never creates redundancy. For a truly
              independent G, delete r random bars: still independent.
              Violation = predicted independent on G but redundant on G-.
              (Aligned with masked augmentation, which trains on subsets.)
  one_sum     Attaching any graph at a single vertex never changes the label
              (1-sums are direct sums in the rigidity matroid). Attach a
              random Laman graph of s vertices at one non-candidate vertex;
              violation = the prediction changes. s up to 20 exceeds the
              decoys seen in training (5-7 vertices).

Every perturbed graph is also re-labeled by the exact oracle as a sanity check
(the laws must hold; an assertion fails otherwise).

Usage:
  python scripts/rule_tests.py --runs mixed_gin maskaug_gin ... --out results/rule_tests.md
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "4")

import argparse  # noqa: E402
import collections  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import statistics as st  # noqa: E402
import sys  # noqa: E402
from itertools import combinations  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import NUM_NODE_FEATURES, record_to_data  # noqa: E402
from src.laman import random_laman_graph  # noqa: E402
from src.models import model_from_config  # noqa: E402
from src.oracle import truth_label  # noqa: E402

TESTS = [("add_edges", 2), ("add_edges", 6), ("delete", 2), ("delete", 6),
         ("one_sum", 8), ("one_sum", 20)]


def perturb(rec, test, size, rng):
    n, cand = rec["num_nodes"], tuple(rec["candidate_edge"])
    edges = [tuple(e) for e in rec["edges"]]
    if test == "add_edges":
        present = set(edges) | {cand}
        non = [e for e in combinations(range(n), 2) if e not in present]
        if len(non) < size:
            return None
        edges = sorted(set(edges) | set(rng.sample(non, size)))
    elif test == "delete":
        if len(edges) <= size:
            return None
        edges = sorted(rng.sample(edges, len(edges) - size))
    else:  # one_sum
        anchor = rng.choice([w for w in range(n) if w not in cand])
        mod = random_laman_graph(size, rng)
        mapping = {0: anchor, **{i: n + i - 1 for i in range(1, size)}}
        edges = sorted(set(edges) | {tuple(sorted((mapping[a], mapping[b]))) for a, b in mod})
        n = n + size - 1
    out = dict(rec, num_nodes=n, edges=[list(e) for e in edges], circuit_edges=None, certificates=None)
    expected = rec["label"]
    assert truth_label(n, edges, cand) == expected, (test, rec["graph_id"])  # the law holds
    return out


def build_cases(records, reps, seed):
    rng = random.Random(seed)
    cases = []  # (test, size, source_index, perturbed record)
    for test, size in TESTS:
        for i, rec in enumerate(records):
            if test == "add_edges" and rec["label"] != 1:
                continue
            if test == "delete" and rec["label"] != 0:
                continue
            for _ in range(reps):
                pr = perturb(rec, test, size, rng)
                if pr is not None:
                    cases.append((test, size, i, pr))
    return cases


@torch.no_grad()
def predict(model, recs):
    preds = []
    for b in DataLoader([record_to_data(r) for r in recs], batch_size=256):
        preds += model(b.x, b.edge_index, b.batch, b.candidate).argmax(-1).tolist()
    return preds


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--data", type=Path, nargs="+", default=[
        ROOT / "data_new/generated/hard_eval.json", ROOT / "data_new/generated/chain_matched_eval.json",
        ROOT / "data_new/generated/ood_suite.json"])
    p.add_argument("--reps", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=ROOT / "results/rule_tests.md")
    args = p.parse_args()
    torch.set_num_threads(8)

    records = [r for path in args.data for r in json.loads(path.read_text())["records"] if r.get("split") == "test"]
    cases = build_cases(records, args.reps, args.seed)
    print(f"{len(records)} source examples, {len(cases)} perturbed graphs (all laws verified by the oracle)")

    # results[run][(test, size)] -> list over seeds of violation rate
    results = collections.defaultdict(lambda: collections.defaultdict(list))
    for run in args.runs:
        for ck_path in sorted((ROOT / "results" / run).glob("seed*/model.pt")):
            ck = torch.load(ck_path, weights_only=False)
            cfg = ck["config"]
            model = model_from_config(cfg, NUM_NODE_FEATURES)
            model.load_state_dict(ck["state_dict"])
            model.eval()
            base = predict(model, records)
            pert = predict(model, [c[3] for c in cases])
            tally = collections.defaultdict(lambda: [0, 0])
            for (test, size, i, _), y in zip(cases, pert):
                if test == "add_edges" and base[i] != 1:
                    continue  # law is about keeping a redundant prediction
                if test == "delete" and base[i] != 0:
                    continue
                tally[(test, size)][0] += int(y != base[i])
                tally[(test, size)][1] += 1
            for key, (bad, tot) in tally.items():
                results[run][key].append(100 * bad / tot if tot else float("nan"))

    lines = ["# Metamorphic rule tests (violation %, lower is better; 0 = obeys the law)\n",
             "add_edges: redundant stays redundant when bars are added (never trained on). "
             "delete: independent stays independent when bars are removed (aligned with masked training). "
             "one_sum: attaching a Laman graph at one vertex never changes the prediction. "
             "Number = bars added/removed or attached-graph size. Sources: hard_eval, chains, ood_suite.\n",
             "| run | " + " | ".join(f"{t} {s}" for t, s in TESTS) + " |",
             "|---|" + "---|" * len(TESTS)]
    for run in args.runs:
        cells = []
        for key in TESTS:
            v = [x for x in results[run].get(key, []) if x == x]
            cells.append(f"{st.mean(v):.1f} ± {st.pstdev(v):.1f}" if len(v) > 1 else (f"{v[0]:.1f}" if v else "–"))
        lines.append(f"| {run} | " + " | ".join(cells) + " |")
    text = "\n".join(lines) + "\n"
    args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

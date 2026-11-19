"""Step 0 diagnosis: WHERE do models fail on the stress sets? (no training)

For each example we compute:
  reach        how many message-passing rounds the certificate needs to reach
               the candidate: an edge (a, b) influences the candidate's
               endpoints after min(dist(a), dist(b)) + 1 rounds, where dist is
               the hop distance from {u, v}; reach = max over certificate
               edges. Our models run 4 rounds, so reach > 4 means part of the
               proof is outside what the model can see.
               Matched pairs: the independent member gets its redundant
               partner's reach (same graph, one edge different). Unpaired
               independent examples (lattice) are bucketed by "ecc", the
               farthest node from {u, v}.
  size         number of joints
  cut          whether the base graph has a cut vertex (articulation point)

Accuracy (mean over seeds) is reported per bucket for each run.

Usage: python scripts/error_analysis.py --runs mixed_gin maskaug_gin ...
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
from src.models import model_from_config  # noqa: E402

DATA = ["data_new/generated/ood_suite.json", "data_new/generated/chain_matched_eval.json",
        "data_new/generated/hard_eval.json", "data_new/generated/truss_eval.json"]
ROUNDS = 4


def hop_dist(rec):
    g = nx.Graph()
    g.add_nodes_from(range(rec["num_nodes"]))
    g.add_edges_from(map(tuple, rec["edges"]))
    u, v = rec["candidate_edge"]
    du = nx.single_source_shortest_path_length(g, u)
    dv = nx.single_source_shortest_path_length(g, v)
    inf = 10 ** 6
    return g, {w: min(du.get(w, inf), dv.get(w, inf)) for w in g.nodes}


def annotate(records):
    reach_of_pair = {}
    for r in records:
        g, dist = hop_dist(r)
        r["_size"] = r["num_nodes"]
        r["_cut"] = len(list(nx.articulation_points(g))) > 0
        finite = [d for d in dist.values() if d < 10 ** 6]
        r["_ecc"] = max(finite) if finite else 0
        if r["label"] == 1 and r.get("circuit_edges"):
            cand = tuple(r["candidate_edge"])
            body = [tuple(e) for e in r["circuit_edges"] if tuple(e) != cand]
            r["_reach"] = max(min(dist[a], dist[b]) + 1 for a, b in body)
            if r.get("pair_id"):
                reach_of_pair[r["pair_id"]] = r["_reach"]
    for r in records:
        if r["label"] == 0:
            r["_reach"] = reach_of_pair.get(r.get("pair_id"))


def bucket_reach(x):
    if x is None:
        return None
    return "≤2" if x <= 2 else ("3–4" if x <= 4 else ("5–6" if x <= 6 else "7+"))


def bucket_size(n):
    return "≤20" if n <= 20 else ("21–40" if n <= 40 else ("41–70" if n <= 70 else "71+"))


@torch.no_grad()
def predict(model, recs):
    out = []
    for b in DataLoader([record_to_data(r) for r in recs], batch_size=256):
        out += model(b.x, b.edge_index, b.batch, b.candidate).argmax(-1).tolist()
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", type=Path, default=ROOT / "results/error_analysis.md")
    args = p.parse_args()
    torch.set_num_threads(8)

    records = []
    for path in DATA:
        for r in json.loads((ROOT / path).read_text())["records"]:
            if r.get("split") == "test":
                r["_set"] = r.get("family") or Path(path).stem
                records.append(r)
    annotate(records)

    views = {
        "reach (redundant + matched independent), stress sets": (
            lambda r: bucket_reach(r["_reach"]) if r["_set"] in ("ood_large", "ood_chain", "chain_matched_eval") else None,
            ["≤2", "3–4", "5–6", "7+"]),
        "reach, redundant only, all sets": (
            lambda r: bucket_reach(r["_reach"]) if r["label"] == 1 else None, ["≤2", "3–4", "5–6", "7+"]),
        "graph size (joints), all sets": (lambda r: bucket_size(r["_size"]), ["≤20", "21–40", "41–70", "71+"]),
        "cut vertex in base graph, all sets": (lambda r: "cut" if r["_cut"] else "no cut", ["no cut", "cut"]),
    }
    counts = {name: collections.Counter(f(r) for r in records if f(r) is not None) for name, (f, _) in views.items()}

    # acc[run][view][bucket] -> per-seed accuracies
    acc = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
    for run in args.runs:
        for ck_path in sorted((ROOT / "results" / run).glob("seed*/model.pt")):
            ck = torch.load(ck_path, weights_only=False)
            model = model_from_config(ck["config"], NUM_NODE_FEATURES)
            model.load_state_dict(ck["state_dict"])
            model.eval()
            preds = predict(model, records)
            for name, (f, _) in views.items():
                tally = collections.defaultdict(lambda: [0, 0])
                for r, y in zip(records, preds):
                    b = f(r)
                    if b is not None:
                        tally[b][0] += int(y == r["label"])
                        tally[b][1] += 1
                for b, (ok, n) in tally.items():
                    acc[run][name][b].append(ok / n)

    lines = [f"# Where do models fail? (models run {ROUNDS} message-passing rounds)\n"]
    for name, (_, order) in views.items():
        lines.append(f"\n## Accuracy by {name}\n")
        lines.append("| run | " + " | ".join(f"{b} (n={counts[name][b]})" for b in order) + " |")
        lines.append("|---|" + "---|" * len(order))
        for run in args.runs:
            cells = []
            for b in order:
                v = acc[run][name].get(b, [])
                cells.append(f"{st.mean(v):.2f}" if v else "–")
            lines.append(f"| {run} | " + " | ".join(cells) + " |")
    text = "\n".join(lines) + "\n"
    args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

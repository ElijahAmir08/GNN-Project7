"""Experiment 3 training sets (see notes/preregistration.md).

From mixed_masked_circ_train.json (train/val records only; the test split is untouched):
  B  4 extra candidate-marked queries per graph (other non-edges, oracle-labelled)
  C  each graph gets up to 96 non-edge vertex pairs with R_ij = 1 iff i, j are rigidly
     linked (adding bar ij is redundant; exact oracle)
  D  the SAME pairs with target 1 iff hop distance <= 2 (rigidity-irrelevant)
Pairs are sampled with a per-graph seed, so C and D receive identical pairs.

Usage: python scripts/build_aux_data.py
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import collections  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402
from itertools import combinations  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import networkx as nx  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.oracle import rigidity_rank  # noqa: E402

GEN = ROOT / "data_new/generated"


def _graph_task(args):
    idx, rec, max_pairs, n_queries = args
    rng = random.Random(1_000_003 + idx)
    n = rec["num_nodes"]
    edges = [tuple(e) for e in rec["edges"]]
    present = set(edges)
    cand = tuple(rec["candidate_edge"])
    non = [p for p in combinations(range(n), 2) if p not in present]
    pairs = rng.sample(non, min(max_pairs, len(non)))
    base = rigidity_rank(n, edges)
    rigid = [int(rigidity_rank(n, edges + [p]) == base) for p in pairs]
    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from(edges)
    dist = dict(nx.all_pairs_shortest_path_length(g, cutoff=2))
    hop = [int(j in dist.get(i, {})) for i, j in pairs]
    others = [p for p in non if p != cand]
    queries = []
    for p in rng.sample(others, min(n_queries, len(others))):
        queries.append(dict(rec, graph_id=f"{rec['graph_id']}_q{p[0]}-{p[1]}", candidate_edge=list(p),
                            label=int(rigidity_rank(n, edges + [p]) == base), circuit_edges=None,
                            certificates=None, family=f"{rec.get('family', 'base')}_query"))
    return [list(p) for p in pairs], rigid, hop, queries


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=GEN / "mixed_masked_circ_train.json")
    p.add_argument("--max-pairs", type=int, default=96)
    p.add_argument("--queries", type=int, default=4)
    p.add_argument("--workers", type=int, default=48)
    args = p.parse_args()

    blob = json.loads(args.data.read_text())
    recs = blob["records"]
    work = [(i, r, args.max_pairs, args.queries) for i, r in enumerate(recs) if r["split"] in ("train", "val")]
    with Pool(args.workers) as pool:
        out = pool.map(_graph_task, work, chunksize=32)

    rec_c = [dict(r) for r in recs]
    rec_d = [dict(r) for r in recs]
    extra = []
    stats = collections.Counter()
    for (i, _, _, _), (pairs, rigid, hop, queries) in zip(work, out):
        rec_c[i]["aux_pairs"], rec_c[i]["aux_labels"] = pairs, rigid
        rec_d[i]["aux_pairs"], rec_d[i]["aux_labels"] = pairs, hop
        extra += queries
        stats["pairs"] += len(pairs)
        stats["rigid_pos"] += sum(rigid)
        stats["hop_pos"] += sum(hop)
        stats["query_pos"] += sum(q["label"] for q in queries)
        stats["queries"] += len(queries)
    for name, rs in (("exp3_B_train.json", recs + extra), ("exp3_C_train.json", rec_c), ("exp3_D_train.json", rec_d)):
        meta = dict(blob.get("meta", {}), experiment3=name, args={k: str(v) for k, v in vars(args).items()})
        (GEN / name).write_text(json.dumps({"meta": meta, "records": rs}) + "\n", encoding="utf-8")
    print(f"graphs with auxiliary pairs: {len(work)}; pairs {stats['pairs']} "
          f"(R_ij positive {stats['rigid_pos'] / stats['pairs']:.1%}, hop<=2 positive {stats['hop_pos'] / stats['pairs']:.1%})")
    print(f"B extra queries: {stats['queries']} (redundant {stats['query_pos'] / max(stats['queries'], 1):.1%})")


if __name__ == "__main__":
    main()

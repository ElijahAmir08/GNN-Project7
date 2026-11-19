"""Step 1: search for rigidity examples that NO 1-WL model can distinguish.

Two examples (G1, c1) and (G2, c2) are indistinguishable for every model in
our ladder up to GIN (gcn, mpnn, gin: message passing bounded by 1-WL, with
the candidate's endpoints marked and our readout [h_u + h_v, h_u * h_v,
sum-pool]) when color refinement with the endpoints marked gives the same
multiset of node colors and the same unordered pair of endpoint colors. If
such a pair has different true labels (exact oracle), every such model gets at
most 50% on the pair, a provable limitation rather than an empirical one.

We search where 1-WL is known to be weak: random d-regular graphs, circulant
graphs, prism / Moebius ladders, and products, with every non-edge as a
candidate. Collisions are found both within a graph (two candidates) and across
graphs of the same size.

Usage: python scripts/wl_collision_search.py --out data_new/generated/wl_collisions.json
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
from scripts.wl_diagnostic import wl_signature  # noqa: E402
from src.oracle import is_redundant  # noqa: E402


def graph_pool(seed: int, n_random: int) -> list[tuple[str, nx.Graph]]:
    rng = random.Random(seed)
    pool = []
    for n in range(8, 17):
        for k in range(2, n // 2 + 1):  # circulants C_n(1, k) and C_n(1, 2, k)
            pool.append((f"circ{n}_1_{k}", nx.circulant_graph(n, [1, k])))
            if k > 2:
                pool.append((f"circ{n}_1_2_{k}", nx.circulant_graph(n, [1, 2, k])))
    for m in range(4, 9):
        pool.append((f"prism{m}", nx.circular_ladder_graph(m)))
        pool.append((f"mobius{m}", nx.circulant_graph(2 * m, [1, m])))
        pool.append((f"ladder_tri{m}", nx.circular_ladder_graph(m)))
    for a in range(3, 6):
        for b in range(3, 6):
            pool.append((f"grid{a}x{b}", nx.convert_node_labels_to_integers(nx.grid_2d_graph(a, b))))
            pool.append((f"torus{a}x{b}", nx.convert_node_labels_to_integers(nx.grid_2d_graph(a, b, periodic=True))))
    for _ in range(n_random):
        d = rng.choice([3, 4, 5])
        n = rng.choice([x for x in range(8, 17) if (x * d) % 2 == 0])
        g = nx.random_regular_graph(d, n, seed=rng.randrange(10 ** 9))
        pool.append((f"reg{d}_{n}_{len(pool)}", g))
    out = []
    for name, g in pool:
        if nx.is_connected(g):
            out.append((name, nx.convert_node_labels_to_integers(g)))
    return out


def scan(item):
    name, g = item
    n = g.number_of_nodes()
    edges = sorted(tuple(sorted(e)) for e in g.edges())
    present = set(edges)
    rows = []
    for cand in combinations(range(n), 2):
        if cand in present:
            continue
        sig = wl_signature(n, [list(e) for e in edges], list(cand))
        rows.append((repr(sig), int(is_redundant(n, edges, cand)), name, n, edges, cand))
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--random-graphs", type=int, default=3000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=48)
    p.add_argument("--max-pairs", type=int, default=400)
    p.add_argument("--out", type=Path, default=ROOT / "data_new/generated/wl_collisions.json")
    args = p.parse_args()

    pool = graph_pool(args.seed, args.random_graphs)
    with Pool(args.workers) as workers:
        all_rows = [r for rows in workers.map(scan, pool, chunksize=8) for r in rows]

    by_sig = collections.defaultdict(lambda: {0: [], 1: []})
    for sig, lab, name, n, edges, cand in all_rows:
        by_sig[sig][lab].append((name, n, edges, cand))
    colliding = [s for s, d in by_sig.items() if d[0] and d[1]]
    within = [s for s in colliding if {x[0] for x in by_sig[s][0]} & {x[0] for x in by_sig[s][1]}]

    rng = random.Random(args.seed)
    records = []
    for i, s in enumerate(colliding[: args.max_pairs]):
        d = by_sig[s]
        neg, pos = rng.choice(d[0]), rng.choice(d[1])
        for lab, (name, n, edges, cand) in ((0, neg), (1, pos)):
            records.append({
                "graph_id": f"wl{i}_{lab}_{name}_c{cand[0]}-{cand[1]}", "base_id": f"wlpair{i}", "pair_id": f"wlpair{i}",
                "num_nodes": n, "edges": [list(e) for e in edges], "candidate_edge": list(cand),
                "label": lab, "circuit_edges": None, "split": "test", "family": "wl_collision", "source_graph": name,
            })
    meta = {"purpose": "1-WL-indistinguishable pairs with different labels", "graphs_scanned": len(pool),
            "candidates_scanned": len(all_rows), "colliding_signatures": len(colliding),
            "within_graph_collisions": len(within)}
    args.out.write_text(json.dumps({"meta": meta, "records": records}) + "\n", encoding="utf-8")

    fam = collections.Counter(by_sig[s][0][0][0].rstrip("0123456789_").split("_")[0] for s in colliding)
    print(f"graphs scanned: {len(pool)}  candidates: {len(all_rows)}")
    print(f"WL signatures with BOTH labels: {len(colliding)}  (within a single graph: {len(within)})")
    print("colliding signatures by source family:", dict(fam.most_common()))
    print(f"saved {len(records)} records ({len(records) // 2} pairs) to {args.out}")


if __name__ == "__main__":
    main()

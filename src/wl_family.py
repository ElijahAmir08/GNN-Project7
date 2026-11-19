"""An infinite family of rigidity questions no 1-WL GNN can answer.

wheel(k)       hub h joined to every vertex of one cycle C_{2k}. Wheels are
               generically rigid, so any added bar is REDUNDANT. Candidate:
               two antipodal rim vertices (distance k on the rim).
hinged(k)      hub h joined to every vertex of two disjoint cycles C_k. Each
               half (hub + C_k) is a rigid wheel, but the halves share only
               the hub, a cut vertex: they can rotate about it (a hinge), so
               a bar between the halves is INDEPENDENT. Candidate: one rim
               vertex in each half.

Both graphs have 2k + 1 vertices and 4k edges (hub degree 2k, rim degree 3).
With the candidate's endpoints marked, color refinement (1-WL) assigns the two
graphs identical color multisets and identical endpoint colors (C_{2k} with
two antipodal marks and two copies of C_k with one mark each refine
identically), so every message-passing GNN bounded by 1-WL, with any readout
built from node embeddings, outputs the same value on both: at most 50%
accuracy on each pair. k = 3 is the pair found by exhaustive search over all
graphs with up to 7 vertices (atlas #1008 / #1009).
"""

from __future__ import annotations


def wheel(k: int) -> tuple[int, list[tuple[int, int]], tuple[int, int]]:
    m = 2 * k
    hub = m
    edges = [(i, (i + 1) % m) for i in range(m)] + [(i, hub) for i in range(m)]
    return m + 1, sorted(tuple(sorted(e)) for e in edges), (0, k)


def hinged(k: int) -> tuple[int, list[tuple[int, int]], tuple[int, int]]:
    hub = 2 * k
    edges = []
    for start in (0, k):
        edges += [(start + i, start + (i + 1) % k) for i in range(k)]
        edges += [(start + i, hub) for i in range(k)]
    return 2 * k + 1, sorted(tuple(sorted(e)) for e in edges), (0, k)


def pair_records(k: int) -> list[dict]:
    out = []
    for lab, (name, fn) in ((0, ("hinged", hinged)), (1, ("wheel", wheel))):
        n, edges, cand = fn(k)
        out.append({
            "graph_id": f"{name}{k}", "base_id": f"wlfam{k}", "pair_id": f"wlfam{k}",
            "num_nodes": n, "edges": [list(e) for e in edges], "candidate_edge": list(cand),
            "label": lab, "circuit_edges": None, "split": "test", "family": "wl_family", "k": k,
        })
    return out

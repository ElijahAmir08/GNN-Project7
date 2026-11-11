"""Edge orbits under the candidate-stabilizer, plus orbit corrections.

Stand-in for Elijah's orbit utility (same interface: list of undirected-edge
index groups that partition range(E)).

Task-preserving symmetry = automorphisms of the base graph that map the
candidate pair {u, v} to itself (possibly swapping u and v).
"""

from __future__ import annotations

import networkx as nx
import numpy as np
from networkx.algorithms.isomorphism import GraphMatcher


def candidate_stabilizer_orbits(num_nodes: int, edges: list[tuple[int, int]], candidate: tuple[int, int]) -> list[list[int]]:
    g = nx.Graph()
    g.add_nodes_from(range(num_nodes))
    g.add_edges_from(edges)
    u, v = candidate
    # Encode the candidate as a node colour so matching respects it.
    for w in g.nodes:
        g.nodes[w]["c"] = int(w in (u, v))
    gm = GraphMatcher(g, g, node_match=lambda a, b: a["c"] == b["c"])

    index = {tuple(e): i for i, e in enumerate(edges)}
    parent = list(range(len(edges)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for phi in gm.isomorphisms_iter():
        for i, (a, b) in enumerate(edges):
            j = index[tuple(sorted((phi[a], phi[b])))]
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj

    groups: dict[int, list[int]] = {}
    for i in range(len(edges)):
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values())


def orbit_average(scores: np.ndarray, orbits: list[list[int]]) -> np.ndarray:
    out = scores.astype(float).copy()
    for orb in orbits:
        out[orb] = scores[orb].mean()
    return out


def topk(scores: np.ndarray, k: int) -> np.ndarray:
    """Plain top-k (ties broken by index, i.e. arbitrarily w.r.t. symmetry)."""
    return np.argsort(-scores, kind="stable")[:k]


def orbit_select(scores: np.ndarray, orbits: list[list[int]], k: int) -> np.ndarray:
    """Whole-orbit selection: add orbits by mean score until >= k edges.
    May overshoot k (reported, not hidden)."""
    ranked = sorted(orbits, key=lambda o: -scores[o].mean())
    chosen: list[int] = []
    for orb in ranked:
        if len(chosen) >= k:
            break
        chosen.extend(orb)
    return np.array(chosen, dtype=int)


def splits_an_orbit(selected: np.ndarray, orbits: list[list[int]]) -> bool:
    s = set(selected.tolist())
    return any(0 < len(s & set(o)) < len(o) for o in orbits)

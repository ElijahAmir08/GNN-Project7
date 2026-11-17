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


def _colored_graph(num_nodes, edges, candidate) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(range(num_nodes))
    g.add_edges_from(edges)
    u, v = candidate
    for w in g.nodes:
        g.nodes[w]["c"] = int(w in (u, v))  # candidate endpoints are coloured
    return g


def _refined_colors(g: nx.Graph) -> dict:
    """1-WL colour refinement (with candidate colours): a necessary condition
    for two vertices to lie in the same automorphism orbit."""
    colors = {w: hash(("c", g.nodes[w]["c"])) for w in g.nodes}
    for _ in range(g.number_of_nodes()):
        new = {w: hash((colors[w], tuple(sorted(colors[x] for x in g[w])))) for w in g.nodes}
        if len(set(new.values())) == len(set(colors.values())):
            return new
        colors = new
    return colors


def candidate_stabilizer_orbits(num_nodes: int, edges: list[tuple[int, int]], candidate: tuple[int, int]) -> list[list[int]]:
    """Partition edge indices into orbits of the automorphisms that fix the
    candidate pair setwise.

    Edges e, f share an orbit iff some such automorphism maps e to f. We test
    that directly (one VF2 first-match search per pair, with e's and f's
    endpoints marked) instead of enumerating the group: enumeration blows up
    on highly symmetric graphs (e.g. several identical bridges), first-match
    searches do not. Pairs with different refined colours are skipped.
    """
    g = _colored_graph(num_nodes, edges, candidate)
    colors = _refined_colors(g)
    key = [tuple(sorted((colors[a], colors[b]))) for a, b in edges]

    def maps(e, f) -> bool:
        g1, g2 = g.copy(), g.copy()
        for w in g.nodes:
            g1.nodes[w]["t"] = (g.nodes[w]["c"], int(w in e))
            g2.nodes[w]["t"] = (g.nodes[w]["c"], int(w in f))
        return GraphMatcher(g1, g2, node_match=lambda a, b: a["t"] == b["t"]).is_isomorphic()

    orbit_of = [-1] * len(edges)
    orbits: list[list[int]] = []
    for i in range(len(edges)):
        if orbit_of[i] >= 0:
            continue
        orbit_of[i] = len(orbits)
        orbit = [i]
        for j in range(i + 1, len(edges)):
            if orbit_of[j] < 0 and key[j] == key[i] and maps(edges[i], edges[j]):
                orbit_of[j] = orbit_of[i]
                orbit.append(j)
        orbits.append(orbit)
    return sorted(orbits)


def _orbits_by_enumeration(num_nodes, edges, candidate) -> list[list[int]]:
    """Reference implementation (enumerates all automorphisms). Exponential on
    highly symmetric graphs; kept only to test candidate_stabilizer_orbits."""
    g = _colored_graph(num_nodes, edges, candidate)
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
            ri, rj = find(i), find(index[tuple(sorted((phi[a], phi[b])))])
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

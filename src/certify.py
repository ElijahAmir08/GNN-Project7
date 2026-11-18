"""Certificate-constrained explanations (explanation-side fixes).

certify(...)        Any edge ranking -> a minimal valid rigidity certificate:
                    shortest valid prefix of the ranking (binary search; validity
                    is monotone), then greedy pruning from the lowest-scored edge
                    up. In the rigidity matroid this always yields a circuit
                    through the candidate (minus the candidate) in O(log|E| + j)
                    oracle calls, even for random scores, so validity itself is
                    NOT the contribution; which certificate is chosen is.

alternatives(...)   Further minimal certificates: re-run certify with one edge
                    of an already-found certificate forbidden, breadth-first.
                    Finds the other certificates when several exist (dependent
                    bases, symmetric copies).

select_faithful(...) Among valid minimal certificates, the one the model relies
                    on most: highest predicted-class probability when the model
                    sees only that certificate (hard mask).

symmetric_set(...)  Set-valued explanation for symmetric graphs: a minimal
                    certificate C together with every alternative that is an
                    image phi(C) under an automorphism fixing the candidate pair
                    (checked directly with an edge-marked VF2 test). Every
                    member is minimal and valid; the set is symmetry-closed when
                    `complete` is True (its edge union equals the union of the
                    edge orbits of C's edges).
"""

from __future__ import annotations

import networkx as nx
import numpy as np
from networkx.algorithms.isomorphism import GraphMatcher

from src.oracle import is_redundant


class OracleCounter:
    def __init__(self, n, edges, cand):
        self.n, self.edges, self.cand, self.calls = n, edges, tuple(cand), 0

    def valid(self, idx) -> bool:
        self.calls += 1
        return is_redundant(self.n, [self.edges[i] for i in idx], self.cand)


def certify(oc: OracleCounter, scores: np.ndarray, forbidden=frozenset()) -> list[int] | None:
    order = [i for i in np.argsort(-scores, kind="stable") if i not in forbidden]
    if not oc.valid(order):
        return None
    lo, hi = 1, len(order)
    while lo < hi:  # shortest valid prefix
        mid = (lo + hi) // 2
        if oc.valid(order[:mid]):
            hi = mid
        else:
            lo = mid + 1
    cert = list(order[:lo])
    for i in sorted(cert, key=lambda e: scores[e]):  # prune lowest-scored first
        trial = [e for e in cert if e != i]
        if oc.valid(trial):
            cert = trial
    return sorted(int(i) for i in cert)


def alternatives(oc: OracleCounter, scores: np.ndarray, first: list[int], max_certs: int = 8) -> list[list[int]]:
    found = [first]
    frontier = [frozenset()]
    seen_forbidden = {frozenset()}
    while frontier and len(found) < max_certs:
        nxt = []
        for forb in frontier:
            base = certify(oc, scores, forb) if forb else first
            if base is None:
                continue
            for e in base:
                f2 = forb | {e}
                if f2 in seen_forbidden:
                    continue
                seen_forbidden.add(f2)
                c = certify(oc, scores, f2)
                if c is not None and c not in found:
                    found.append(c)
                    nxt.append(f2)
                    if len(found) >= max_certs:
                        return found
        frontier = nxt
    return found


def select_faithful(certs: list[list[int]], prob_on_subset) -> tuple[list[int], float]:
    """prob_on_subset(idx) -> model probability of its prediction on idx only."""
    probs = [prob_on_subset(c) for c in certs]
    j = int(np.argmax(probs))
    return certs[j], probs[j]


def _is_image(n, edges, cand, c1: list[int], c2: list[int]) -> bool:
    """Is there an automorphism fixing {u, v} setwise that maps edge set c1 onto c2?"""
    def graph(mark):
        g = nx.Graph()
        g.add_nodes_from((w, {"c": int(w in cand)}) for w in range(n))
        g.add_edges_from((a, b, {"m": int(i in mark)}) for i, (a, b) in enumerate(edges))
        return g
    gm = GraphMatcher(graph(set(c1)), graph(set(c2)),
                      node_match=lambda a, b: a["c"] == b["c"], edge_match=lambda a, b: a["m"] == b["m"])
    return gm.is_isomorphic()


def symmetric_set(n, edges, cand, cert: list[int], certs: list[list[int]], orbits: list[list[int]]) -> dict:
    images = [c for c in certs if c == cert or _is_image(n, edges, tuple(cand), cert, c)]
    union = set().union(*map(set, images))
    orbit_of = {i: k for k, orb in enumerate(orbits) for i in orb}
    closure = set().union(*(set(orbits[orbit_of[i]]) for i in cert))
    return {"members": images, "complete": union == closure, "size_each": len(cert),
            "union_size": len(union)}

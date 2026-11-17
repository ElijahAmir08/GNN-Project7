"""Harder edge-rigidity examples: decoys, multiple certificates, symmetry.

Building block — a *bridge*: a random Laman graph B with a chosen non-edge
(a, b). Since B is rigid, (a, b) is redundant in B + (a, b), with a unique
fundamental circuit. We identify a -> u and b -> v, the candidate's endpoints,
so every bridge carries one certificate for the candidate (u, v).

Families (labels always come from the exact oracle in src/oracle.py):
  decoy      one bridge with one deleted edge (in its circuit -> independent,
             outside -> redundant) plus 1-2 Laman "decoy" modules attached at
             a single vertex. A 1-vertex join is a direct sum in the 2D
             rigidity matroid, so decoys never change labels or certificates,
             but they add dense rigid structure outside the circuit.
  multi      2-3 different bridges sharing only u, v, one deletion each. The
             base is dependent when 2+ bridges stay intact; the candidate is
             redundant iff some deletion missed its bridge's circuit. Both
             labels have identical vertex and edge counts.
  symmetric  2-3 identical copies of one bridge with the same deletion in
             each copy. The candidate's stabilizer permutes the copies, so
             minimal certificates come in symmetry orbits: no symmetric
             explanation is a minimal certificate (orbit selection must
             overshoot).

Each record follows the existing schema plus:
  certificates      distinct minimal certificates found (each includes the
                    candidate edge, like circuit_edges)
  circuit_edges     the smallest certificate (defines budget k = |C| - 1)
  base_dependent    whether the base edge set is dependent
  family            decoy | multi | symmetric
Certificates are found by greedy deletion from many random orders, seeded
with each bridge circuit; each result is a minimal certificate (a circuit
through the candidate), but the list is not guaranteed exhaustive.
"""

from __future__ import annotations

import random
from itertools import combinations

from src.laman import random_laman_graph
from src.oracle import is_redundant, rigidity_rank


def _laman_with_nonedge(n: int, rng: random.Random):
    edges = random_laman_graph(n, rng)
    present = set(edges)
    non = [e for e in combinations(range(n), 2) if e not in present]
    return edges, rng.choice(non)


def _circuit_in(n: int, edges: list[tuple[int, int]], cand) -> list[tuple[int, int]]:
    """Fundamental circuit (minus candidate) of cand in an independent edge set."""
    cur = list(edges)
    for e in list(cur):
        trial = [x for x in cur if x != e]
        if is_redundant(n, trial, cand):
            cur = trial
    return cur


def _minimal_certificate(n, edges, cand, order) -> list[tuple[int, int]]:
    cur = list(edges)
    for e in order:
        trial = [x for x in cur if x != e]
        if is_redundant(n, trial, cand):
            cur = trial
    return sorted(cur)


def _certificates(n, edges, cand, seeds, rng, tries=40) -> list[list[tuple[int, int]]]:
    found = set()
    for seed_set in seeds:  # start inside each known circuit for coverage
        inside = [e for e in edges if e in seed_set]
        outside = [e for e in edges if e not in seed_set]
        if is_redundant(n, inside, cand):
            found.add(tuple(_minimal_certificate(n, edges, cand, outside + inside)))
    for _ in range(tries):
        order = list(edges)
        rng.shuffle(order)
        found.add(tuple(_minimal_certificate(n, edges, cand, order)))
    return sorted((list(c) for c in found), key=len)


class _Builder:
    """Accumulates bridges and decoys on a shared vertex set; u=0, v=1."""

    def __init__(self):
        self.n = 2
        self.edges: set[tuple[int, int]] = set()

    def _fresh(self, k):
        ids = list(range(self.n, self.n + k))
        self.n += k
        return ids

    def add_bridge(self, b_edges, b_n, ab):
        a, b = ab
        others = [x for x in range(b_n) if x not in (a, b)]
        mapping = {a: 0, b: 1, **dict(zip(others, self._fresh(len(others))))}
        mapped = [tuple(sorted((mapping[x], mapping[y]))) for x, y in b_edges]
        self.edges.update(mapped)
        return mapped

    def add_decoy(self, rng, size):
        d_edges = random_laman_graph(size, rng)
        anchor = rng.randrange(2, self.n) if self.n > 2 else 0
        others = self._fresh(size - 1)
        mapping = {0: anchor, **{i: others[i - 1] for i in range(1, size)}}
        self.edges.update(tuple(sorted((mapping[x], mapping[y]))) for x, y in d_edges)


def _finish(builder, cand, bridge_circuits, family, rng, base_id):
    n = builder.n
    perm = list(range(n))
    rng.shuffle(perm)
    relabel = lambda e: tuple(sorted((perm[e[0]], perm[e[1]])))  # noqa: E731
    edges = sorted(relabel(e) for e in builder.edges)
    c = relabel(cand)
    label = int(is_redundant(n, edges, c))
    certs = []
    if label:
        seeds = [{relabel(e) for e in circ} for circ in bridge_circuits]
        certs = [sorted(cert + [c]) for cert in _certificates(n, edges, c, seeds, rng)]
    return {
        "graph_id": f"{base_id}_c{c[0]}-{c[1]}",
        "base_id": base_id,
        "num_nodes": n,
        "edges": [list(e) for e in edges],
        "candidate_edge": list(c),
        "label": label,
        "circuit_edges": [list(e) for e in certs[0]] if certs else None,
        "certificates": [[list(e) for e in cert] for cert in certs] if certs else None,
        "base_dependent": rigidity_rank(n, edges) < len(edges),
        "family": family,
    }


def make_example(family: str, rng: random.Random, base_id: str, want_label: int) -> dict:
    """Build one example of `family`, steering deletions toward `want_label`
    (the oracle label is still what gets recorded)."""
    builder = _Builder()
    circuits = []
    if family == "decoy":
        b_n = rng.randint(6, 8)
        b_edges, ab = _laman_with_nonedge(b_n, rng)
        mapped = builder.add_bridge(b_edges, b_n, ab)
        circ = _circuit_in(builder.n, mapped, (0, 1))
        pool = circ if want_label == 0 else [e for e in mapped if e not in circ] or circ
        builder.edges.discard(rng.choice(pool))
        circuits.append([e for e in circ if e in builder.edges])
        for _ in range(rng.randint(1, 2)):
            builder.add_decoy(rng, rng.randint(5, 7))
    elif family in ("multi", "symmetric"):
        m = rng.randint(2, 3)
        if family == "symmetric":
            b_n = rng.randint(5, 7)
            b_edges, ab = _laman_with_nonedge(b_n, rng)
            proto = builder.add_bridge(b_edges, b_n, ab)
            proto_circ = _circuit_in(builder.n, proto, (0, 1))
            pool = proto_circ if want_label == 0 else [e for e in proto if e not in proto_circ] or proto_circ
            drop_idx = proto.index(rng.choice(pool))
            bridges = [(proto, proto_circ)]
            for _ in range(m - 1):
                copy = builder.add_bridge(b_edges, b_n, ab)
                bridges.append((copy, [copy[proto.index(e)] for e in proto_circ]))
            drops = [br[drop_idx] for br, _ in bridges]
        else:
            bridges = []
            for _ in range(m):
                b_n = rng.randint(5, 7)
                b_edges, ab = _laman_with_nonedge(b_n, rng)
                br = builder.add_bridge(b_edges, b_n, ab)
                bridges.append((br, _circuit_in(builder.n, br, (0, 1))))
            # Every bridge loses exactly one edge (same size for both labels).
            # label 0: every deletion hits its bridge's circuit.
            # label 1: some deletions miss; usually 2+ bridges stay intact, so
            # the base is dependent and there are several certificates.
            if want_label == 1:
                n_keep = rng.randint(2, m) if rng.random() < 0.7 else 1
                keep = set(rng.sample(range(m), n_keep))
            else:
                keep = set()
            drops = []
            for i, (br, circ) in enumerate(bridges):
                off = [e for e in br if e not in circ]
                pool = (off or circ) if i in keep else circ
                drops.append(rng.choice(pool))
        for d in drops:
            builder.edges.discard(d)
        # Seed certificate search with every bridge circuit; broken ones are
        # skipped by _certificates (they no longer prove redundancy).
        circuits = [circ for _, circ in bridges]
    else:
        raise ValueError(family)
    return _finish(builder, (0, 1), circuits, family, rng, base_id)

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
             Built as feature-matched pairs (make_decoy_pair): both labels on
             the same graph with identical shortcut features.
  multi      2-3 different bridges sharing only u, v, one deletion each. The
             base is dependent when 2+ bridges stay intact; the candidate is
             redundant iff some deletion missed its bridge's circuit. Both
             labels have identical vertex and edge counts. Built as
             feature-matched pairs (make_multi_pair).
  symmetric  2-3 identical copies of one bridge with the same deletion in
             each copy. The candidate's stabilizer permutes the copies, so
             minimal certificates come in symmetry orbits: no symmetric
             explanation is a minimal certificate (orbit selection must
             overshoot). Built as feature-matched pairs (make_symmetric_pair).

All families are built as (independent, redundant) pairs on the same graph
with identical seven shortcut features; MAKERS maps family -> pair builder.

Each record follows the existing schema plus:
  certificates      distinct minimal certificates found (each includes the
                    candidate edge, like circuit_edges)
  circuit_edges     the smallest certificate (defines budget k = |C| - 1)
  base_dependent    whether the base edge set is dependent
  family            decoy | multi | symmetric
  pair_id           shared by the two records of a matched pair
Certificates are found by greedy deletion from many random orders, seeded
with each bridge circuit; each result is a minimal certificate (a circuit
through the candidate), but the list is not guaranteed exhaustive.
"""

from __future__ import annotations

import random
from itertools import combinations

from src.laman import random_laman_graph
from src.laman_diagnostics import candidate_features
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


def _finish(builder, cand, bridge_circuits, family, rng, base_id, cert_tries=40):
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
        certs = [sorted(cert + [c]) for cert in _certificates(n, edges, c, seeds, rng, tries=cert_tries)]
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


def _features(n, edges, cand):
    return candidate_features(n, sorted(edges), cand)


def make_decoy_pair(rng: random.Random, base_id: str, tries: int = 20) -> list[dict] | None:
    """Feature-matched decoy pair on one bridge + decoy graph.

    Deleting a circuit edge makes the candidate independent; deleting an
    off-circuit edge keeps it redundant. We pick one of each such that all
    seven shortcut features (n, |E|, rank deficit, endpoint degrees, common
    neighbours, endpoint distance) are identical, so no feature-only
    classifier can beat 50% on the pair.
    """
    for _ in range(tries):
        builder = _Builder()
        b_n = rng.randint(6, 8)
        b_edges, ab = _laman_with_nonedge(b_n, rng)
        mapped = builder.add_bridge(b_edges, b_n, ab)
        circ = _circuit_in(builder.n, mapped, (0, 1))
        off = [e for e in mapped if e not in circ]
        if not off:
            continue
        for _ in range(rng.randint(1, 2)):
            builder.add_decoy(rng, rng.randint(5, 7))
        full = set(builder.edges)
        by_feat = {}
        for e in off:
            by_feat.setdefault(_features(builder.n, full - {e}, (0, 1)), []).append(e)
        matches = [(e_in, by_feat[f]) for e_in in circ
                   if (f := _features(builder.n, full - {e_in}, (0, 1))) in by_feat]
        if not matches:
            continue
        e_in, outs = rng.choice(matches)
        e_out = rng.choice(outs)
        pair = []
        for drop, circuits in ((e_in, []), (e_out, [[e for e in circ]])):
            b = _Builder()
            b.n, b.edges = builder.n, full - {drop}
            rec = _finish(b, (0, 1), circuits, "decoy", rng, base_id)
            rec["pair_id"] = base_id
            pair.append(rec)
        if [r["label"] for r in pair] == [0, 1]:
            return pair
    return None


def make_multi_pair(rng: random.Random, base_id: str, tries: int = 20, samples: int = 150) -> list[dict] | None:
    """Feature-matched multi-certificate pair: m different bridges, one edge
    deleted per bridge. Label 0: every deletion hits its bridge's circuit.
    Label 1: deletions miss in a random subset of bridges (usually 2+, so the
    base is dependent with several certificates). The two deletion patterns
    are chosen to give identical shortcut features."""
    for _ in range(tries):
        builder = _Builder()
        m = rng.randint(2, 3)
        bridges = []
        for _ in range(m):
            b_n = rng.randint(5, 7)
            b_edges, ab = _laman_with_nonedge(b_n, rng)
            br = builder.add_bridge(b_edges, b_n, ab)
            circ = _circuit_in(builder.n, br, (0, 1))
            bridges.append((circ, [e for e in br if e not in circ]))
        if not any(off for _, off in bridges):
            continue
        full = set(builder.edges)

        def pattern(keep, bridges=bridges):
            return tuple(rng.choice(off if (i in keep and off) else circ)
                         for i, (circ, off) in enumerate(bridges))

        neg = {}
        for _ in range(samples):
            drops = pattern(set())
            neg.setdefault(_features(builder.n, full - set(drops), (0, 1)), drops)
        found = None
        for _ in range(samples):
            n_keep = rng.randint(2, m) if rng.random() < 0.7 else 1
            keep = {i for i in rng.sample(range(m), n_keep) if bridges[i][1]}
            if not keep:
                continue
            drops = pattern(keep)
            f = _features(builder.n, full - set(drops), (0, 1))
            if f in neg:
                found = (neg[f], drops)
                break
        if not found:
            continue
        pair = []
        for drops, seeds in ((found[0], []), (found[1], [circ for circ, _ in bridges])):
            b = _Builder()
            b.n, b.edges = builder.n, full - set(drops)
            rec = _finish(b, (0, 1), seeds, "multi", rng, base_id)
            rec["pair_id"] = base_id
            pair.append(rec)
        if [r["label"] for r in pair] == [0, 1]:
            return pair
    return None


def make_symmetric_pair(rng: random.Random, base_id: str, tries: int = 20) -> list[dict] | None:
    """Feature-matched symmetric pair: m identical bridges; the same edge is
    deleted in every copy. Deleting a circuit edge -> independent; an
    off-circuit edge -> redundant with m symmetry-related certificates. Both
    records share identical shortcut features."""
    for _ in range(tries):
        builder = _Builder()
        m = rng.randint(2, 3)
        b_n = rng.randint(5, 7)
        b_edges, ab = _laman_with_nonedge(b_n, rng)
        copies = [builder.add_bridge(b_edges, b_n, ab) for _ in range(m)]
        proto = copies[0]
        circ_idx = {proto.index(e) for e in _circuit_in(builder.n, proto, (0, 1))}
        off_idx = [i for i in range(len(proto)) if i not in circ_idx]
        if not off_idx:
            continue
        full = set(builder.edges)

        def without(i, full=full, copies=copies):
            return full - {c[i] for c in copies}

        by_feat = {}
        for i in off_idx:
            by_feat.setdefault(_features(builder.n, without(i), (0, 1)), []).append(i)
        matches = [(i, by_feat[f]) for i in sorted(circ_idx)
                   if (f := _features(builder.n, without(i), (0, 1))) in by_feat]
        if not matches:
            continue
        i_in, outs = rng.choice(matches)
        i_out = rng.choice(outs)
        circuits = [[c[i] for i in sorted(circ_idx)] for c in copies]
        pair = []
        for drop_i, seeds in ((i_in, []), (i_out, circuits)):
            b = _Builder()
            b.n, b.edges = builder.n, without(drop_i)
            rec = _finish(b, (0, 1), seeds, "symmetric", rng, base_id)
            rec["pair_id"] = base_id
            pair.append(rec)
        if [r["label"] for r in pair] == [0, 1]:
            return pair
    return None


MAKERS = {"decoy": make_decoy_pair, "multi": make_multi_pair, "symmetric": make_symmetric_pair}

"""Out-of-distribution stress families for testing whether models learned the RULE.

All labels come from the exact oracle (src/oracle.py). Graphs are far larger
and circuits far longer than anything in the training sets (training graphs
have at most ~20 vertices).

  ood_large    a random Laman graph on 24-48 vertices with a candidate
               non-edge whose circuit is long (best of several samples),
               then a feature-matched deletion pair: delete a circuit edge
               (independent) or an off-circuit edge (redundant).
  ood_chain    3-4 random Laman modules glued in sequence along shared
               edges (shared-edge gluing preserves the Laman property); the
               candidate joins the first and last module, so its circuit
               runs through the whole chain. Feature-matched deletion pair.
  lattice      bond-diluted triangular lattice (the standard model of
               rigidity percolation, e.g. Jacobs & Thorpe), L x L sites,
               bond probability p near the rigidity-percolation threshold.
               The candidate is a missing lattice bond. Bases are generally
               dependent; labels are balanced per lattice by sampling
               candidates; NOT feature-matched (report a feature baseline).
"""

from __future__ import annotations

import random

from src.hard_generator import _Builder, _circuit_in, _features, _finish
from src.laman import random_laman_graph
from src.oracle import is_redundant


def _matched_deletion_pair(builder: _Builder, cand, circ, rng, family, base_id):
    """Given a builder whose (independent) edge set makes `cand` redundant with
    circuit `circ`, return a feature-matched (independent, redundant) pair."""
    full = set(builder.edges)
    off = [e for e in full if e not in circ]
    if not off:
        return None
    by_feat = {}
    for e in off:
        by_feat.setdefault(_features(builder.n, full - {e}, cand), []).append(e)
    matches = [(e_in, by_feat[f]) for e_in in circ if (f := _features(builder.n, full - {e_in}, cand)) in by_feat]
    if not matches:
        return None
    e_in, outs = rng.choice(matches)
    e_out = rng.choice(outs)
    pair = []
    for drop, seeds in ((e_in, []), (e_out, [list(circ)])):
        b = _Builder()
        b.n, b.edges = builder.n, full - {drop}
        rec = _finish(b, cand, seeds, family, rng, base_id, cert_tries=3)
        rec["pair_id"] = base_id
        pair.append(rec)
    return pair if [r["label"] for r in pair] == [0, 1] else None


def make_large_pair(rng: random.Random, base_id: str, n_choices=(24, 32, 48), samples: int = 12):
    n = rng.choice(n_choices)
    edges = random_laman_graph(n, rng)
    present = set(edges)
    best = None
    for _ in range(samples):  # prefer candidates with long circuits
        u, v = sorted(rng.sample(range(n), 2))
        if (u, v) in present:
            continue
        circ = _circuit_in(n, edges, (u, v))
        if best is None or len(circ) > len(best[1]):
            best = ((u, v), circ)
    if best is None:
        return None
    b = _Builder()
    b.n, b.edges = n, set(edges)
    return _matched_deletion_pair(b, best[0], best[1], rng, "ood_large", base_id)


def make_chain_pair(rng: random.Random, base_id: str, modules=(3, 4)):
    k = rng.choice(modules)
    b = _Builder()
    b.n = 0
    module_vertices = []
    for i in range(k):
        m = rng.randint(6, 9)
        m_edges = random_laman_graph(m, rng)
        if i == 0:
            b.n = m
            b.edges = set(m_edges)
            module_vertices.append(set(range(m)))
            continue
        prev = sorted(e for e in b.edges if e[0] in module_vertices[-1] and e[1] in module_vertices[-1])
        a, c = rng.choice(prev)               # shared edge in the previous module
        x, y = rng.choice(m_edges)            # matching edge in the new module
        mapping = {x: a, y: c}
        for w in range(m):
            if w not in mapping:
                mapping[w] = b.n
                b.n += 1
        b.edges |= {tuple(sorted((mapping[p], mapping[q]))) for p, q in m_edges}
        module_vertices.append(set(mapping.values()))
    first = sorted(module_vertices[0] - set().union(*module_vertices[1:]))
    last = sorted(module_vertices[-1] - set().union(*module_vertices[:-1]))
    cands = [(min(p, q), max(p, q)) for p in first for q in last]
    rng.shuffle(cands)
    for cand in cands[:5]:
        if cand in b.edges:
            continue
        circ = _circuit_in(b.n, sorted(b.edges), cand)
        pair = _matched_deletion_pair(b, cand, circ, rng, "ood_chain", base_id)
        if pair:
            for r in pair:
                r["modules"] = k
            return pair
    return None


def triangular_lattice(L: int) -> list[tuple[int, int]]:
    idx = lambda i, j: i * L + j  # noqa: E731
    edges = []
    for i in range(L):
        for j in range(L):
            for di, dj in ((1, 0), (0, 1), (-1, 1)):
                a, c = i + di, j + dj
                if 0 <= a < L and 0 <= c < L:
                    edges.append(tuple(sorted((idx(i, j), idx(a, c)))))
    return sorted(edges)


def make_lattice_examples(rng: random.Random, base_id: str, per_class: int = 2,
                          sizes=(6, 8, 10), probs=(0.65, 0.75)):
    L, p = rng.choice(sizes), rng.choice(probs)
    full = triangular_lattice(L)
    kept = [e for e in full if rng.random() < p]
    missing = [e for e in full if e not in set(kept)]
    rng.shuffle(missing)
    by_label = {0: [], 1: []}
    for cand in missing:
        lab = int(is_redundant(L * L, kept, cand))
        if len(by_label[lab]) < per_class:
            by_label[lab].append(cand)
        if all(len(v) >= per_class for v in by_label.values()):
            break
    k = min(len(by_label[0]), len(by_label[1]))
    out = []
    for lab in (0, 1):
        for cand in by_label[lab][:k]:
            b = _Builder()
            b.n, b.edges = L * L, set(kept)
            rec = _finish(b, cand, [], "lattice", rng, base_id, cert_tries=2)
            rec.update(lattice_L=L, lattice_p=p)
            out.append(rec)
    return out

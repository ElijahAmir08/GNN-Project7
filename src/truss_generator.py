"""Planar bridge trusses (Pratt, Howe, Warren-with-verticals) as a realistic test.

A panel truss with p panels has bottom joints B0..Bp and top joints T0..Tp,
chords along both, a vertical in every panel boundary, and one diagonal per
panel: 2(p+1) joints and 4p+1 members = 2n-3, i.e. statically determinate
(minimally rigid, a Laman graph).

  pratt   diagonals slope down toward mid-span
  howe    diagonals slope up toward mid-span
  warren  diagonals alternate direction

Task framing ("repair"): k members are removed (damage); the candidate is a
member that could be added in a panel (any missing diagonal or vertical).
Independent = the repair adds a genuinely new constraint; redundant = the
remaining structure already fixes that distance. Labels come from the exact
oracle. Generic rigidity is a property of the layout; real trusses can also
be non-generic (e.g. collinear joints), which generic labels ignore.
"""

from __future__ import annotations

import random

from src.hard_generator import _Builder, _finish
from src.oracle import is_redundant


def panel_truss(p: int, style: str) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    """Returns (n, members, all potential panel members)."""
    B = list(range(p + 1))
    T = list(range(p + 1, 2 * p + 2))
    members = set()
    for i in range(p):
        members.add((B[i], B[i + 1]))
        members.add((T[i], T[i + 1]))
    for i in range(p + 1):
        members.add((B[i], T[i]))
    for i in range(p):
        left_half = i < p / 2
        if style == "pratt":
            d = (T[i], B[i + 1]) if left_half else (B[i], T[i + 1])
        elif style == "howe":
            d = (B[i], T[i + 1]) if left_half else (T[i], B[i + 1])
        elif style == "warren":
            d = (B[i], T[i + 1]) if i % 2 == 0 else (T[i], B[i + 1])
        else:
            raise ValueError(style)
        members.add(d)
    potential = set(members)
    for i in range(p):
        potential |= {(B[i], T[i + 1]), (T[i], B[i + 1])}
    norm = lambda es: sorted(tuple(sorted(e)) for e in es)  # noqa: E731
    return 2 * p + 2, norm(members), norm(potential)


def make_truss_examples(rng: random.Random, base_id: str, per_class: int = 2,
                        panels=(4, 6, 8, 10), styles=("pratt", "howe", "warren"), max_damage: int = 3):
    p, style = rng.choice(panels), rng.choice(styles)
    n, members, potential = panel_truss(p, style)
    damaged = rng.sample(members, rng.randint(1, max_damage))
    kept = [e for e in members if e not in damaged]
    cands = [e for e in potential if e not in set(kept)]
    rng.shuffle(cands)
    by_label = {0: [], 1: []}
    for c in cands:
        lab = int(is_redundant(n, kept, c))
        if len(by_label[lab]) < per_class:
            by_label[lab].append(c)
    k = min(len(by_label[0]), len(by_label[1]))
    out = []
    for lab in (0, 1):
        for c in by_label[lab][:k]:
            b = _Builder()
            b.n, b.edges = n, set(kept)
            rec = _finish(b, c, [], "truss", rng, base_id, cert_tries=5)
            rec.update(truss_style=style, panels=p, damaged=[list(e) for e in damaged])
            out.append(rec)
    return out

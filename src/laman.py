"""Random Laman (minimally rigid, generic 2D) graphs via Henneberg moves.

Every Laman graph can be built from a triangle by Henneberg moves:
  * Type I  (vertex addition): new vertex w joined to two existing vertices.
  * Type II (edge split):      remove edge (a, b); new vertex w joined to a, b,
                               and one other existing vertex c.
Both moves preserve the Laman property, so the result is always minimally
rigid with exactly 2n - 3 edges. Node labels are randomly permuted at the end
so vertex ids carry no information about construction order.
"""

from __future__ import annotations

import random


def random_laman_graph(num_nodes: int, rng: random.Random, p_split: float = 0.5) -> list[tuple[int, int]]:
    if num_nodes < 3:
        raise ValueError("Laman construction starts from a triangle (n >= 3)")
    edges: set[tuple[int, int]] = {(0, 1), (0, 2), (1, 2)}
    for w in range(3, num_nodes):
        if rng.random() < p_split:
            a, b = rng.choice(sorted(edges))
            c = rng.choice([v for v in range(w) if v not in (a, b)])
            edges.remove((a, b))
            edges.update({(a, w), (b, w), (c, w)})
        else:
            a, b = rng.sample(range(w), 2)
            edges.update({(a, w), (b, w)})

    perm = list(range(num_nodes))
    rng.shuffle(perm)
    return sorted(tuple(sorted((perm[u], perm[v]))) for u, v in edges)

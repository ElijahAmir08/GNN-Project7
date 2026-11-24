"""Family B: composed rigidity circuits with controlled composition depth.

Building blocks are 2D rigidity circuits: K4 and wheels W_m (hub + m-cycle,
m + 1 vertices, 2m edges = 2|V| - 2). The 2-sum of two circuits (identify an
edge of each, then delete the identified edge) is again a circuit, so repeated
2-sums give large circuits.

  chain  each new component is 2-summed onto the previous one:
         composition depth = number of components - 1.
  tree   components form a balanced binary tree of 2-sums:
         depth = tree height. Same number of components (same size) as a chain.

Query construction (matched pair, identical seven shortcut features):
  base circuit C, candidate e in the root component. Decoy Laman modules are
  attached at single joints (1-sums: they never change labels).
    positive  delete one decoy edge          -> base contains C - e, e REDUNDANT,
                                                unique certificate C - e
    negative  delete one circuit edge f != e -> e INDEPENDENT (C is minimal)
All labels are re-checked by the exact oracle when records are built.
"""

from __future__ import annotations

import random

import networkx as nx

from src.hard_generator import _features
from src.laman import random_laman_graph
from src.oracle import is_redundant


def k4():
    return 4, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


def wheel_circuit(m: int):
    hub = m
    return m + 1, sorted([tuple(sorted((i, (i + 1) % m))) for i in range(m)] + [(i, hub) for i in range(m)])


COMPONENTS = [k4, lambda: wheel_circuit(4), lambda: wheel_circuit(5)]


class Composite:
    """A circuit under construction; remembers which component each edge came from."""

    def __init__(self, n, edges):
        self.n = n
        self.edges = {tuple(sorted(e)): 0 for e in edges}  # edge -> component id
        self.n_comp = 1

    def two_sum(self, host_edge, comp_n, comp_edges, rng):
        """2-sum a new component onto host_edge (deleted afterwards)."""
        a, b = host_edge
        c, d = rng.choice(comp_edges)
        if rng.random() < 0.5:
            c, d = d, c
        mapping = {c: a, d: b}
        for w in range(comp_n):
            if w not in mapping:
                mapping[w] = self.n
                self.n += 1
        del self.edges[host_edge]
        cid = self.n_comp
        self.n_comp += 1
        for p, q in comp_edges:
            e = tuple(sorted((mapping[p], mapping[q])))
            if e != host_edge:
                self.edges[e] = cid
        return cid


def compose(specs: list[int], shape: str, rng: random.Random):
    """specs: component index per component (into COMPONENTS), root first.
    Returns (Composite, depth, root-component candidate edge)."""
    n, edges = COMPONENTS[specs[0]]()
    comp = Composite(n, edges)
    # Reserve the candidate edge in the root component before composing.
    cand = rng.choice(sorted(comp.edges))
    level = {0: 0}
    children = {0: 0}
    order = [0]
    for spec in specs[1:]:
        if shape == "chain":
            host_cid = comp.n_comp - 1
        else:  # balanced binary tree: attach to the shallowest component with < 2 children
            host_cid = next(c for c in order if children[c] < 2)
        hosts = [e for e, c in comp.edges.items() if c == host_cid and e != cand]
        cn, ce = COMPONENTS[spec]()
        cid = comp.two_sum(rng.choice(hosts), cn, ce, rng)
        children[host_cid] = children.get(host_cid, 0) + 1
        children[cid] = 0
        level[cid] = level[host_cid] + 1
        order.append(cid)
    return comp, max(level.values()), cand


def degree_profile(n, edges, cand, hops: int = 4) -> tuple:
    """Rigidity-free summary used as a shortcut check: degree counts in each
    hop ring around the candidate, the global degree histogram, size, and the
    number of cut vertices. v2 pairs must agree on all of it."""
    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from(edges)
    u, v = cand
    du = nx.single_source_shortest_path_length(g, u)
    dv = nx.single_source_shortest_path_length(g, v)
    dist = {w: min(du.get(w, 99), dv.get(w, 99)) for w in g}
    prof = []
    for hop in range(hops + 1):
        ring = [w for w in g if dist[w] == hop]
        prof += [sum(min(g.degree[w], 6) == k for w in ring) for k in range(7)]
    prof += [sum(min(g.degree[w], 6) == k for w in g) for k in range(7)]
    prof += [n, g.number_of_edges(), len(list(nx.articulation_points(g)))]
    return tuple(prof)


def make_circuit_pair(rng: random.Random, base_id: str, specs: list[int], shape: str,
                      decoy_sizes: list[int], tries: int = 20, decoy_specs: list[int] | None = None):
    """Matched (independent, redundant) pair for one composed circuit.
    Chain and tree twins share specs and decoys, so only the shape differs.

    v1 (decoy_specs=None): decoys are random Laman graphs; pairs match on the
       seven shortcut features only. A degree-profile classifier still reaches
       0.69 on v1 (decoy and circuit motifs differ), so v1 is confounded.
    v2 (decoy_specs given): decoys are circuit components (K4 / wheels) hung at
       one joint (1-sum, label unchanged), so positive and negative both delete
       an edge from the same kind of motif; the only difference is whether that
       component is 2-summed into the candidate's circuit or hangs off a single
       joint. Pairs must also agree on the full degree_profile."""
    n_components = len(specs)
    for _ in range(tries):
        comp, depth, cand = compose(specs, shape, rng)
        circuit = sorted(comp.edges)
        n = comp.n
        edges = set(circuit)
        decoy_edges = set()
        if decoy_specs is None:
            for size in decoy_sizes:
                mod = random_laman_graph(size, rng)
                anchor = rng.randrange(n)
                mapping = {0: anchor, **{i: n + i - 1 for i in range(1, size)}}
                n += size - 1
                new = {tuple(sorted((mapping[p], mapping[q]))) for p, q in mod}
                decoy_edges |= new
                edges |= new
        else:
            for spec in decoy_specs:
                cn, ce = COMPONENTS[spec]()
                anchor = rng.randrange(n)
                pivot = rng.randrange(cn)
                mapping = {pivot: anchor}
                for w in range(cn):
                    if w != pivot:
                        mapping[w] = n
                        n += 1
                new = {tuple(sorted((mapping[p], mapping[q]))) for p, q in ce}
                decoy_edges |= new
                edges |= new
        base_full = edges - {cand}

        def key(drop, n=n, base_full=base_full, cand=cand):
            k = _features(n, base_full - {drop}, cand)
            if decoy_specs is not None:
                k = k + degree_profile(n, sorted(base_full - {drop}), cand)
            return k

        by_feat = {}
        for g in decoy_edges:
            by_feat.setdefault(key(g), []).append(g)
        matches = [(f, by_feat[ft]) for f in circuit if f != cand
                   if (ft := key(f)) in by_feat]
        if not matches:
            continue
        f, gs = rng.choice(matches)
        g = rng.choice(gs)
        out = []
        for label, drop in ((0, f), (1, g)):
            out.append(_record(n, sorted(base_full - {drop}), cand, label, circuit, depth, n_components,
                               shape, base_id, rng))
        if [r["label"] for r in out] == [0, 1]:
            return out
    return None


def make_twin_pairs(rng: random.Random, base_id: str, n_components: int, version: int = 1):
    """A chain pair and a tree pair built from the same components and decoys."""
    specs = [rng.randrange(len(COMPONENTS)) for _ in range(n_components)]
    decoy_sizes = [rng.randint(5, 7) for _ in range(rng.randint(1, 2))]
    decoy_specs = [rng.randrange(len(COMPONENTS)) for _ in range(rng.randint(2, 3))] if version == 2 else None
    tries = 20 if version == 1 else 60
    chain = make_circuit_pair(rng, f"{base_id}_chain", specs, "chain", decoy_sizes, tries, decoy_specs)
    tree = make_circuit_pair(rng, f"{base_id}_tree", specs, "tree", decoy_sizes, tries, decoy_specs)
    return (chain or []) + (tree or []) if chain and tree else []


def _record(n, edges, cand, label, circuit, depth, n_comp, shape, base_id, rng):
    perm = list(range(n))
    rng.shuffle(perm)
    rel = lambda e: tuple(sorted((perm[e[0]], perm[e[1]])))  # noqa: E731
    E = sorted(rel(e) for e in edges)
    c = rel(cand)
    oracle_label = int(is_redundant(n, E, c))
    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from(E)
    try:
        dist = nx.shortest_path_length(g, c[0], c[1])
    except nx.NetworkXNoPath:
        dist = None
    return {
        "graph_id": f"{base_id}_{'pos' if label else 'neg'}",
        "base_id": base_id, "pair_id": base_id,
        "num_nodes": n, "edges": [list(e) for e in E], "candidate_edge": list(c),
        "label": oracle_label, "intended_label": label,
        "circuit_edges": [list(rel(e)) for e in circuit] if oracle_label else None,
        "certificates": [[list(rel(e)) for e in circuit]] if oracle_label else None,
        "family": f"circuit_{shape}", "composition_depth": depth, "n_components": n_comp,
        "candidate_distance": dist, "articulation_points": len(list(nx.articulation_points(g))),
        "split": "test",
    }

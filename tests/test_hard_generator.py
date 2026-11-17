import random

import pytest

from src.hard_generator import make_example
from src.oracle import is_redundant
from src.orbits import candidate_stabilizer_orbits, _orbits_by_enumeration
from src.rigidity_labels import make_graph


def pyrigi_redundant(n, edges, cand) -> bool:
    """Independent check via PyRigi: greedy rank (exact for a matroid)."""
    def rank(es):
        basis = []
        for e in es:
            if make_graph(n, basis + [e]).is_Rd_independent(dim=2):
                basis.append(e)
        return len(basis)
    return rank(edges + [cand]) == rank(edges)


@pytest.mark.parametrize("family", ["decoy", "multi", "symmetric"])
def test_labels_and_certificates(family):
    for seed in range(6):
        for want in (0, 1):
            r = make_example(family, random.Random(seed), "t", want)
            n, cand = r["num_nodes"], tuple(r["candidate_edge"])
            edges = [tuple(e) for e in r["edges"]]
            assert edges == sorted(edges) and cand not in edges
            assert r["label"] == int(pyrigi_redundant(n, edges, cand))
            if r["label"] == 0:
                assert r["certificates"] is None and r["circuit_edges"] is None
                continue
            assert r["circuit_edges"] == r["certificates"][0]
            for cert in r["certificates"]:
                body = [tuple(e) for e in cert if tuple(e) != cand]
                assert set(body) <= set(edges)
                assert is_redundant(n, body, cand)                       # valid
                for drop in body:                                        # minimal
                    assert not is_redundant(n, [e for e in body if e != drop], cand)


def test_symmetric_family_has_swapped_certificates():
    found = False
    for seed in range(10):
        r = make_example("symmetric", random.Random(seed), "t", 1)
        if r["label"] == 1 and len(r["certificates"]) >= 2:
            assert r["base_dependent"]
            edges = [tuple(e) for e in r["edges"]]
            orbits = candidate_stabilizer_orbits(r["num_nodes"], edges, tuple(r["candidate_edge"]))
            assert any(len(o) > 1 for o in orbits)
            found = True
    assert found


def test_fast_orbits_match_enumeration_on_generated_graphs():
    for family in ("decoy", "multi"):
        for seed in range(5):
            r = make_example(family, random.Random(seed), "t", seed % 2)
            args = (r["num_nodes"], [tuple(e) for e in r["edges"]], tuple(r["candidate_edge"]))
            assert candidate_stabilizer_orbits(*args) == _orbits_by_enumeration(*args)

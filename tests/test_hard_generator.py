import random

import pytest

from src.hard_generator import MAKERS
from src.laman_diagnostics import candidate_features
from src.oracle import is_redundant
from src.orbits import _orbits_by_enumeration, candidate_stabilizer_orbits
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


def pairs(family, n_seeds=6):
    out = []
    for seed in range(n_seeds):
        pair = MAKERS[family](random.Random(seed), f"t{seed}")
        if pair:
            out.append(pair)
    assert out, f"no {family} pairs generated"
    return out


@pytest.mark.parametrize("family", list(MAKERS))
def test_pairs_are_feature_matched_with_opposite_labels(family):
    for neg, pos in pairs(family):
        assert (neg["label"], pos["label"]) == (0, 1)
        assert neg["pair_id"] == pos["pair_id"]
        f = [candidate_features(r["num_nodes"], [tuple(e) for e in r["edges"]], tuple(r["candidate_edge"]))
             for r in (neg, pos)]
        assert f[0] == f[1]


@pytest.mark.parametrize("family", list(MAKERS))
def test_labels_and_certificates(family):
    for pair in pairs(family):
        for r in pair:
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
    for _, pos in pairs("symmetric"):
        assert pos["base_dependent"] and len(pos["certificates"]) >= 2
        edges = [tuple(e) for e in pos["edges"]]
        orbits = candidate_stabilizer_orbits(pos["num_nodes"], edges, tuple(pos["candidate_edge"]))
        assert any(len(o) > 1 for o in orbits)


def test_fast_orbits_match_enumeration_on_generated_graphs():
    for family in ("decoy", "multi"):
        for pair in pairs(family, n_seeds=4):
            for r in pair:
                args = (r["num_nodes"], [tuple(e) for e in r["edges"]], tuple(r["candidate_edge"]))
                assert candidate_stabilizer_orbits(*args) == _orbits_by_enumeration(*args)

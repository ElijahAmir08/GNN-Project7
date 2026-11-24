import random

import pytest

from src.circuit_generator import COMPONENTS, compose, make_twin_pairs
from src.laman_diagnostics import candidate_features
from src.oracle import is_redundant, rigidity_rank


def is_circuit(n, edges):
    edges = [tuple(e) for e in edges]
    r = rigidity_rank(n, edges)
    return r == len(edges) - 1 and all(rigidity_rank(n, [x for x in edges if x != d]) == r for d in edges)


def test_components_are_circuits():
    for make in COMPONENTS:
        assert is_circuit(*make())


@pytest.mark.parametrize("shape", ["chain", "tree"])
def test_two_sums_stay_circuits_with_expected_depth(shape):
    rng = random.Random(0)
    for k in (2, 3, 5, 8):
        comp, depth, cand = compose([rng.randrange(len(COMPONENTS)) for _ in range(k)], shape, rng)
        edges = sorted(comp.edges)
        assert is_circuit(comp.n, edges) and len(edges) == 2 * comp.n - 2
        assert cand in comp.edges
        assert depth == (k - 1 if shape == "chain" else max(1, (k).bit_length() - 1))


def test_twins_share_size_and_pairs_are_matched_and_correct():
    for seed in range(4):
        recs = make_twin_pairs(random.Random(seed), "t", 6)
        if not recs:
            continue
        sizes = {(r["num_nodes"], len(r["edges"])) for r in recs}
        assert len(sizes) == 1                                   # chain and tree twins: same size
        for i in range(0, 4, 2):
            neg, pos = recs[i], recs[i + 1]
            assert (neg["label"], pos["label"]) == (0, 1)
            feats = [candidate_features(r["num_nodes"], [tuple(e) for e in r["edges"]], tuple(r["candidate_edge"]))
                     for r in (neg, pos)]
            assert feats[0] == feats[1]
            for r in (neg, pos):
                assert r["label"] == int(is_redundant(r["num_nodes"], [tuple(e) for e in r["edges"]],
                                                      tuple(r["candidate_edge"])))


def test_v2_pairs_match_full_degree_profile():
    from src.circuit_generator import degree_profile
    for seed in range(4):
        recs = make_twin_pairs(random.Random(seed), "t", 5, version=2)
        assert recs
        for i in range(0, len(recs), 2):
            prof = [degree_profile(r["num_nodes"], [tuple(e) for e in r["edges"]], tuple(r["candidate_edge"]))
                    for r in recs[i:i + 2]]
            assert prof[0] == prof[1]
            assert [r["label"] for r in recs[i:i + 2]] == [0, 1]

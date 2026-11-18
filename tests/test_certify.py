import random

import numpy as np

from src.certify import OracleCounter, alternatives, certify, select_faithful, symmetric_set
from src.hard_generator import MAKERS
from src.oracle import is_redundant
from src.orbits import candidate_stabilizer_orbits

# Two K4-minus-edge diamonds sharing the candidate pair (0, 1): two certificates
# (edges 0-4 and 5-9) exchanged by the automorphism swapping 2<->4, 3<->5.
DIAMONDS = [(0, 2), (0, 3), (1, 2), (1, 3), (2, 3), (0, 4), (0, 5), (1, 4), (1, 5), (4, 5)]
CAND = (0, 1)


def minimal_valid(n, edges, cand, idx):
    sel = [edges[i] for i in idx]
    return is_redundant(n, sel, cand) and all(
        not is_redundant(n, [e for e in sel if e != d], cand) for d in sel)


def test_certify_any_scores_gives_minimal_certificate():
    oc = OracleCounter(6, DIAMONDS, CAND)
    for seed in range(10):
        c = certify(oc, np.random.default_rng(seed).random(len(DIAMONDS)))
        assert c in ([0, 1, 2, 3, 4], [5, 6, 7, 8, 9])
    assert oc.calls < 10 * 25


def test_certify_follows_scores():
    oc = OracleCounter(6, DIAMONDS, CAND)
    scores = np.array([0.1] * 5 + [0.9] * 5)
    assert certify(oc, scores) == [5, 6, 7, 8, 9]


def test_certify_returns_none_when_independent():
    oc = OracleCounter(6, DIAMONDS[:4] + DIAMONDS[5:9], CAND)  # both diamonds broken
    assert certify(oc, np.ones(8)) is None


def test_alternatives_and_symmetric_set():
    oc = OracleCounter(6, DIAMONDS, CAND)
    scores = np.linspace(1, 0, len(DIAMONDS))
    first = certify(oc, scores)
    certs = alternatives(oc, scores, first)
    assert sorted(certs) == [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
    orbits = candidate_stabilizer_orbits(6, DIAMONDS, CAND)
    out = symmetric_set(6, DIAMONDS, CAND, first, certs, orbits)
    assert sorted(out["members"]) == sorted(certs) and out["complete"]


def test_select_faithful_picks_highest_probability():
    certs = [[0, 1], [2, 3]]
    assert select_faithful(certs, lambda c: 0.9 if c == [2, 3] else 0.2) == ([2, 3], 0.9)


def test_certify_on_generated_hard_examples():
    for maker in MAKERS.values():
        for seed in range(3):
            pair = maker(random.Random(seed), "t")
            if not pair:
                continue
            pos = pair[1]
            n, cand = pos["num_nodes"], tuple(pos["candidate_edge"])
            edges = [tuple(e) for e in pos["edges"]]
            oc = OracleCounter(n, edges, cand)
            c = certify(oc, np.random.default_rng(seed).random(len(edges)))
            assert minimal_valid(n, edges, cand, c)
            for alt in alternatives(oc, np.random.default_rng(seed).random(len(edges)), c):
                assert minimal_valid(n, edges, cand, alt)

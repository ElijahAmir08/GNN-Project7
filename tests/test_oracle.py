from src.oracle import certificate_valid, is_redundant, rigidity_rank, truth_label

K4 = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


def test_ranks_of_known_graphs():
    assert rigidity_rank(3, [(0, 1), (0, 2), (1, 2)]) == 3          # triangle, rigid
    assert rigidity_rank(4, [(0, 1), (1, 2), (2, 3), (0, 3)]) == 4  # 4-cycle, flexible
    assert rigidity_rank(4, K4) == 5                                # 2n - 3, one redundant edge
    assert rigidity_rank(5, K4) == 5                                # isolated vertex changes nothing


def test_square_diagonals():
    square = [(0, 1), (1, 2), (2, 3), (0, 3)]
    assert truth_label(4, square, (0, 2)) == 0
    assert truth_label(4, square + [(0, 2)], (1, 3)) == 1


def test_certificate_validity_on_k4_circuit():
    base = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3)]  # K4 minus (2, 3)
    assert certificate_valid(4, base, (2, 3), range(5))
    for drop in range(5):
        assert not certificate_valid(4, base, (2, 3), [i for i in range(5) if i != drop])


def test_dependent_base_with_two_circuits_through_candidate():
    # Two K4-minus-edge "diamonds" sharing the candidate pair (0, 1):
    # vertices {0, 1, 2, 3} and {0, 1, 4, 5}. Each diamond alone is a certificate.
    left = [(0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    right = [(0, 4), (0, 5), (1, 4), (1, 5), (4, 5)]
    base = left + right
    cand = (0, 1)
    assert is_redundant(6, base, cand)
    assert certificate_valid(6, base, cand, range(0, 5))      # left diamond
    assert certificate_valid(6, base, cand, range(5, 10))     # right diamond
    # Removing one edge from each diamond kills both certificates.
    assert not certificate_valid(6, base, cand, [1, 2, 3, 4, 6, 7, 8, 9])
    # Removing an edge from only one diamond keeps the other one valid.
    assert certificate_valid(6, base, cand, [1, 2, 3, 4] + list(range(5, 10)))

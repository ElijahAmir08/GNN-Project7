import unittest
from src.rigidity_labels import label_proposed_edge


class RigidityDay1Tests(unittest.TestCase):
    square = [(0, 1), (1, 2), (2, 3), (0, 3)]

    def test_first_diagonal_is_independent(self):
        result = label_proposed_edge(4, self.square, (0, 2))
        self.assertEqual(result["label"], 0)
        self.assertIsNone(result["circuit_edges"])

    def test_second_diagonal_is_redundant(self):
        result = label_proposed_edge(4, self.square + [(0, 2)], (1, 3))
        self.assertEqual(result["label"], 1)
        self.assertEqual(len(result["circuit_edges"]), 6)
        self.assertIn([1, 3], result["circuit_edges"])

    def test_candidate_must_not_already_exist(self):
        with self.assertRaises(ValueError):
            label_proposed_edge(4, self.square, (0, 1))


if __name__ == "__main__":
    unittest.main()

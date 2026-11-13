
import unittest

from src.edge_orbits import compute_edge_orbits


class EdgeOrbitTests(unittest.TestCase):

    def test_square(self):
        edges = [
            (0, 1),
            (0, 3),
            (1, 2),
            (2, 3)
        ]

        result = compute_edge_orbits(4, edges)

        self.assertEqual(
            result,
            [[0, 1, 2, 3]]
        )

    def test_path(self):
        edges = [
            (0, 1),
            (1, 2),
            (2, 3)
        ]

        result = compute_edge_orbits(4, edges)

        self.assertEqual(
            result,
            [[0, 2], [1]]
        )

    def test_square_with_diagonal(self):
        edges = [
            (0, 1),
            (0, 3),
            (1, 2),
            (2, 3),
            (0, 2)
        ]

        result = compute_edge_orbits(4, edges)

        self.assertEqual(
            result,
            [[0, 1, 2, 3], [4]]
        )

        
    def test_real_laman_graph(self):
        edges = [
            (0, 3),
            (0, 4),
            (1, 2),
            (1, 3),
            (1, 4),
            (2, 3),
            (2, 4)
        ]

        result = compute_edge_orbits(5, edges)

        self.assertEqual(
            result,
            [[0, 1], [2], [3, 4, 5, 6]]
        )



if __name__ == "__main__":
    unittest.main()

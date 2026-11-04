import unittest
from src.laman_loader import decode_laman_integer

class LamanLoaderTests(unittest.TestCase):
    def test_decode_four_node_graph(self):
        edges = decode_laman_integer(31, 4)

        expected = [
            (0, 2),
            (0, 3),
            (1, 2),
            (1, 3),
            (2, 3),
        ]

        self.assertEqual(edges, expected)


if __name__ == "__main__":
    unittest.main()
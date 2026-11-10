import unittest
from src.laman_loader import (decode_laman_integer, load_laman_graphs)
from pathlib import Path
from tempfile import TemporaryDirectory

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
    def test_load_laman_graphs(self):
        with TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir)

            #Create a temporary LamanGraphs4.txt file
            test_file = dataset_dir / "LamanGraphs4.txt"
            test_file.write_text("31\n\n", encoding="utf-8")
            graphs = load_laman_graphs(4, dataset_dir)
            self.assertEqual(len(graphs), 1)

            expected_edges = [
                (0, 2),
                (0, 3),
                (1, 2),
                (1, 3),
                (2, 3),
            ]
            self.assertEqual(graphs[0], expected_edges)

if __name__ == "__main__":
    unittest.main()
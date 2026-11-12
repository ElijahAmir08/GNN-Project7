import json
import unittest
from pathlib import Path

import networkx as nx

from src.graph_rigidity import is_graph_rigid

DATASET_PATH = Path("data/processed/rigidity_seed.json")

class SeedDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = json.loads(
            DATASET_PATH.read_text(encoding="utf-8")
        )

    def test_equal_cases(self):
        self.assertEqual(len(self.records), 24)

        labels = [record["label"] for record in self.records]
        self.assertEqual(labels.count(1), 12)
        self.assertEqual(labels.count(0), 12)

    def test_graph_labels(self):
        for record in self.records:
            num_nodes = record["num_nodes"]
            edges = record["edges"]

            self.assertEqual(len(edges), 2 * num_nodes - 3)

            self.assertEqual(
                is_graph_rigid(num_nodes, edges),
                bool(record["label"])
            )

    def test_unique_graph_ids(self):
        ids = [record["graph_id"] for record in self.records]

        self.assertEqual(len(ids), len(set(ids)))
if __name__ == "__main__":
    unittest.main()
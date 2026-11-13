"""Test the exported edge-orbit metadata and dataset splits."""

import json
import unittest
from pathlib import Path

from src.edge_orbits import compute_edge_orbits


DATA_DIR = Path(__file__).resolve().parents[1] / "data/processed"


class PilotMetadataTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load the graph dataset
        cls.records = json.loads(
            (DATA_DIR / "rigidity_seed.json").read_text(
                encoding="utf-8"
            )
        )

        # Load the edge-orbit metadata
        cls.orbits = json.loads(
            (DATA_DIR / "rigidity_seed_orbits.json").read_text(
                encoding="utf-8"
            )
        )

        # Load the training/validation/test assignments
        cls.splits = json.loads(
            (DATA_DIR / "rigidity_seed_splits.json").read_text(
                encoding="utf-8"
            )
        )


    def test_orbit_metadata(self):
    # All 24 graphs should have orbit information.
        graph_ids = {
            record["graph_id"] for record in self.records
        }

        self.assertEqual(set(self.orbits), graph_ids)

        for record in self.records:
            graph_id = record["graph_id"]
            edges = record["edges"]

            groups = self.orbits[graph_id]

            # Every edge index must appear exactly once.
            covered = sorted(
                index
                for orbit in groups
                for index in orbit
            )

            self.assertEqual(
                covered,
                list(range(len(edges)))
            )

            # Verify the saved orbits match our function.
            expected = compute_edge_orbits(
                record["num_nodes"], edges
            )

            self.assertEqual(groups, expected)



    def test_split_assignments(self):
        split_groups = self.splits["splits"]

        # Verify the expected graph counts.
        self.assertEqual(len(split_groups["train"]), 16)
        self.assertEqual(len(split_groups["val"]), 4)
        self.assertEqual(len(split_groups["test"]), 4)

        # Collect graph IDs from all three splits.
        all_ids = []

        for graph_ids in split_groups.values():
            all_ids.extend(graph_ids)

        expected_ids = {
            record["graph_id"] for record in self.records
        }

        # Every graph should appear exactly once.
        self.assertEqual(len(all_ids), 24)
        self.assertEqual(len(set(all_ids)), 24)
        self.assertEqual(set(all_ids), expected_ids)

        
    def test_pairs_stay_together(self):
        # Map every graph ID to its assigned split.
        graph_to_split = {}

        for split_name, graph_ids in self.splits["splits"].items():
            for graph_id in graph_ids:
                graph_to_split[graph_id] = split_name

        # Reconstruct the original rigid/non-rigid pairs.
        pairs = {}

        for record in self.records:
            pair_id, kind = record["graph_id"].rsplit("_", 1)

            if pair_id not in pairs:
                pairs[pair_id] = {}

            pairs[pair_id][kind] = record["graph_id"]

        # Both graphs from each pair must be in the same split.
        for pair in pairs.values():
            rigid_id = pair["rigid"]
            nonrigid_id = pair["nonrigid"]

            self.assertEqual(
                graph_to_split[rigid_id],
                graph_to_split[nonrigid_id]
            )



    def test_class_balance(self):
        # Create a lookup from graph ID to rigidity label.
        labels_by_id = {
            record["graph_id"]: record["label"]
            for record in self.records
        }

        for split_name, graph_ids in self.splits["splits"].items():
            labels = [
                labels_by_id[graph_id]
                for graph_id in graph_ids
            ]

            with self.subTest(split=split_name):
                self.assertEqual(
                    labels.count(1),
                    labels.count(0)
                )


if __name__ == "__main__":
    unittest.main()

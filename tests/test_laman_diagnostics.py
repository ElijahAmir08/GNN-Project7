"""Check dataset decoding and the circuit-preserving deletion"""

import tempfile
import unittest
from pathlib import Path

from src.laman_diagnostics import (
    candidate_features, dataset_record, diagnose_record, find_matched_pair, isomorphic_base_groups,
)
from src.laman_loader import decode_laman_integer, load_laman_graphs, load_laman_integers
from src.rigidity_labels import label_proposed_edge


class LamanDiagnosticTests(unittest.TestCase):
    def test_read_codes_preserves_file_order_and_rejects_nonintegers(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "LamanGraphs5.txt"
            path.write_text("{254,\n239, 223}", encoding="utf-8")
            self.assertEqual(load_laman_integers(5, Path(folder)), [254, 239, 223])
            self.assertEqual(load_laman_graphs(5, Path(folder)),
                             [decode_laman_integer(code, 5) for code in (254, 239, 223)])
            path.write_text("{}", encoding="utf-8")
            self.assertEqual(load_laman_integers(5, Path(folder)), [])
            self.assertEqual(load_laman_graphs(5, Path(folder)), [])
            path.write_text("{'223'}", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_laman_integers(5, Path(folder))

    def test_reject_integer_outside_graph_bit_width(self):
        for code in (-1, 64):
            with self.subTest(code=code), self.assertRaises(ValueError):
                decode_laman_integer(code, 4)

    def test_features_use_existing_baseline_order_and_disconnected_distance(self):
        edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3),
                 (2, 4), (2, 5), (2, 6), (3, 4), (5, 6)]
        self.assertEqual(candidate_features(7, edges, (0, 4)), (7, 10, 1, 2, 2, 1, 2))
        self.assertEqual(candidate_features(7, edges, (0, 6)), (7, 10, 1, 2, 2, 1, 2))
        self.assertEqual(candidate_features(4, [(0, 1), (2, 3)], (0, 2)), (4, 2, 3, 1, 1, 0, 4))

    def test_matched_pair_preserves_circuit_and_has_both_mask_classes(self):
        parent = decode_laman_integer(2821346399, 9)
        pair = find_matched_pair(9, parent, min_circuit_size=10)
        self.assertIsNotNone(pair)
        removed, independent, redundant = pair
        self.assertEqual(independent["label"], 0)
        self.assertIsNone(independent["circuit_edges"])
        self.assertEqual(redundant["label"], 1)
        base = list(map(tuple, redundant["existing_edges"]))
        self.assertEqual(base, list(map(tuple, independent["existing_edges"])))
        self.assertEqual(len(base), 2 * 9 - 4)
        self.assertNotIn(removed, base)
        self.assertEqual(
            candidate_features(9, base, independent["candidate_edge"]),
            candidate_features(9, base, redundant["candidate_edge"]),
        )
        original = label_proposed_edge(9, parent, tuple(redundant["candidate_edge"]))
        self.assertEqual(redundant["circuit_edges"], original["circuit_edges"])
        self.assertGreaterEqual(len(redundant["circuit_edges"]), 10)
        row = diagnose_record(dataset_record("fixture", redundant, 2821346399, "circuit_matched"))
        self.assertEqual(row["rigidity_rank"], 14)
        self.assertGreaterEqual(row["off_circuit_edges"], 1)
        self.assertTrue(row["circuit_is_union_of_orbits"])

    def test_no_pair_when_circuit_has_no_outside_edges(self):
        self.assertIsNone(find_matched_pair(4, decode_laman_integer(31, 4), min_circuit_size=6))
        with self.assertRaises(ValueError):
            find_matched_pair(4, [(0, 1), (1, 2), (2, 3), (0, 3)])

    def test_relabelled_bases_from_different_parents_share_a_group(self):
        edges = [(0, 1), (1, 2), (2, 3), (0, 3)]
        perm = [2, 0, 3, 1]
        relabeled = sorted(tuple(sorted((perm[u], perm[v]))) for u, v in edges)
        self.assertNotEqual(sorted(edges), relabeled)
        records = [
            {"base_id": "a", "num_nodes": 4, "edges": edges},
            {"base_id": "a", "num_nodes": 4, "edges": edges},
            {"base_id": "b", "num_nodes": 4, "edges": relabeled},
            {"base_id": "c", "num_nodes": 4, "edges": edges[:-1]},
        ]
        self.assertEqual(isomorphic_base_groups(records), [["a", "b"], ["c"]])


if __name__ == "__main__":
    unittest.main()

"""Export edge-orbit metadata and train/validation/test splits"""

import json
import random
from pathlib import Path

from src.edge_orbits import compute_edge_orbits

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/processed"
SEED = 42

def main():
    # Load the 24-graph dataset
    dataset_path = DATA_DIR / "rigidity_seed.json"

    records = json.loads(
        dataset_path.read_text(encoding="utf-8")
    )

    # Compute orbits for every graph
    orbits_by_id = {}

    for record in records:
        graph_id = record["graph_id"]

        orbits = compute_edge_orbits(
            record["num_nodes"],
            record["edges"],
        )

        # Verify all edge indices belong to exactly one orbit
        covered = sorted(
            index for orbit in orbits for index in orbit
        )

        assert covered == list(range(len(record["edges"])))

        orbits_by_id[graph_id] = orbits

    # Save the edge-orbit metadata
    orbit_path = DATA_DIR / "rigidity_seed_orbits.json"

    orbit_path.write_text(
        json.dumps(orbits_by_id, indent=2) + "\n",
        encoding="utf-8"
    )

    print(f"Exported edge orbits for {len(orbits_by_id)} graphs.")

    # Keep related rigid/non-rigid graphs together so the model
    # isn't tested on graphs that are nearly identical to its training data.    
    pairs = {}
    for record in records:
        pair_id, kind = record["graph_id"].rsplit("_", 1)

        if pair_id not in pairs:
            pairs[pair_id] = {}

        pairs[pair_id][kind] = record["graph_id"]

    # Ensure that there are 12 pairs and each pair has one rigid/nonrigid
    assert len(pairs) == 12
    assert all(
        set(pair) == {"rigid", "nonrigid"}
        for pair in pairs.values()
    )

    # Shuffle the pair IDs to split better into training/validation/testing sets
    pair_ids = sorted(pairs.keys())
    random.Random(SEED).shuffle(pair_ids)

    # Convert pairs back into individual graphs
    def expand_pairs(selected_pairs):
        graph_ids = []

        for pair_id in selected_pairs:
            graph_ids.append(pairs[pair_id]["rigid"])
            graph_ids.append(pairs[pair_id]["nonrigid"])

        return graph_ids

    # 8 pairs to training, 2 to validation,
    # and 2 to testing
    splits = {
        "seed": SEED,
        "split_unit": "rigid_nonrigid_pair",
        "splits": {
            "train": expand_pairs(pair_ids[:8]),
            "val": expand_pairs(pair_ids[8:10]),
            "test": expand_pairs(pair_ids[10:12])
        }
    }

    # Save the split assignments
    split_path = DATA_DIR / "rigidity_seed_splits.json"

    split_path.write_text(
        json.dumps(splits, indent=2) + "\n",
        encoding="utf-8"
    )

    for split_name, graph_ids in splits["splits"].items():
        print(f"{split_name}: {len(graph_ids)} graphs")


if __name__ == "__main__":
    main()

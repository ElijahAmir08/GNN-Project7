"""Day 1: run two known rigidity examples and save their verified labels."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.rigidity_labels import label_proposed_edge

SQUARE = [(0, 1), (1, 2), (2, 3), (0, 3)]
FIRST_DIAGONAL = (0, 2)
SECOND_DIAGONAL = (1, 3)


def main() -> None:
    independent = label_proposed_edge(4, SQUARE, FIRST_DIAGONAL)
    redundant = label_proposed_edge(4, SQUARE + [FIRST_DIAGONAL], SECOND_DIAGONAL)

    assert independent["label"] == 0, independent
    assert independent["circuit_edges"] is None
    assert redundant["label"] == 1, redundant
    assert redundant["circuit_edges"] is not None
    assert len(redundant["circuit_edges"]) == 6

    print("EXAMPLE A: adding first diagonal")
    print("  label =", independent["label"], independent["label_name"])
    print("  circuit =", independent["circuit_edges"])
    print("EXAMPLE B: adding second diagonal")
    print("  label =", redundant["label"], redundant["label_name"])
    print("  circuit =", redundant["circuit_edges"])

    destination = ROOT / "data" / "generated" / "day1_examples.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps([independent, redundant], indent=2) + "\n", encoding="utf-8"
    )
    print("PASS: two correct labels and verified K4 circuit")
    print("Saved", destination.relative_to(ROOT))


if __name__ == "__main__":
    main()

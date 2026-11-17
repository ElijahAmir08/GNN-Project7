"""Inspect a small Laman sample; write a diagnostic report, not training splits."""

import argparse
import csv
import hashlib
import json
import random
import statistics
import sys
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.edge_dataset import examples_for_base
from src.laman_diagnostics import dataset_record, diagnose_record, find_matched_pair, isomorphic_base_groups
from src.laman_loader import decode_laman_integer, load_laman_integers
from src.rigidity_labels import make_graph


def summarize(rows):
    redundant = [row for row in rows if row["label"] == 1]
    return {
        "examples": len(rows), "redundant_examples": len(redundant),
        "parents": len({(row["num_nodes"], row["parent_code"]) for row in rows}),
        "common_neighbor_accuracy": statistics.mean(
            row["common_neighbor_prediction"] == row["label"] for row in rows
        ) if rows else None,
        "mean_circuit_size": statistics.mean(row["circuit_size"] for row in redundant) if redundant else None,
        "fraction_with_nontrivial_edge_orbits": statistics.mean(
            row["has_nontrivial_edge_orbit"] for row in rows
        ) if rows else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--nodes", type=int, nargs="+", default=[8, 9])
    parser.add_argument("--parents-per-size", type=int, default=24)
    parser.add_argument("--min-circuit-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=ROOT / "data/generated/laman_diagnostic.json")
    args = parser.parse_args()
    if args.parents_per_size < 1 or args.min_circuit_size < 6:
        parser.error("Use at least one parent and a circuit threshold of at least six")
    rng = random.Random(args.seed)
    records, sources, yields = [], {}, {}
    for n in sorted(set(args.nodes)):
        path = args.data_dir / f"LamanGraphs{n}.txt"
        codes = sorted(load_laman_integers(n, args.data_dir))
        selected = sorted(rng.sample(codes, min(args.parents_per_size, len(codes))))
        sources[str(n)] = {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                           "available": len(codes), "selected_codes": selected}
        counts = {"parents": len(selected), "random_retained": 0, "matched_retained": 0}
        for code in selected:
            edges = decode_laman_integer(code, n)
            if len(edges) != 2 * n - 3 or not make_graph(n, edges).is_min_rigid(dim=2):
                raise ValueError(f"Invalid Laman parent: n={n}, code={code}")
            base_id = f"n{n}_p{code}"
            control_rng = random.Random(f"{args.seed}:{n}:{code}")
            removed = control_rng.choice(edges)
            control = examples_for_base(base_id + "_random", n, [e for e in edges if e != removed], control_rng, 1)
            if control:
                counts["random_retained"] += 1
            for rec in control:
                rec.update(parent_code=code, construction_family="random_delete_one",
                           removed_edge=list(removed), seed=args.seed)
                records.append(rec)

            pair = find_matched_pair(n, edges, args.min_circuit_size)
            if pair is not None:
                removed, independent, redundant = pair
                pair_id = f"{base_id}_matched_d{removed[0]}-{removed[1]}"
                for labeled in (independent, redundant):
                    rec = dataset_record(pair_id, labeled, code, "circuit_matched")
                    rec.update(removed_edge=list(removed), seed=args.seed)
                    records.append(rec)
                counts["matched_retained"] += 1
        yields[str(n)] = counts
        print(f"n={n}: {counts}", flush=True)
    rows = [diagnose_record(record) for record in records]
    summary = {family: summarize([row for row in rows if row["construction_family"] == family])
               for family in ("random_delete_one", "circuit_matched")}
    matched_parents = {(row["num_nodes"], row["parent_code"]) for row in rows
                       if row["construction_family"] == "circuit_matched"}
    summary["random_on_matched_parents"] = summarize([
        row for row in rows if row["construction_family"] == "random_delete_one"
        and (row["num_nodes"], row["parent_code"]) in matched_parents
    ])
    report = {
        "meta": {"seed": args.seed, "min_circuit_size": args.min_circuit_size,
                 "python": sys.version.split()[0],
                 "pyrigi": version("pyrigi"), "networkx": version("networkx"),
                 "rank_method": "edge count after PyRigi verifies base independence",
                 "scope": "diagnostic selection sample; no training splits or GIN benchmark",
                 "sources": sources},
        "yield_by_size": yields, "summary": summary,
        "base_isomorphism_groups": {
            "all": isomorphic_base_groups(records),
            "circuit_matched": isomorphic_base_groups([
                rec for rec in records if rec["construction_family"] == "circuit_matched"
            ]),
        },
        "records": records, "diagnostics": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with args.out.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["graph_id"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.out} and {args.out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()

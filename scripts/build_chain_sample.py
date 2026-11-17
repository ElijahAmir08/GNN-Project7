"""Build a small diagnostic sample from glued Laman modules; no training splits."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys
from importlib.metadata import version

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.chain_graphs import glue_laman_graphs, cross_module_candidates, circuit_module_support
from src.laman_diagnostics import dataset_record, diagnose_record, find_matched_pair, isomorphic_base_groups
from src.laman_loader import decode_laman_integer, load_laman_integers
from src.rigidity_labels import make_graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/laman"))
    parser.add_argument("--nodes", type=int, nargs="+", default=[8, 9], help="vertices per module")
    parser.add_argument("--pairs-per-size", type=int, default=8)
    parser.add_argument("--min-circuit-size", type=int, default=16)
    parser.add_argument("--min-distance", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=ROOT / "data/generated/chain_sample.json")
    args = parser.parse_args()
    if args.pairs_per_size < 1 or args.min_circuit_size < 6 or args.min_distance < 2:
        parser.error("Require positive sample size, circuit size >= 6, and distance >= 2")
    rng = random.Random(args.seed)
    records, rows, cases, sources = [], [], [], {}
    for n in sorted(set(args.nodes)):
        codes = sorted(load_laman_integers(n, args.data_dir))
        if not codes:
            raise ValueError(f"No source graphs for module size {n}")
        path = args.data_dir / f"LamanGraphs{n}.txt"
        sources[str(n)] = {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        kept = 0
        for index in range(args.pairs_per_size):
            left_code, right_code = rng.choice(codes), rng.choice(codes)
            left = decode_laman_integer(left_code, n)
            right = decode_laman_integer(right_code, n)
            left_edge, right_edge = rng.choice(left), rng.choice(right)
            chain = glue_laman_graphs(n, left, n, right, left_edge, right_edge)
            total_n = chain["num_nodes"]
            parent_code = make_graph(total_n, chain["edges"]).to_int(vertex_order=list(range(total_n)))
            case_id = f"glue_n{n}_s{args.seed}_b{index}"
            module_ids = [f"n{n}_c{left_code}", f"n{n}_c{right_code}"]
            case = {
                "case_id": case_id, "parent_code": parent_code, "num_nodes": total_n,
                "module_ids": module_ids, "source_codes": [left_code, right_code],
                "source_attachment_edges": [left_edge, right_edge],
                "shared_edge": chain["shared_edge"],
                "module_vertices": chain["module_vertices"],
                "right_vertex_map": chain["right_vertex_map"], "matched": False,
            }
            candidates = cross_module_candidates(chain)
            pair = find_matched_pair(total_n, chain["edges"], args.min_circuit_size,
                                     candidate_edges=candidates, independent_candidates=candidates,
                                     min_distance=args.min_distance)
            if pair is not None:
                removed, independent, redundant = pair
                case["matched"] = True
                case["removed_edge"] = removed
                kept += 1
                for labeled in (independent, redundant):
                    rec = dataset_record(case_id, labeled, parent_code, "glued_laman_matched")
                    rec.update(seed=args.seed, removed_edge=removed, module_ids=module_ids)
                    records.append(rec)
                    row = diagnose_record(rec)
                    row["module_circuit_edges"] = circuit_module_support(chain, rec["circuit_edges"]) if rec["label"] else None
                    row["shared_edge_in_circuit"] = list(chain["shared_edge"]) in rec["circuit_edges"] if rec["label"] else None
                    rows.append(row)
            cases.append(case)
        print(f"{n}+{n} vertices -> {2*n-2}: {kept}/{args.pairs_per_size} parents yielded a pair", flush=True)

    redundant_rows = [row for row in rows if row["label"] == 1]
    groups = isomorphic_base_groups(records)
    summary = {
        "attempted_parents": len(cases), "matched_pairs": len(redundant_rows),
        "examples": len(rows), "unique_base_topologies": len(groups),
        "chain_vertex_counts": sorted({row["num_nodes"] for row in rows}),
        "circuit_sizes": dict(Counter(row["circuit_size"] for row in redundant_rows)),
        "mean_circuit_size": statistics.mean(row["circuit_size"] for row in redundant_rows) if redundant_rows else None,
        "candidate_distances_per_pair": dict(Counter(row["candidate_distance"] for row in redundant_rows)),
        "common_neighbor_accuracy": statistics.mean(row["label"] == row["common_neighbor_prediction"] for row in rows) if rows else None,
        "examples_with_nontrivial_edge_orbits": sum(row["has_nontrivial_edge_orbit"] for row in rows),
    }
    report = {
        "meta": {"seed": args.seed, "min_circuit_size": args.min_circuit_size,
                 "min_distance": args.min_distance, "python": sys.version.split()[0],
                 "pyrigi": version("pyrigi"), "networkx": version("networkx"),
                 "sources": sources, "scope": "diagnostic sample; no training splits or GIN benchmark",
                 "split_requirement": "group shared source modules and isomorphic derived bases"},
        "summary": summary, "cases": cases, "base_isomorphism_groups": groups,
        "records": records, "diagnostics": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with args.out.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["graph_id"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.out.name} and {args.out.with_suffix('.csv').name}")


if __name__ == "__main__":
    main()

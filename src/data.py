"""JSON records -> PyG Data objects for the edge-level rigidity task.

Edge-alignment contract (shared with the explanation / orbit code):
  * record["edges"] is the sorted list of undirected edges; undirected edge i
    is edges[i].
  * edge_index has 2E columns: column 2i = (u_i -> v_i), column 2i+1 = (v_i -> u_i).
  * A directed edge mask m (length 2E) folds back to undirected scores with
    `directed_to_undirected(m)` (mean of the two directions by default).
  * The candidate edge is NOT in edge_index (it is not a bar yet); it is stored
    in `candidate` and marked by node feature `is_endpoint`.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch_geometric.data import Data

NUM_NODE_FEATURES = 2  # [constant 1, is_candidate_endpoint]


class EdgeTaskData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == "candidate":
            return self.num_nodes
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == "candidate":
            return 0
        return super().__cat_dim__(key, value, *args, **kwargs)


def record_to_data(rec: dict) -> EdgeTaskData:
    n = rec["num_nodes"]
    edges = [tuple(e) for e in rec["edges"]]
    if edges != sorted(edges) or any(u >= v for u, v in edges):
        raise ValueError(f"{rec['graph_id']}: edges must be sorted (u < v) pairs")

    u, v = rec["candidate_edge"]
    x = torch.zeros(n, NUM_NODE_FEATURES)
    x[:, 0] = 1.0
    x[[u, v], 1] = 1.0

    if edges:
        e = torch.tensor(edges, dtype=torch.long)            # [E, 2]
        edge_index = torch.stack([e, e.flip(1)], dim=1).reshape(-1, 2).t().contiguous()
    else:
        edge_index = torch.empty(2, 0, dtype=torch.long)

    circuit = {tuple(c) for c in (rec.get("circuit_edges") or [])}
    circuit_mask = torch.tensor([e in circuit for e in edges], dtype=torch.bool)

    return EdgeTaskData(
        x=x,
        edge_index=edge_index,
        y=torch.tensor([rec["label"]], dtype=torch.long),
        candidate=torch.tensor([[u, v]], dtype=torch.long),
        circuit_mask=circuit_mask,
        num_nodes=n,
        graph_id=rec["graph_id"],
        base_id=rec.get("base_id", rec["graph_id"]),
    )


def directed_to_undirected(mask: torch.Tensor, reduce: str = "mean") -> torch.Tensor:
    pairs = mask.view(-1, 2)
    if reduce == "mean":
        return pairs.mean(dim=1)
    if reduce == "max":
        return pairs.max(dim=1).values
    raise ValueError(reduce)


def load_split_datasets(path: Path) -> dict[str, list[EdgeTaskData]]:
    blob = json.loads(Path(path).read_text(encoding="utf-8"))
    records = blob["records"] if isinstance(blob, dict) else blob
    out: dict[str, list[EdgeTaskData]] = {"train": [], "val": [], "test": []}
    for rec in records:
        out[rec["split"]].append(record_to_data(rec))
    return out

# Edge-level task: design notes (Salaar, Nov 11)

Task follows the Monday pitch: given an independent base graph G and a proposed
non-edge (u, v), predict **0 = independent** or **1 = redundant** in generic 2D
rigidity. Redundant examples come with a unique fundamental rigidity circuit
(ground truth for explanation agreement). Independent examples have no circuit,
so only faithfulness/consistency metrics apply to them.

## Data recipe (`src/edge_dataset.py`, `scripts/build_edge_dataset.py`)
1. Random Laman graph on n nodes via Henneberg I/II moves (`src/laman.py`),
   with nodes randomly relabeled.
2. Delete k ∈ {1,2,3} edges → independent, flexible base.
3. Label every non-edge with `label_proposed_edge` (PyRigi).
4. Keep `min(per_class, #indep, #redundant)` candidates per label per base,
   so each base contributes a balanced set (graph size / edge count carry no signal).
5. Split by base graph; isomorphic bases are forced into the same split.

Note: adding any edge to a full Laman graph is always redundant, which is why we delete edges first.

## Data contract (`src/data.py`)
- `edges` sorted (u < v); undirected edge i ↔ `edges[i]`.
- `edge_index` columns 2i, 2i+1 = both directions of edge i.
- `directed_to_undirected(mask)` folds a 2E mask to E scores (mean by default).
- Candidate is not in `edge_index`; stored in `candidate` and marked by `x[:, 1]`.
- `circuit_mask[i]` = edge i is in the circuit (candidate excluded).
- Orbits for symmetry correction must be automorphisms of the base graph that
  **fix the candidate pair setwise** (stabilizer), not the full automorphism group.

## Findings so far
- Fixed bug in `fundamental_circuit`: PyRigi's `is_Rd_circuit` returns False if
  the graph has isolated vertices, so any circuit not spanning all n vertices
  crashed. The circuit is now verified on its edge-induced subgraph.
- Redundant candidates are rare after deleting edges (e.g. n=6, k=3: ~1%), so
  ~half of bases have one class only and are skipped. Generator should be
  biased toward bases with larger rigid subcomponents.
- Shortcut: endpoints with ≥2 common neighbours are almost always redundant
  (single-feature lookup accuracy 0.84). Many circuits are small, local motifs.
  Consider rebalancing by circuit size / endpoint distance.

## Preliminary explainer run (Nov 11, `scripts/explain_prelim.py`, gin_edge_medium seed0, 300 test examples)
- Circuit AUROC: PGExplainer 0.94, GNNExplainer 0.72, random 0.50, oracle 1.00.
- fid- (keep only top-k): oracle 0.01, PGExplainer 0.21, GNNExplainer 0.49, random 0.50.
- fid+ saturates (~0.5 for all methods): removing ANY circuit edge breaks redundancy
  (independent base => unique circuit), and circuits cover ~56% of base edges.
- Only 19% of examples have a nontrivial candidate-stabilizer orbit.
- Circuit is a union of stabilizer orbits in 100% of redundant examples
  (expected: stabilizer preserves the rigidity matroid and fixes the candidate;
  the fundamental circuit is unique, so it is mapped to itself).
- Orbit splitting on symmetric examples: GNNExplainer raw 55% -> orbit_avg 22% -> orbit_sel 0%.
  PGExplainer raw 29% -> orbit_avg 29% (scores already tied; splits come from
  arbitrary tie-breaking, which averaging cannot fix) -> orbit_sel 0%.
- GNNExplainer is noisy: Spearman vs. itself with a new seed 0.59; vs. relabeled 0.40.

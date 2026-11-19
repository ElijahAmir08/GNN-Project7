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

## Model ladder + full explainer sweep (Nov 11; results/tables.md via scripts/make_tables.py)
- Classification test acc (3 seeds): features 0.773, gcn 0.975, mpnn 1.000, gin 0.997, dsgin 0.998.
  1-WL ceiling on our data is 100%, so dsgin has nothing extra to resolve here.
- Explanation quality is dominated by checkpoint variance: PGExplainer circuit AUROC on
  six same-config GIN checkpoints ranges 0.11–0.93 at identical test accuracy.
- Soft-mask OOD: with all edges at mask 0.5 (explainer init), ~50% of predictions flip
  for every model; GCN margin jumps 9.5 -> ~249 and its explainer gradient vanishes.
- Orbit corrections: whole-orbit selection always gives 0% orbit splits and lowers fid-
  (better sufficiency) at +0.2–0.3 extra edges; circuit AUROC/precision change <= 0.03.
- GNNExplainer relabeling Spearman (0.28–0.59) is always below its same-graph reseed
  Spearman (0.48–0.73): relabeling adds inconsistency on top of optimizer noise.

## Stress tests on Elijah's sets (Nov 17; scripts/eval_checkpoints.py, no retraining)
Checkpoints trained on edge_medium (n = 6–10). Labels of all 56 Elijah records re-verified with PyRigi.
2/26 laman-matched bases are isomorphic to a medium train base (reported with and without).

| test acc (3 seeds)      | medium test | matched pairs (n 8–9) | Henneberg n 14–16 | chains n 14–16 |
|---|---|---|---|---|
| features                | 0.773 | 0.500 | 0.764 | 0.500 |
| gcn                     | 0.975 | 0.936 | 0.821 | 0.578 |
| mpnn                    | 1.000 | 1.000 | 0.917 | 0.689 |
| gin (run A / run B)     | 0.995 / 0.997 | 1.000 / 1.000 | 0.882 / 0.831 | 0.733 / 0.678 |
| dsgin                   | 0.998 | 1.000 | 0.924 | 0.856 |

- Matched pairs kill the shortcut (features 50%) but not the GNNs (100%): the GNNs never relied on it.
- Size alone (Henneberg 14–16) costs 8–17 points; chains cost a further 7–25 points
  (long circuits spanning both modules). GCN/MPNN fail mostly by calling redundant
  chain candidates independent (redundant acc 0.22 / 0.38).
- DS-GIN (> 1-WL) is best on both stress sets; with n = 30 the chain gap is suggestive, not conclusive.

## Oracle-graded explanations (Nov 17; src/oracle.py, scripts/oracle_eval.py, results/oracle_tables.md)
Exact oracle = generic 2D rigidity rank at random realizations; matches PyRigi on 5,290 labels
and 330 random (incl. 74 dependent) edge sets. F = hard-mask sufficiency, V = S alone is a
rigidity certificate for the prediction; budget k = |C| - 1 (strict: minimal certificate).
- Medium test: strict F∧V oracle 94–100%; best explainer PGExplainer on gin run A 44 ± 23%;
  other explainer/model pairs 0.4–20%; random 0%. Almost no "faithful but invalid" cases, and
  the model is correct on 96–100% of masked graphs: in-distribution, faithfulness and truth
  coincide, so explainer failures are search failures, not a model that reasons differently.
- Chains: divergence appears. The true certificate is valid but NOT faithful in 15–25% of
  cases for GCN/GIN (model rejects the real circuit; GCN is right on only 66% of oracle-masked
  graphs). MPNN 96% and DS-GIN 83% oracle F∧V. Real explainers ~0% on chains, and certificate
  efficiency ≈ random (0.78–0.85) because circuits cover ~80% of edges: chain sets cannot rank
  explainers until circuit share is reduced (decoy structure) and graded metrics are added.
- Whole-orbit selection never lowers F∧V and raises it slightly (e.g. DS-GIN GNNExplainer 10.7 -> 14.7%).

## Hard set: explainers + oracle (Nov 17; data_new/generated/hard_eval.json, results/oracle_tables.md)
Families from src/hard_generator.py (360 examples, PyRigi-verified, minimal certificates).
- Decoy (circuits ~30% of edges): explainers clearly beat random (slack 0.37–0.58 vs 0.13;
  recall 0.36–0.65 vs 0.30). MPNN shows the first large "faithful but INVALID" mass:
  28–38% of its explanations (and 31% of random subsets) keep its 'redundant' prediction
  without containing a certificate; MPNN is right on only ~80% of masked graphs.
  High accuracy on full graphs does not mean the decision depends on a complete certificate.
- Multi / symmetric: GIN, DS-GIN, GCN predict 'redundant' on 55–88% of independent examples
  (FP); MPNN mostly correct. Strict validity of real explainers ~0–4%; slack barely above random.
- Exact necessity breaks with non-unique ground truth: removing the oracle's minimal certificate
  changes the true label in only 62% (multi) and 0% (symmetric) of cases — the symmetric copy
  still proves redundancy. Deletion-based fidelity is ill-posed when certificates are not unique.
- Symmetry vs. minimality, first evidence: on the symmetric family whole-orbit selection
  overshoots by ~1 edge, never yields a minimal valid certificate (F∧V 0%), and LOWERS slack
  (e.g. PGExplainer on MPNN 0.29 -> 0.19, GIN 0.39 -> 0.26). On every other dataset the
  corrections were neutral or helpful.

## Fixes evaluated: three training regimes (Nov 18; results/comparison.md via scripts/compare_runs.py)
original = edge_medium (3 seeds); mixed = + 11–16-node graphs + hard families (5 seeds);
maskaug = mixed + oracle-labeled masked subgraphs (5 seeds). Same evaluation pipeline
(scripts/run_eval_sweep.sh); hard set = matched pairs; PGExplainer explains predictions.
- Accuracy: mixed training removes the hard-family failures (GIN symmetric 0.53 -> 1.00) and
  generalizes to the never-seen chains (GIN 0.68 -> 0.97, DS-GIN 0.86 -> 0.99). GCN stays at
  ~0.58 on chains in every regime (architecture limit). maskaug keeps accuracy.
- Accuracy is not certificate reliance: after mixed training GIN/DS-GIN accept the true
  certificate alone in only 21–80% of cases. Masked-subgraph augmentation raises this to
  99–100% for MPNN, GIN and DS-GIN on every set (including unseen chains), and drives MPNN's
  faithful-but-invalid rate (random subsets) from 27–40% to 0%.
  Caveat: maskaug trains on certificate-like subgraphs; its gains are on held-out graphs, but the
  training distribution is aligned with the faithfulness test by design.
- Explainers: EdgeGrad (gradient on the unmasked graph) dominates PGExplainer and GNNExplainer
  in every regime (slack 0.85–1.00 vs 0.0–0.6 on chains/decoy/multi). On symmetric graphs it ties
  symmetric copies, so top-k splits them (slack ~0.46) — the predicted symmetry/minimality issue.
- Combined fix: EdgeGrad + certificate step on maskaug models gives faithful AND valid
  explanations in 99–100% of predicted-redundant cases on chains, decoy, multi and symmetric
  (MPNN, GIN, DS-GIN), and the set-valued output stays symmetry-closed.

## Rule tests, OOD suite, trusses, ablation, readout (Nov 18; research log artifact has all tables)
- Rule tests (scripts/rule_tests.py): monotonicity learned (add/delete violations 0–3% for
  MPNN/GIN/DS-GIN); the 1-sum law is not (4–26%, worse for larger attachments, all regimes).
- OOD suite (scripts/build_ood_suite.py; 24–100 joints, certificates of 24–50 bars): best
  accuracy ood_chain 0.96, ood_large 0.82, lattice 0.91 — the real generalization benchmark.
- Trusses (src/truss_generator.py; damaged Pratt/Howe/Warren, repair framing): GNNs 0.77–0.89
  vs feature baseline 0.80; they miss repairs that would work. Not feature-matched yet.
- Random-mask ablation: random masks alone raise GIN/DS-GIN certificate acceptance to 97–100%
  on hard families and chains, and keep EdgeGrad + certificate F∧V at 97–100%. Masked training
  is not just teaching the test; certificate-shaped masks matter most on lattices/large graphs.
- Pair (endpoint-only) readout: +15–26 points on lattices (MPNN 0.91), MPNN 1-sum violations
  25.6% -> 9.4%, but −8–25 points on chains/large Laman graphs. Trade-off; try mean pooling.

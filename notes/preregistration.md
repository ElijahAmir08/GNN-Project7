# Pre-registration: diagnostic phase (written Nov 23, 2026, before running Experiments 0–3)

Three candidate bottlenecks for why the GNNs do not learn the rigidity rule:
1. **Expressivity**: the model cannot represent the required distinction.
2. **Propagation**: the distinction is representable but its signal decays with composition.
3. **Wrong computation**: the information is available but the model learns shortcuts.

Rules for every experiment: hyperparameters, round counts and epochs are chosen on validation
data only; test sets are evaluated once per configuration; results are reported per family and
per composition depth (never pooled into one OOD number); confidence intervals by paired
bootstrap over examples, plus the spread across training seeds.

## Experiment 0: controlled benchmark (gate, no learning)
Family A: wheel/hinge pairs (`src/wl_family.py`, existing).
Family B: composed rigidity circuits (`src/circuit_generator.py`): K4 and wheel circuits joined
by 2-sums into a **chain** (composition depth = number of components − 1) or a **balanced
tree** (depth = tree height) with the same number of components, so depth varies at fixed size.
Matched pairs: decoy Laman modules are attached at single joints; the positive deletes a decoy
edge (certificate C − e intact, label redundant), the negative deletes a circuit edge f ≠ e
(label independent); both deletions are chosen to give identical seven shortcut features.

Gate (must pass before any training):
- every label equal under the exact oracle and PyRigi's pebble game;
- every stored certificate valid and minimal;
- the seven-feature lookup is at exactly 50% on matched pairs.

## Experiment 1: is it expressivity?
H1a. On Family A, every 1-WL model outputs identical values on each pair (already verified).
H1b. On Family B, positive and negative members are generally NOT 1-WL-equivalent (they
differ in where an edge was deleted). Prediction: ≥ 90% of Family B pairs have different
1-WL signatures, so Family B failures are not expressivity failures.
H1c. Biconnectivity or resistance-distance features fix Family A at all held-out k.
H1d. Those features do not by themselves fix deep Family B compositions (2-sums of circuits are
2-connected: no cut vertex marks the answer).
Falsifiers: H1b fails if many Family B pairs are 1-WL-equivalent; H1c fails if resistance
features leave held-out Family A pairs at chance; H1d fails if they solve all depths.
Next action: expressivity-limited examples are reported separately from learnable ones.

## Experiment 2: does the signal decay with composition?
Measure the normalized separation Δ_k = ||h(G+) − h(G−)|| / (½(||h(G+)|| + ||h(G−)||) + ε)
on graph-level embeddings, at initialization and after training, in float32 and float64,
by message-passing round.
H2a. Δ_k decreases with chain depth for existing models.
H2b. At equal size, chains (deep) separate worse than balanced trees (shallow).
H2c. Accuracy by depth tracks Δ_k.
Falsifiers: H2b fails if chain ≈ tree at equal size (then size, not depth, is the driver);
H2c fails if Δ_k stays large while accuracy collapses (then the problem is optimization or the
objective, not propagation).
A decay bound Δ_k ≤ Cρ^k is a conjecture unless proven for a linearized model.

## Experiment 3: does structural supervision teach the rule?
Same backbone and training budget for all of:
- A: candidate labels only;
- B: A + extra candidate-marked queries on the same graphs (more labels, same content);
- C: A + all-pairs rigid-component targets R_ij through a symmetric pairwise head;
- D: A + an equally sized rigidity-irrelevant pairwise target (i, j within 2 hops).
Run with a 1-WL backbone and a more expressive one.
H3. C beats A, B and D on held-out depths and OOD families, with fewer 1-sum violations and
better R_ij recovery.
Falsifiers: C ≈ B (more labels explain it) or C ≈ D (any auxiliary target explains it), or
C improves familiar graphs only.
Next action if H3 fails: algorithm-aligned processor (pebble-game supervision) as future work.

## Outcomes so far (Nov 23)
Experiment 0 (scripts/build_circuit_benchmark.py): 840 records, 420 matched pairs, 30 chain/tree twins per k = 2..12.
Gates passed: G1 labels oracle = intended = PyRigi 840/840; G2 certificates valid + minimal 840/840;
G3 shortcut features identical 420/420 (feature baseline exactly 50%). Leakage vs all training sets: 0/840.
H1b CONFIRMED: 420/420 pairs have different 1-WL signatures -> Family B failures are not expressivity.
Existing checkpoints (scripts/eval_circuit_depth.py), pair accuracy, chain vs tree at equal size:
  k=6 (depth 5 vs 2): masked MPNN 0.41 vs 0.87, masked GIN 0.49 vs 0.83, mixed GIN 0.47 vs 0.80.
  Recurrent models collapse on deep chains to "redundant" (independent-class accuracy 0.01–0.10).
H2b CONFIRMED at the accuracy level: depth hurts at fixed size (size also hurts: trees decline with k).
Experiment 2 (scripts/separation_by_depth.py): normalized separation of matched pairs shrinks with k
  (masked MPNN 0.38 -> 0.12; ~100x larger than at initialization) but is the SAME for chains and trees at
  equal k (k=6: 0.19 vs 0.22). H2a confirmed; H2c FALSIFIED: accuracy does not track separation.
Experiment 2b (scripts/probe_by_depth.py): a linear probe on frozen classifier inputs (5-fold CV by pair)
  recovers the label at 0.85–0.99 for every depth, where models score 0.51–0.74 on deep chains (recurrent
  MPNN: model 0.51, probe 0.96 at chain k=6). A probe fit only on shallow composed circuits (k <= 4)
  reaches 0.79–0.89 on deep chains (models 0.66–0.74).
Interpretation (per pre-registered rule): the label information survives composition in the
representation; the failure is mostly in how the learned readout maps it to a decision (bottleneck 3,
"wrong computation" / objective), with a smaller residual shift across depth (CV probe 0.95 vs shallow-
fit probe 0.80 on deep chains). Caveat: the shallow-fit probe is trained on the same family, the models'
heads on other families, so part of its advantage is in-family calibration.

## Analysis before any new training (Nov 23, scripts/head_diagnostics.py + counterfactuals)
D1/D2 (head vs probe, same input = the head's input vector): on deep chains the trained heads still ORDER
matched pairs correctly 88–97% of the time and never reverse them (0.00); failures are "both called
redundant" (46–97% of pairs). Negatives' logit difference drifts from about −7 (shallow) to −2.4..+7
(deep) while positives stay positive; the head-input norm roughly doubles (e.g. 123 -> 257).
D4: probes fit on shallow chains vs shallow trees transfer about equally to deep chains (0.71–0.88 vs
0.70–0.85): the label direction does not rotate with depth (hypothesis C weakened).
Causal tests:
- Rescaling the whole-graph sum term by (mean train size / n): accuracy got WORSE everywhere, including
  shallow graphs. "Sum magnitude alone causes the drift" is not supported.
- First block-swap test was flawed: it replaced a block with its shallow SAME-LABEL mean, which leaks the
  label; its apparent fix is discarded.
- Label-agnostic block swap (block set to its mean over ALL shallow examples): swapping the endpoint-sum or
  endpoint-product block changes deep accuracy by ≤ 0.03; swapping the whole-graph-sum block drops deep
  accuracy to chance (0.46–0.56) and pair accuracy to ~0. The recurrent MPNN differs (endpoint-driven).
Conclusion: for fixed-depth models the decision is read mainly from the whole-graph sum, which carries
both the label information and a size-dependent component that grows with the graph; the head reads a
combination that shifts with size, so negatives drift across the threshold while ordering is preserved.
This supports hypothesis B (rigidity information entangled with size/composition in the global readout),
not D (wrong direction) or C (rotating direction). It reconciles F10 (pair readout removes the global term:
fixes large lattices, loses chains, because the label information lives there).

## Audit of everything since the benchmark (Nov 23) — v1 confounded, rebuilt as v2, all conclusions re-verified
Checks that PASSED on v1: probe shuffled-label controls 0.45–0.52; head(H) reproduces model predictions
840/840 (the captured vector is the true head input); recurrent models evaluated at 10 rounds (val-chosen);
per-block probes show label decodable from the graph-sum block (0.93–0.98), not the endpoint blocks
(0.55–0.73), so the block-swap reading was not an off-manifold artifact.
Check that FAILED on v1: a rigidity-free logistic regression on degree counts around the candidate reached
0.69 (0.62–0.73 per subgroup). Cause: positives damaged Laman decoys, negatives damaged circuit motifs.
v1 numbers are therefore confounded by a degree shortcut and are superseded.
v2 (circuit_bench_v2.json): decoys are circuit components (K4 / wheels) hung at one joint; pairs must match the
seven features AND the full degree profile (rings 0–4 around the candidate, global degree histogram, n, |E|,
cut vertices). Gates: G1 840/840, G2 840/840, G3 420/420, G4 profiles identical 420/420, degree-profile
classifier 0.500. Leakage vs training: 0/840; overlap with v1: 0. 1-WL distinguishable: 420/420.
v2 re-verification (all conclusions hold; numbers are lower than v1 because the shortcut is gone):
- Depth hurts at fixed size: k=6 chain vs tree pair accuracy, masked MPNN 0.17 vs 0.53, masked GIN 0.21 vs 0.49.
- Separation equal for chains and trees at equal k (masked MPNN k=6: 0.138 vs 0.146): H2c still falsified.
- CV probe on deep groups 0.86–0.95 vs models 0.51–0.68; shuffled labels ~0.50; UNTRAINED same-architecture
  network 0.53–0.71 (training creates most of the decodable information; for GIN some is generic, 0.71 on chains).
- Label decodable from the graph-sum block 0.89–0.94, endpoint blocks 0.51–0.61 (recurrent MPNN tree 0.75).
- Shallow-fit probe on deep groups 0.72–0.88 vs models 0.51–0.68.
- Heads still order deep-chain pairs correctly 0.79–0.96 (probe 0.94–0.99); failures mostly "both redundant";
  negatives' logit difference drifts upward (masked MPNN −1.7 -> +3.4); head-input norm roughly doubles.
Note: in head_diagnostics the shallow-row probe accuracy/ordering is in-sample (probe fit on shallow data);
only deep rows are held out.
Size vs label (v2, all examples): graph size is linearly decodable from the head input (CV R² 0.97–0.98); the
size direction and the shallow-fit label direction are nearly orthogonal (cosine −0.04 to −0.06), so the two
are separable in principle. The head's logit difference on negatives tracks log n more than the probe's does
(masked MPNN +0.68 vs +0.29; recurrent MPNN +0.74 vs +0.39; GIN +0.46 vs +0.38; mixed MPNN +0.32 vs +0.26).
Reading: size is encoded alongside, and almost orthogonal to, the rigidity signal; trained heads (most clearly
the MPNNs) partly read the size component, which drifts negatives across the threshold on larger graphs.

## Milestone 1A: retrained heads on frozen encoders (Nov 24, scripts/head_retrain.py)
Fresh head-training set: circuit_shallow_train_v2.json (k = 2..4, seed 1, all v2 gates passed, degree-profile
classifier 0.500, 0/1800 overlap with the benchmark). Selection on validation only; benchmark deep groups
evaluated once. Mean ± std over 5 encoder seeds.
- Retraining the head on the SAME original data barely helps: deep-chain pair accuracy MPNN 0.10 -> 0.27–0.29,
  GIN 0.16 -> 0.12–0.16. The original head's training procedure is not the bottleneck.
- A head trained on fresh shallow circuits only (frozen encoder) extrapolates to deep ones: MPNN 0.65 ± 0.06,
  GIN 0.44–0.47; negatives' score −6 (drift gone); ordering 0.94–0.97. The encoder supports depth extrapolation.
- But it costs other families: ood_large 0.82 -> 0.53–0.56 (MPNN), chains/medium drop; trained on both data
  sources, deep chain 0.57 (MPNN) / 0.40 (GIN) with ood_large still 0.59 / 0.61 (from 0.82 / 0.76).
- Label directions fit per family (shared standardization) are only partly aligned: cosine 0.32–0.60
  (original data vs circuits 0.49–0.60; circuits vs ood_large 0.32–0.35). Diagnostic only (ood_large direction
  fit on that set itself).
Decision (pre-registered gate): a readout-only fix does NOT pass — it improves deep compositions only by
sacrificing existing OOD performance. The representation encodes rigidity partly family-specifically; the
size drift is one symptom. Next: an intervention on the representation (Milestone 1B), not the head.

## Milestone 1B pre-registration (written Nov 24, before training)
Intervene on the representation (whole model retrained), matched budgets: MPNN and GIN, 4 layers, 60 epochs,
5 seeds each, model selection on validation only.
- Baseline: existing maskaug_{mpnn,gin} (masked training, sum readout, no composed circuits).
- 1B-1 "circ": masked training data + the fresh shallow composed circuits (k = 2..4, seed 1; same train/val
  pair split as Milestone 1A), sum readout. Control for exposure to the family.
- 1B-2 "circ+mean": same data, global term = mean pool instead of sum (size-normalized readout).
Evaluation (held out, once): deep v2 benchmark (k >= 6) chain/tree pair accuracy, ordering, negatives' score;
stress sets (medium, chains, ood_chain, ood_large, lattice); 1-sum violation rate (rule tests); cosine between
label directions fit on original data vs circuits vs ood_large.
Predictions:
P1 (circ): deep-chain pair accuracy rises well above the baseline (≥ 0.40 for MPNN) because the encoder sees the
family; ood_large stays within 0.05 of the baseline; cross-family label-direction cosine increases.
P2 (circ+mean): negatives' score drift with size shrinks and deep-chain pair accuracy ≥ P1; lattice accuracy
improves (size-normalized global term); 1-sum violations fall; ood_large not below P1 − 0.05.
Falsifiers: P1 fails if deep chains improve only by losing > 0.05 on ood_large (same trade-off as 1A, now in the
representation); P2 fails if mean pooling does not change drift or 1-sum behaviour relative to P1.
Gate: proceed with readout/representation research only if a condition improves deep v2 without materially
sacrificing pair discrimination or existing OOD accuracy, consistently across seeds.

## Milestone 1B outcome (Nov 24; results/eval_milestone1b.md, results/rules_1b.md; 5 seeds each)
| run | deep chain pair | deep tree pair | ood_large | lattice | 1-sum@20 | cos circ–ood_large |
| maskaug_mpnn | 0.10 | 0.28 | 0.82 | 0.65 | 25.6% | 0.35 |
| circ_mpnn | 0.59 | 0.79 | 0.63 | 0.65 | 24.3% | 0.47 |
| circmean_mpnn | 0.51 | 0.84 | 0.68 | 0.82 | 15.5% | 0.49 |
| maskaug_gin | 0.16 | 0.30 | 0.76 | 0.64 | 12.8% | 0.32 |
| circ_gin | 0.69 | 0.72 | 0.55 | 0.69 | 25.0% | 0.29 |
| circmean_gin | 0.65 | 0.76 | 0.67 | 0.78 | 20.3% | 0.36 |
P1: deep chains ≥ 0.40 YES (0.59 / 0.69); ood_large within 0.05 NO (−0.19 / −0.21); direction alignment mixed.
P2: lattice improves YES (+0.17 / +0.14); 1-sum violations fall YES for MPNN, NO for GIN; deep chains ≥ P1 NO
(slightly lower); ood_large ≥ P1 − 0.05 YES.
Gate: FAILED for all four conditions (ood_large drops 0.09–0.21). Adding a family to training fixes that family
and costs another; size normalization softens but does not remove the trade-off. Per the stopping rule,
readout/data-coverage fixes are closed. Next: a family-invariant training target (Experiment 3, all-pairs
rigid components) with controls B (more candidate labels) and D (rigidity-irrelevant pairwise target); gate:
deep v2 chains improve AND ood_large within 0.05 of baseline AND cross-family direction cosine increases.

## Experiment 3 design details (fixed Nov 24, before building data or training)
Shared: training data = mixed_masked_circ_train (as in 1B), sum readout, 4 layers, 60 epochs, 5 seeds,
MPNN and GIN, selection on validation candidate accuracy only. A = existing circ_{mpnn,gin}.
Auxiliary pairs: for every train/val graph, up to 96 vertex pairs that are NOT edges, sampled uniformly
(seeded per graph); C and D use the SAME pairs, so they get identical numbers of auxiliary labels.
- B: each train/val graph gets 4 extra candidate-marked queries (other non-edges, oracle-labelled), i.e.
  5x the candidate labels. (Matching C's ~96 labels/graph with marked queries is infeasible: each needs its
  own forward pass. B tests "more labels of the same kind" within budget.)
- C: auxiliary target R_ij = 1 iff i and j are rigidly linked (adding bar ij is redundant), exact oracle.
- D: auxiliary target = 1 iff hop distance(i, j) <= 2 (structural, rigidity-irrelevant).
C and D: symmetric pairwise head on node embeddings, MLP([h_i + h_j, h_i * h_j]) -> 1 logit; loss =
candidate CE + 1.0 * BCE(aux). lambda = 1.0 fixed (not tuned).
Extra measurement: R_ij recovery for every model via a pairwise linear probe on frozen node embeddings,
fit on the fresh shallow circuits and tested on the deep benchmark graphs (AUROC).
Gate (unchanged): deep v2 chains improve AND ood_large within 0.05 of the masked baseline AND cross-family
direction cosine increases; plus C must beat B and D (otherwise "more labels" or "any auxiliary target"
explains it).
Deviation recorded before training: B has ~5.7x the training records of A, so "60 epochs" would give B
5.7x the optimizer steps. To compare supervision rather than compute, B trains for 11 epochs (≈ the same
number of optimizer steps as A's 60). C and D add their auxiliary labels on the same batches as A (60 epochs).

## Experiment 3 outcome (Nov 24; results/eval_exp3.md, rules_exp3.md, rigid_recovery_exp3.md; 5 seeds each)
| model | cond | deep chain | deep tree | ood_large | cos circ–ood_large | R_ij AUROC | 1-sum@20 |
| MPNN | base | 0.10 | 0.28 | 0.82 | 0.35 | 0.822 | 25.6% |
| MPNN | A | 0.59 | 0.79 | 0.63 | 0.47 | 0.817 | 24.3% |
| MPNN | B | 0.43 | 0.40 | 0.55 | 0.44 | 0.822 | 14.1% |
| MPNN | C | 0.77 | 0.88 | 0.65 | 0.46 | 0.912 | 20.8% |
| MPNN | D | 0.65 | 0.86 | 0.65 | 0.42 | 0.856 | 25.9% |
| GIN | base | 0.16 | 0.30 | 0.76 | 0.32 | 0.788 | 12.8% |
| GIN | A | 0.69 | 0.72 | 0.55 | 0.29 | 0.838 | 25.0% |
| GIN | B | 0.62 | 0.68 | 0.57 | 0.44 | 0.831 | 17.1% |
| GIN | C | 0.68 | 0.79 | 0.67 | 0.29 | 0.901 | 17.4% |
| GIN | D | 0.76 | 0.84 | 0.64 | 0.30 | 0.855 | 19.9% |
Gate: FAILED for both backbones (ood_large −0.17 / −0.09 vs baseline; cross-family cosine not increased).
H3 partially supported: C gives the best R_ij recovery for both backbones (0.90–0.91, own head 0.90–0.91 on
deep graphs); for MPNN, C beats A, B and D on deep chains (+0.18 over A, +0.12 over D; Welch t over 5 seeds
≈ 2.7 and 2.4, suggestive) and trees; for GIN, D ≥ C, so the GIN gain is not structure-specific.
Conclusion across Milestones 1A, 1B and Experiment 3: readout fixes, data coverage, size normalization and
structural supervision each improve the failure they target, but none yields a family-independent decision
rule; ood_large always pays. Structural supervision teaches the structure (decodable) without making the
decision rule use it consistently (MPNN partly, GIN not).

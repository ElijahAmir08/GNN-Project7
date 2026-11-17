# Laman candidate diagnostic

## Goal

Find candidates that defeat the seven-feature shortcut baseline while retaining larger circuits. This is a selection diagnostic.


Added loader function `load_laman_integers` to expose the existing parser for sampling and provenance.
Graph loading retains file order; only the diagnostic sorts codes for sampling.

## Method

1. Sample 24 parent graphs at each of sizes 8 and 9, seed 0.
2. Verify each parent is Laman with PyRigi.
3. For the control, delete one random edge and sample one candidate per label.
4. For the selected examples, find a candidate with a circuit of at least
   ten edges. Delete an edge outside that circuit, leaving another outside edge.
5. Pair the original redundant candidate with the independent restoration edge.
   Require identical endpoint degrees, common neighbors, and distance on the same base. 
   Prefer larger circuits; retain one pair per parent.

Both candidates also share vertex count, edge count, and rank deficit, so
all seven inputs to `FeatureMLP` match. The preserved circuit supplies positive
explanation edges; the remaining off-circuit edges supply negatives.

The integer decoder follows the dataset's
[Description.txt](https://zenodo.org/records/1245517/preview/Description.txt?include_deleted=0):
upper-triangle adjacency bits, row by row, most significant first.
The published example `n=4, code=31` has missing edge `(0,1)`.

## Results

Of 48 parents, 39 produced balanced random controls and 13 produced matched
pairs: one at size 8 and twelve at size 9. The table restricts the control
to those same 13 parents. Each row contains 26 examples, balanced by label.

| Selection | CN >= 2 accuracy | Mean circuit edges | Nontrivial edge orbits |
|---|---:|---:|---:|
| Random deletion/candidates on matched parents | 65.4% | 8.62 | 7.7% |
| Circuit-preserving deletion and feature matching | 50.0% | 10.62 | 11.5% |

Circuit sizes include the candidate. Selected circuits: nine of size 10 and
four of size 12. Each has three or five off-circuit base edges. All redundant
circuits were unions of candidate-stabilizer edge orbits.

Across all 39 retained random-control parents, the heuristic scored 80.8%
and mean circuit size was 7.95. This sample differs from the earlier default
dataset: sizes 8/9, one deletion, and one candidate per label.

Example: parent code `2821346399`, nine vertices. Delete `(2,4)`.
Candidate `(0,4)` is redundant with a 12-edge circuit; restoring `(2,4)` is independent. Both feature vectors are `(9,14,1,2,2,0,3)`. Three base edges lie outside the circuit. Neither candidate has a nontrivial edge orbit.


## Limits

- Exact feature matching forces any deterministic seven-feature classifier
  to 50% on these balanced pairs. This is a construction property, does not establish GIN difficulty.
- Only three of the 26 selected examples have nontrivial edge orbits.
  A useful symmetry subset needs separate attention.
- The 13 matched bases contain only 12 distinct topologies. No train/test split is assigned. Future splits must group related parents and isomorphic
  derived bases together; parent IDs alone do not prevent leakage.
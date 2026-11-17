# Larger examples from Laman modules

## Construction

Join two dataset Laman graphs along one chosen edge. Only those two vertices (As suggested by Ambuj from our update meeting)
are identified, and the shared edge is kept once. The result has
`n = n_left + n_right - 2` vertices and `m = 2*n - 3` edges.
Shared-edge gluing preserves Laman sparsity; the generator also verifies both
inputs and the assembled parent with PyRigi. Edge count alone is not enough.

The generator returns vertex mappings and each module's edges so circuit
coverage can be checked after gluing.

## Candidate selection

1. Consider missing edges between exclusive vertices of different modules.
2. Find a redundant candidate with a circuit of at least 16 edges.
3. Remove a parent edge outside that circuit, retaining an off-circuit edge.
4. Find an independent candidate that also crosses between the modules.
5. Require identical seven-feature vectors and candidate distance at least 3.
6. Recheck both labels and preservation of the redundant circuit with the
   existing labeler. Retain one pair per chain parent.

Both labels cross the join, so crossing it cannot distinguish them. Both
labels also have identical summary features. Candidate masks and circuits
retain the existing format; no model or labeling changes were needed.

The matcher caches parent circuits. Deleting edge 'f' makes a parent nonedge independent exactly when 'f' belonged to its fundamental circuit. This screens possible matches; selected examples are still verified through PyRigi.

## Fixed sample: seed 0

| Modules | Parent vertices / edges | Flexible base edges | Matched parents |
|---|---:|---:|---:|
| 8 + 8 vertices | 14 / 25 | 24 | 8 of 8 |
| 9 + 9 vertices | 16 / 29 | 28 | 7 of 8 |

This gives 15 pairs, 30 balanced examples, and 15 distinct base topologies.

| Property | Earlier small matched sample | chain sample |
|---|---:|---:|
| Vertices | 8 or 9 | 14 or 16 |
| Circuit sizes, including candidate | 10-12 | 16-26 |
| Mean circuit size | 10.62 | 21.07 |
| Candidate distances | 2-3 | 3-5 |
| Examples with nontrivial edge orbits | 3 of 26 | 9 of 30 |

These are different selected samples, not a controlled model-performance
comparison. The chain pairs have distances 3 (ten pairs), 4 (four), and
5 (one). Every redundant circuit contains the shared edge and at least four edges exclusive to each module. All certificates are unions of stabilizer edge orbits and retain negative base-edge circuit labels.

## Run

From the project root:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_chain_graphs.py" -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_laman*.py" -v
.\.venv\Scripts\python.exe scripts\build_chain_sample.py
```

Six chain tests and the existing eight Laman tests pass. The generator
writes `data/generated/chain_sample.json` and `.csv`
The JSON includes source filenames/hashes, module codes, attachment edges, vertex mappings, removed edges, records, diagnostics, and isomorphism groups.

New files: `src/chain_graphs.py`, `scripts/build_chain_sample.py`,
`tests/test_chain_graphs.py`, and this note. Update
`src/laman_diagnostics.py` for the optional candidate pools and distance filter.
The original small-sample command and its selected examples are unchanged.

## Limits and next step

- These examples block the seven-feature baseline on the selected balanced pairs. GIN and explainers have not been evaluated on them.
- Larger circuits and distance five do not establish a GIN limitation. Its readout also uses a pooled graph embedding.
- Single-edge deletion can break symmetry between identical modules. Measure the surviving candidate-preserving symmetry rather than assuming it.
- This is a small pilot. The current orbit utility enumerates automorphisms, so highly symmetric large assemblies need additional scaling work.
- No train/test splits are assigned. Partition source modules before larger generation, keep related assemblies together, and group isomorphic bases.
Then compare GIN accuracy and explanation quality by circuit size/distance.

[Laman criterion](https://pyrigi.github.io/PyRigi/math/rigidity/generic.html).

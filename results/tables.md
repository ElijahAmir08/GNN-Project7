# Results: symmetry-consistent GNN explanations for 2D rigidity (preliminary)

All runs on CPU. Numbers are mean ± std over 3 checkpoint seeds unless only one exists. Regenerate with `python scripts/make_tables.py`.

## 1. Datasets

| dataset | examples | base graphs | train / val / test | nodes | edges (mean) | circuit sizes (incl. candidate) | circuit share of base edges | 1-WL ceiling |
|---|---|---|---|---|---|---|---|---|
| edge_small | 1324 | 303 | 836 / 234 / 254 | 6–10 | 12.2 (6–16) | 6:350, 8:175, 10:85, 12:29, 14:17, 16:6 | 0.55 | 1.0000 |
| edge_medium | 5234 | 1198 | 3878 / 734 / 622 | 6–10 | 12.1 (6–16) | 6:1380, 8:700, 10:331, 12:137, 14:55, 16:14 | 0.56 | 1.0000 |

Labels are balanced 50/50 within every base graph and split; graph size and edge count alone give 50% accuracy.

## 2. Classification (independent vs. redundant proposed edge)

| model | expressivity | dataset | seeds | train acc | val acc | test acc | test F1 (redundant) | best epoch | sec / seed |
|---|---|---|---|---|---|---|---|---|---|
| features | none (no message passing) | medium | 3 | 0.854 ± 0.001 | 0.840 ± 0.004 | **0.773 ± 0.014** | 0.785 ± 0.011 | 5 ± 4 | 16 ± 0 |
| gcn | < 1-WL | medium | 3 | 0.999 ± 0.001 | 0.985 ± 0.003 | **0.975 ± 0.008** | 0.975 ± 0.008 | 51 ± 3 | 49 ± 1 |
| mpnn | ≤ 1-WL | medium | 3 | 1.000 ± 0.000 | 1.000 ± 0.000 | **1.000 ± 0.000** | 1.000 ± 0.000 | 8 ± 2 | 84 ± 3 |
| gin | = 1-WL | medium | 3 | 1.000 ± 0.000 | 1.000 ± 0.000 | **0.997 ± 0.001** | 0.997 ± 0.001 | 15 ± 1 | 50 ± 1 |
| dsgin | > 1-WL | medium | 3 | 1.000 ± 0.000 | 1.000 ± 0.000 | **0.998 ± 0.000** | 0.998 ± 0.000 | 12 ± 3 | 100 ± 13 |
| features | none | small | 3 | 0.846 ± 0.005 | 0.866 ± 0.004 | **0.833 ± 0.005** | 0.839 ± 0.003 | 33 ± 33 | 7 ± 0 |
| gin | = 1-WL | small | 3 | 1.000 ± 0.000 | 0.996 ± 0.000 | **0.992 ± 0.000** | 0.992 ± 0.000 | 78 ± 16 | 19 ± 0 |
| gin (first run) | = 1-WL | medium | 3 | 0.999 ± 0.001 | 1.000 ± 0.000 | **0.995 ± 0.001** | 0.995 ± 0.001 | 12 ± 3 | 53 ± 1 |

Feature baseline uses 7 hand-made features (n, |E|, 2n−3−|E|, endpoint degrees, common neighbours, endpoint distance). Single-feature lookup on 'common neighbours' alone reaches 0.84 on edge_small.

## 3. Explanation quality per model (all 300 test examples; circuit metrics on the 150 redundant ones)

Logits are temperature-scaled per checkpoint to a median validation margin of 4 before explaining (T ≈ 2–3 for all models). Caveat: soft-mask explainers start with every edge weighted ≈0.5. At that point ~50% of predictions flip for every model, and the GCN's median logit margin jumps from 9.5 to ~249 (its explainer gradient vanishes), so GCN explanations stay near chance. gin run A and run B are two identical-config training runs (3 seeds each).

k = |circuit| − 1 for redundant examples, k = 4 for independent. circuit precision = fraction of selected edges in the circuit; AUROC = ranking of circuit vs. non-circuit edges; fid− = drop in predicted-class probability when keeping only the selected edges (lower = more sufficient); fid+ = mean drop when deleting top-1..k edges (higher = more necessary).

| model | temperature | explainer | circuit AUROC ↑ | per-seed AUROC | circuit precision ↑ | fid− ↓ | fid+ ↑ |
|---|---|---|---|---|---|---|---|
| gcn | 2.8 ± 0.0 | oracle | 1.000 ± 0.000 | 1.00, 1.00, 1.00 | 1.000 ± 0.000 | 0.007 ± 0.002 | 0.426 ± 0.007 |
| gcn | 2.8 ± 0.0 | PGExplainer | 0.553 ± 0.072 | 0.50, 0.65, 0.50 | 0.607 ± 0.036 | 0.384 ± 0.029 | 0.334 ± 0.008 |
| gcn | 2.8 ± 0.0 | GNNExplainer | 0.510 ± 0.015 | 0.53, 0.51, 0.49 | 0.587 ± 0.012 | 0.402 ± 0.011 | 0.332 ± 0.017 |
| gcn | 2.8 ± 0.0 | random | 0.500 ± 0.000 | 0.50, 0.50, 0.50 | 0.543 ± 0.000 | 0.416 ± 0.009 | 0.352 ± 0.011 |
| mpnn | 2.1 ± 0.2 | oracle | 1.000 ± 0.000 | 1.00, 1.00, 1.00 | 1.000 ± 0.000 | 0.089 ± 0.013 | 0.460 ± 0.001 |
| mpnn | 2.1 ± 0.2 | PGExplainer | 0.516 ± 0.048 | 0.55, 0.56, 0.45 | 0.598 ± 0.026 | 0.522 ± 0.018 | 0.392 ± 0.031 |
| mpnn | 2.1 ± 0.2 | GNNExplainer | 0.735 ± 0.017 | 0.73, 0.76, 0.72 | 0.729 ± 0.012 | 0.442 ± 0.033 | 0.451 ± 0.008 |
| mpnn | 2.1 ± 0.2 | random | 0.500 ± 0.000 | 0.50, 0.50, 0.50 | 0.543 ± 0.000 | 0.505 ± 0.009 | 0.393 ± 0.005 |
| gin (run B) | 2.4 ± 0.1 | oracle | 1.000 ± 0.000 | 1.00, 1.00, 1.00 | 1.000 ± 0.000 | 0.035 ± 0.021 | 0.468 ± 0.003 |
| gin (run B) | 2.4 ± 0.1 | PGExplainer | 0.567 ± 0.333 | 0.11, 0.90, 0.69 | 0.619 ± 0.239 | 0.401 ± 0.098 | 0.346 ± 0.135 |
| gin (run B) | 2.4 ± 0.1 | GNNExplainer | 0.523 ± 0.049 | 0.58, 0.46, 0.53 | 0.583 ± 0.018 | 0.440 ± 0.017 | 0.392 ± 0.001 |
| gin (run B) | 2.4 ± 0.1 | random | 0.500 ± 0.000 | 0.50, 0.50, 0.50 | 0.543 ± 0.000 | 0.471 ± 0.005 | 0.404 ± 0.001 |
| gin (run A) | 2.3 ± 0.1 | oracle | 1.000 ± 0.000 | 1.00, 1.00, 1.00 | 1.000 ± 0.000 | 0.036 ± 0.015 | 0.468 ± 0.002 |
| gin (run A) | 2.3 ± 0.1 | PGExplainer | 0.814 ± 0.162 | 0.93, 0.93, 0.58 | 0.792 ± 0.139 | 0.276 ± 0.114 | 0.415 ± 0.045 |
| gin (run A) | 2.3 ± 0.1 | GNNExplainer | 0.583 ± 0.135 | 0.74, 0.41, 0.61 | 0.611 ± 0.101 | 0.464 ± 0.008 | 0.396 ± 0.055 |
| gin (run A) | 2.3 ± 0.1 | random | 0.500 ± 0.000 | 0.50, 0.50, 0.50 | 0.543 ± 0.000 | 0.478 ± 0.011 | 0.403 ± 0.003 |
| dsgin | 2.3 ± 0.1 | oracle | 1.000 ± 0.000 | 1.00, 1.00, 1.00 | 1.000 ± 0.000 | 0.001 ± 0.009 | 0.475 ± 0.001 |
| dsgin | 2.3 ± 0.1 | PGExplainer | 0.527 ± 0.280 | 0.23, 0.90, 0.45 | 0.568 ± 0.218 | 0.376 ± 0.122 | 0.341 ± 0.083 |
| dsgin | 2.3 ± 0.1 | GNNExplainer | 0.676 ± 0.118 | 0.84, 0.57, 0.62 | 0.682 ± 0.089 | 0.402 ± 0.065 | 0.446 ± 0.015 |
| dsgin | 2.3 ± 0.1 | random | 0.500 ± 0.000 | 0.50, 0.50, 0.50 | 0.543 ± 0.000 | 0.468 ± 0.003 | 0.406 ± 0.001 |

## 4. Symmetry corrections: raw vs. orbit averaging vs. whole-orbit selection

Restricted to test examples with a nontrivial candidate-stabilizer orbit. orbit split = % of selections that include part (not all) of an orbit; overshoot = extra edges whole-orbit selection adds beyond k.

| model | explainer | variant | circuit AUROC | circuit precision | fid− | fid+ | orbit split % ↓ | overshoot |
|---|---|---|---|---|---|---|---|---|
| gcn | PGExplainer | raw | 0.533 ± 0.047 | 0.566 ± 0.019 | 0.485 ± 0.009 | 0.340 ± 0.003 | 66.1 ± 1.6 | 0.00 ± 0.00 |
| gcn | PGExplainer | orbit_avg | 0.533 ± 0.047 | 0.566 ± 0.019 | 0.485 ± 0.009 | 0.340 ± 0.003 | 66.1 ± 1.6 | 0.00 ± 0.00 |
| gcn | PGExplainer | orbit_sel | 0.533 ± 0.047 | 0.604 ± 0.006 | 0.340 ± 0.035 | 0.332 ± 0.003 | 0.0 ± 0.0 | 0.25 ± 0.02 |
| gcn | GNNExplainer | raw | 0.516 ± 0.023 | 0.568 ± 0.023 | 0.478 ± 0.030 | 0.357 ± 0.032 | 74.7 ± 4.3 | 0.00 ± 0.00 |
| gcn | GNNExplainer | orbit_avg | 0.516 ± 0.022 | 0.573 ± 0.029 | 0.460 ± 0.048 | 0.354 ± 0.023 | 50.0 ± 1.4 | 0.00 ± 0.00 |
| gcn | GNNExplainer | orbit_sel | 0.516 ± 0.022 | 0.612 ± 0.018 | 0.326 ± 0.012 | 0.349 ± 0.025 | 0.0 ± 0.0 | 0.28 ± 0.04 |
| gcn | random | raw | 0.502 ± 0.000 | 0.567 ± 0.000 | 0.487 ± 0.007 | 0.395 ± 0.016 | 79.3 ± 0.0 | 0.00 ± 0.00 |
| gcn | random | orbit_avg | 0.523 ± 0.000 | 0.559 ± 0.000 | 0.436 ± 0.010 | 0.360 ± 0.016 | 37.9 ± 0.0 | 0.00 ± 0.00 |
| gcn | random | orbit_sel | 0.523 ± 0.000 | 0.559 ± 0.000 | 0.400 ± 0.009 | 0.360 ± 0.016 | 0.0 ± 0.0 | 0.38 ± 0.00 |
| mpnn | PGExplainer | raw | 0.521 ± 0.028 | 0.598 ± 0.021 | 0.518 ± 0.016 | 0.408 ± 0.023 | 46.6 ± 6.5 | 0.00 ± 0.00 |
| mpnn | PGExplainer | orbit_avg | 0.521 ± 0.028 | 0.598 ± 0.021 | 0.518 ± 0.016 | 0.408 ± 0.023 | 46.6 ± 6.5 | 0.00 ± 0.00 |
| mpnn | PGExplainer | orbit_sel | 0.521 ± 0.028 | 0.609 ± 0.020 | 0.501 ± 0.016 | 0.407 ± 0.023 | 0.0 ± 0.0 | 0.32 ± 0.02 |
| mpnn | GNNExplainer | raw | 0.763 ± 0.018 | 0.764 ± 0.004 | 0.413 ± 0.032 | 0.474 ± 0.013 | 56.3 ± 2.2 | 0.00 ± 0.00 |
| mpnn | GNNExplainer | orbit_avg | 0.767 ± 0.025 | 0.775 ± 0.014 | 0.402 ± 0.024 | 0.466 ± 0.018 | 18.4 ± 2.2 | 0.00 ± 0.00 |
| mpnn | GNNExplainer | orbit_sel | 0.767 ± 0.025 | 0.771 ± 0.016 | 0.389 ± 0.018 | 0.466 ± 0.018 | 0.0 ± 0.0 | 0.21 ± 0.04 |
| mpnn | random | raw | 0.502 ± 0.000 | 0.567 ± 0.000 | 0.527 ± 0.006 | 0.434 ± 0.007 | 79.3 ± 0.0 | 0.00 ± 0.00 |
| mpnn | random | orbit_avg | 0.523 ± 0.000 | 0.559 ± 0.000 | 0.517 ± 0.008 | 0.410 ± 0.008 | 37.9 ± 0.0 | 0.00 ± 0.00 |
| mpnn | random | orbit_sel | 0.523 ± 0.000 | 0.559 ± 0.000 | 0.470 ± 0.009 | 0.410 ± 0.008 | 0.0 ± 0.0 | 0.38 ± 0.00 |
| gin (run B) | PGExplainer | raw | 0.590 ± 0.358 | 0.632 ± 0.248 | 0.392 ± 0.171 | 0.378 ± 0.142 | 37.9 ± 11.3 | 0.00 ± 0.00 |
| gin (run B) | PGExplainer | orbit_avg | 0.590 ± 0.358 | 0.632 ± 0.248 | 0.392 ± 0.171 | 0.378 ± 0.142 | 37.9 ± 11.3 | 0.00 ± 0.00 |
| gin (run B) | PGExplainer | orbit_sel | 0.590 ± 0.358 | 0.655 ± 0.239 | 0.327 ± 0.154 | 0.375 ± 0.140 | 0.0 ± 0.0 | 0.27 ± 0.06 |
| gin (run B) | GNNExplainer | raw | 0.542 ± 0.054 | 0.608 ± 0.027 | 0.473 ± 0.011 | 0.413 ± 0.006 | 62.6 ± 4.3 | 0.00 ± 0.00 |
| gin (run B) | GNNExplainer | orbit_avg | 0.558 ± 0.051 | 0.620 ± 0.024 | 0.459 ± 0.015 | 0.407 ± 0.006 | 25.3 ± 3.3 | 0.00 ± 0.00 |
| gin (run B) | GNNExplainer | orbit_sel | 0.558 ± 0.051 | 0.628 ± 0.022 | 0.416 ± 0.007 | 0.407 ± 0.006 | 0.0 ± 0.0 | 0.26 ± 0.04 |
| gin (run B) | random | raw | 0.502 ± 0.000 | 0.567 ± 0.000 | 0.511 ± 0.005 | 0.449 ± 0.004 | 79.3 ± 0.0 | 0.00 ± 0.00 |
| gin (run B) | random | orbit_avg | 0.523 ± 0.000 | 0.559 ± 0.000 | 0.495 ± 0.008 | 0.427 ± 0.006 | 37.9 ± 0.0 | 0.00 ± 0.00 |
| gin (run B) | random | orbit_sel | 0.523 ± 0.000 | 0.559 ± 0.000 | 0.445 ± 0.008 | 0.427 ± 0.006 | 0.0 ± 0.0 | 0.38 ± 0.00 |
| gin (run A) | PGExplainer | raw | 0.834 ± 0.153 | 0.809 ± 0.103 | 0.273 ± 0.159 | 0.453 ± 0.044 | 35.1 ± 5.9 | 0.00 ± 0.00 |
| gin (run A) | PGExplainer | orbit_avg | 0.834 ± 0.153 | 0.809 ± 0.103 | 0.273 ± 0.159 | 0.453 ± 0.044 | 35.1 ± 5.9 | 0.00 ± 0.00 |
| gin (run A) | PGExplainer | orbit_sel | 0.834 ± 0.153 | 0.808 ± 0.108 | 0.271 ± 0.157 | 0.453 ± 0.044 | 0.0 ± 0.0 | 0.20 ± 0.08 |
| gin (run A) | GNNExplainer | raw | 0.588 ± 0.126 | 0.624 ± 0.095 | 0.477 ± 0.015 | 0.421 ± 0.043 | 59.2 ± 9.9 | 0.00 ± 0.00 |
| gin (run A) | GNNExplainer | orbit_avg | 0.590 ± 0.124 | 0.633 ± 0.092 | 0.460 ± 0.027 | 0.415 ± 0.043 | 24.1 ± 5.1 | 0.00 ± 0.00 |
| gin (run A) | GNNExplainer | orbit_sel | 0.590 ± 0.124 | 0.629 ± 0.085 | 0.433 ± 0.009 | 0.415 ± 0.043 | 0.0 ± 0.0 | 0.24 ± 0.05 |
| gin (run A) | random | raw | 0.502 ± 0.000 | 0.567 ± 0.000 | 0.509 ± 0.006 | 0.449 ± 0.005 | 79.3 ± 0.0 | 0.00 ± 0.00 |
| gin (run A) | random | orbit_avg | 0.523 ± 0.000 | 0.559 ± 0.000 | 0.492 ± 0.004 | 0.427 ± 0.005 | 37.9 ± 0.0 | 0.00 ± 0.00 |
| gin (run A) | random | orbit_sel | 0.523 ± 0.000 | 0.559 ± 0.000 | 0.446 ± 0.003 | 0.427 ± 0.005 | 0.0 ± 0.0 | 0.38 ± 0.00 |
| dsgin | PGExplainer | raw | 0.573 ± 0.284 | 0.592 ± 0.205 | 0.380 ± 0.159 | 0.377 ± 0.091 | 37.9 ± 6.1 | 0.00 ± 0.00 |
| dsgin | PGExplainer | orbit_avg | 0.573 ± 0.284 | 0.592 ± 0.205 | 0.380 ± 0.159 | 0.377 ± 0.091 | 37.9 ± 6.1 | 0.00 ± 0.00 |
| dsgin | PGExplainer | orbit_sel | 0.573 ± 0.284 | 0.609 ± 0.197 | 0.375 ± 0.157 | 0.378 ± 0.090 | 0.0 ± 0.0 | 0.24 ± 0.07 |
| dsgin | GNNExplainer | raw | 0.677 ± 0.119 | 0.682 ± 0.079 | 0.429 ± 0.083 | 0.479 ± 0.017 | 66.1 ± 4.5 | 0.00 ± 0.00 |
| dsgin | GNNExplainer | orbit_avg | 0.690 ± 0.133 | 0.709 ± 0.093 | 0.383 ± 0.113 | 0.477 ± 0.017 | 21.8 ± 6.5 | 0.00 ± 0.00 |
| dsgin | GNNExplainer | orbit_sel | 0.690 ± 0.133 | 0.709 ± 0.092 | 0.335 ± 0.090 | 0.477 ± 0.017 | 0.0 ± 0.0 | 0.22 ± 0.07 |
| dsgin | random | raw | 0.502 ± 0.000 | 0.567 ± 0.000 | 0.502 ± 0.007 | 0.449 ± 0.005 | 79.3 ± 0.0 | 0.00 ± 0.00 |
| dsgin | random | orbit_avg | 0.523 ± 0.000 | 0.559 ± 0.000 | 0.482 ± 0.010 | 0.424 ± 0.006 | 37.9 ± 0.0 | 0.00 ± 0.00 |
| dsgin | random | orbit_sel | 0.523 ± 0.000 | 0.559 ± 0.000 | 0.435 ± 0.006 | 0.424 ± 0.006 | 0.0 ± 0.0 | 0.38 ± 0.00 |

## 5. Symmetry structure and GNNExplainer stability

| model | examples with nontrivial orbit | circuit = union of orbits | GNNExpl. Spearman: same graph, new seed | GNNExpl. Spearman: relabeled graph | mean max |Δscore| after relabeling |
|---|---|---|---|---|---|
| gcn | 19.3 ± 0.0% | 100.0 ± 0.0% | 0.727 ± 0.076 | 0.593 ± 0.109 | 0.140 ± 0.068 |
| mpnn | 19.3 ± 0.0% | 100.0 ± 0.0% | 0.612 ± 0.071 | 0.379 ± 0.099 | 0.436 ± 0.080 |
| gin (run B) | 19.3 ± 0.0% | 100.0 ± 0.0% | 0.516 ± 0.062 | 0.318 ± 0.051 | 0.498 ± 0.044 |
| gin (run A) | 19.3 ± 0.0% | 100.0 ± 0.0% | 0.509 ± 0.049 | 0.292 ± 0.043 | 0.519 ± 0.023 |
| dsgin | 19.3 ± 0.0% | 100.0 ± 0.0% | 0.483 ± 0.050 | 0.275 ± 0.058 | 0.535 ± 0.055 |


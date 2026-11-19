# Metamorphic rule tests (violation %, lower is better; 0 = obeys the law)

add_edges: redundant stays redundant when bars are added (never trained on). delete: independent stays independent when bars are removed (aligned with masked training). one_sum: attaching a Laman graph at one vertex never changes the prediction. Number = bars added/removed or attached-graph size. Sources: hard_eval, chains, ood_suite.

| run | add_edges 2 | add_edges 6 | delete 2 | delete 6 | one_sum 8 | one_sum 20 |
|---|---|---|---|---|---|---|
| randmask_gcn | 6.2 ± 1.6 | 7.2 ± 2.9 | 10.4 ± 3.7 | 6.2 ± 3.4 | 6.3 ± 1.2 | 12.5 ± 3.5 |
| randmask_mpnn | 0.3 ± 0.3 | 0.3 ± 0.1 | 0.3 ± 0.1 | 0.1 ± 0.1 | 6.4 ± 1.2 | 21.9 ± 2.8 |
| randmask_gin | 0.1 ± 0.1 | 0.1 ± 0.1 | 0.8 ± 0.6 | 0.7 ± 0.8 | 4.5 ± 0.6 | 7.5 ± 0.9 |
| randmask_dsgin | 0.0 ± 0.1 | 0.0 ± 0.0 | 0.8 ± 0.4 | 0.3 ± 0.2 | 4.9 ± 0.8 | 12.1 ± 3.9 |

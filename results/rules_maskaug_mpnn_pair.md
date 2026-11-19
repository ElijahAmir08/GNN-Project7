# Metamorphic rule tests (violation %, lower is better; 0 = obeys the law)

add_edges: redundant stays redundant when bars are added (never trained on). delete: independent stays independent when bars are removed (aligned with masked training). one_sum: attaching a Laman graph at one vertex never changes the prediction. Number = bars added/removed or attached-graph size. Sources: hard_eval, chains, ood_suite.

| run | add_edges 2 | add_edges 6 | delete 2 | delete 6 | one_sum 8 | one_sum 20 |
|---|---|---|---|---|---|---|
| maskaug_mpnn_pair | 0.1 ± 0.2 | 0.0 ± 0.1 | 0.6 ± 0.2 | 0.2 ± 0.1 | 5.7 ± 1.1 | 9.4 ± 0.9 |
| maskaug_gin_pair | 0.1 ± 0.1 | 0.0 ± 0.1 | 1.2 ± 0.4 | 0.2 ± 0.1 | 7.3 ± 0.8 | 12.3 ± 1.6 |
| maskaug_dsgin_pair | 0.1 ± 0.1 | 2.9 ± 3.9 | 0.3 ± 0.2 | 0.1 ± 0.1 | 5.7 ± 1.5 | 11.0 ± 3.8 |

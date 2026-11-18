#!/usr/bin/env bash
# Full evaluation for trained runs: accuracy on all test/stress sets, then
# explainers (GNNExplainer, PGExplainer, EdgeGrad, random, oracle) and exact
# oracle grading on the hard families and Elijah's chains.
#
# Usage: scripts/run_eval_sweep.sh TRAIN_DATA RUN [RUN ...]
#   TRAIN_DATA  training JSON used for leakage checks and as --fit-data
#   RUN         results/<RUN>/seed*/model.pt must exist
# Outputs: results/eval_<first-run>.md (accuracy), results/oracle_<first-run>.md
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
TRAIN=$1; shift
RUNS=("$@")
TAG=${RUNS[0]}
HARD=data_new/generated/hard_eval.json
CHAIN=data_new/generated/chain_matched_eval.json
LARGE=data_new/generated/henneberg_large_eval.json

$PY -W ignore scripts/eval_checkpoints.py --runs "${RUNS[@]}" --train-data "$TRAIN" \
    --data data/generated/edge_medium.json "$HARD" "$CHAIN" "$LARGE" --out "results/eval_${TAG}.md"

graph_runs=()
for r in "${RUNS[@]}"; do [[ $r == *features* ]] || graph_runs+=("$r"); done

# Explainers: one process per (run, seed), at most 5 in parallel.
jobs_running=0
for r in "${graph_runs[@]}"; do
  for ck in results/"$r"/seed*/model.pt; do
    s=$(basename "$(dirname "$ck")"); s=${s#seed}
    for data in "$HARD" "$CHAIN"; do
      n=$($PY -c "import json,sys; print(sum(r['split']=='test' for r in json.load(open(sys.argv[1]))['records'])//2)" "$data")
      $PY -W ignore scripts/explain_prelim.py --run "results/$r" --seed-ckpt "$s" --data "$data" \
          --fit-data "$TRAIN" --n-per-class "$n" --pg-epochs 30 --pg-train 600 --consistency-n 20 \
          --workers 32 > "results/log_explain_${r}_s${s}_$(basename "$data" .json).txt" 2>&1 &
      jobs_running=$((jobs_running + 1))
      if (( jobs_running >= 5 )); then wait -n; jobs_running=$((jobs_running - 1)); fi
    done
  done
done
wait
OMP_NUM_THREADS=8 $PY -W ignore scripts/oracle_eval.py --runs "${graph_runs[@]}" \
    --datasets _chain_matched_eval _hard_eval --out "results/oracle_${TAG}.md"
echo "done: results/eval_${TAG}.md results/oracle_${TAG}.md"

#!/bin/bash
# Full evaluation across all checkpoints, conditions, and valid_iters.
# Saves per-run JSON and aggregates to result.csv.

set -e

cd "$(dirname "$0")/.."
source /home/wuzr/miniconda3/bin/activate ffs

MODELS=(
    "weights/15-44-51/model_best_bp2_serialize.pth"
    "weights/20-26-39/model_best_bp2_serialize.pth"
    "weights/20-30-48/model_best_bp2_serialize.pth"
    "weights/23-36-37/model_best_bp2_serialize.pth"
)
ITERS=(4 8)
OUT_BASE="output/full_eval"
mkdir -p "$OUT_BASE"

CSV="$OUT_BASE/result.csv"
echo "model,valid_iters,condition,num_samples,epe,bad_1,bad_3,d1_all,rmse" > "$CSV"

for model_path in "${MODELS[@]}"; do
    model_name=$(basename "$(dirname "$model_path")")
    echo ""
    echo "================================================================"
    echo "  Model: $model_name"
    echo "================================================================"

    for iters in "${ITERS[@]}"; do
        echo "--- valid_iters=$iters ---"

        run_dir="$OUT_BASE/${model_name}_iters${iters}"
        mkdir -p "$run_dir"

        python scripts/eval_drivingstereo.py \
            --model_dir "$model_path" \
            --dataset_dir data \
            --out_dir "$run_dir" \
            --valid_iters "$iters" \
            2>&1 | tee "$run_dir/log.txt"

        # Extract per-condition and overall results from JSON
        python3 -c "
import json
with open('$run_dir/results.json') as f:
    d = json.load(f)
model = '$model_name'
iters = $iters
# Per-condition
for cond, m in d['per_condition'].items():
    print(f'{model},{iters},{cond},{d[\"num_samples\"]},{m[\"epe\"]:.4f},{m[\"bad_1\"]:.2f},{m[\"bad_3\"]:.2f},{m[\"d1_all\"]:.2f},{m[\"rmse\"]:.4f}')
# Overall
o = d['overall']
print(f'{model},{iters},OVERALL,{d[\"num_samples\"]},{o[\"epe\"]:.4f},{o[\"bad_1\"]:.2f},{o[\"bad_3\"]:.2f},{o[\"d1_all\"]:.2f},{o[\"rmse\"]:.4f}')
" >> "$CSV"

    done
done

echo ""
echo "================================================================"
echo "  All evaluations complete!"
echo "  Results: $CSV"
echo "================================================================"
cat "$CSV"

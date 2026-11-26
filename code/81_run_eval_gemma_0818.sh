#!/bin/bash
# Adapted from 81_run_eval.sh for 0818 continuation:
# - GPU assignment: 2 GPUs (3,0) batch 4, with automatic fallback to 1 GPU (3) + batch_size 1
# - safe tmpdir cleanup (no rm -rf per user policy)
set -u
PYTHON="<REPO_ROOT>/.venv/bin/python"
source "<REPO_ROOT>"/.venv/bin/activate
cd "<REPO_ROOT>"/code

export NCCL_P2P_DISABLE=0
export VLLM_ENABLE_V1_MULTIPROCESSING=0

EVAL_SCRIPT="80_llm_score_with_ci.py"
JUDGE_MODEL="<MODELS_DIR>/Qwen2.5-72B-Instruct-AWQ"
OUT_BASE="./t-test_results_gemma_mmlu/res"

BASE="<REPO_ROOT>/results/res/gemma"

evaluate_file() {
    local file_path=$1
    local principle=$2
    local tmpdir=$(mktemp -d)
    cp "$file_path" "$tmpdir/"

    echo "[$(date +%H:%M:%S)] Evaluating: $(basename $file_path) / $principle"
    # Attempt 1: 2 GPUs (3,0), batch_size 4
    export CUDA_VISIBLE_DEVICES=3,0
    $PYTHON $EVAL_SCRIPT \
        --model $JUDGE_MODEL \
        --input_path "$tmpdir" \
        --out_path "$OUT_BASE" \
        --principle $principle \
        --batch_size 4
    if [ $? -ne 0 ]; then
        echo "[$(date +%H:%M:%S)] WARNING: $principle failed with 2 GPUs (batch 4); falling back to 1 GPU (3) batch_size 1..."
        sleep 30
        export CUDA_VISIBLE_DEVICES=3
        $PYTHON $EVAL_SCRIPT \
            --model $JUDGE_MODEL \
            --input_path "$tmpdir" \
            --out_path "$OUT_BASE" \
            --principle $principle \
            --batch_size 1
        if [ $? -ne 0 ]; then
            echo "[$(date +%H:%M:%S)] ERROR: $principle failed on both 2-GPU and 1-GPU attempts; aborting"
            find "$tmpdir" -type f -delete 2>/dev/null || true
            rmdir "$tmpdir" 2>/dev/null || true
            exit 1
        fi
    fi
    # safe cleanup
    find "$tmpdir" -type f -delete 2>/dev/null || true
    rmdir "$tmpdir" 2>/dev/null || true
}

# 1. Union Exp_Info hh coeff=1.5
F1="$BASE/answer_union_steer/Expertise_high.Informativeness_high/Expertise_high.Informativeness_high_1.5.json"
evaluate_file "$F1" "expert"
evaluate_file "$F1" "informative"
evaluate_file "$F1" "fluency"

# 2. Union Exp_Style lh coeff=2.5
F2="$BASE/answer_union_steer/Expertise_low.Style_high/Expertise_low.Style_high_2.5.json"
evaluate_file "$F2" "expert"
evaluate_file "$F2" "style"
evaluate_file "$F2" "fluency"

# 3. Both Exp_Info hh coeff=1.5
F3="$BASE/answer_both_steer/Expertise_high.Informativeness_high/Expertise_high.Informativeness_high_1.5.json"
evaluate_file "$F3" "expert"
evaluate_file "$F3" "informative"
evaluate_file "$F3" "fluency"

echo ""
echo "[$(date +%H:%M:%S)] All 9 evaluations complete!"

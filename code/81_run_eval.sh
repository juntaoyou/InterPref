#!/bin/bash
# Evaluate only the 3 new steering files.
PYTHON="<REPO_ROOT>/.venv/bin/python"
source "<REPO_ROOT>"/.venv/bin/activate
cd "<REPO_ROOT>"/code

export CUDA_VISIBLE_DEVICES=4,5
export NCCL_P2P_DISABLE=0
export VLLM_ENABLE_V1_MULTIPROCESSING=0

EVAL_SCRIPT="80_llm_score_with_ci.py"
JUDGE_MODEL="<MODELS_DIR>/Qwen2.5-72B-Instruct-AWQ"
BATCH_SIZE=4
OUT_BASE="./t-test_results_gemma_mmlu/res"

BASE="<REPO_ROOT>/results/res/gemma"

evaluate_file() {
    local file_path=$1
    local principle=$2
    local tmpdir=$(mktemp -d)
    cp "$file_path" "$tmpdir/"

    echo "[$(date +%H:%M:%S)] Evaluating: $(basename $file_path) / $principle"
    $PYTHON $EVAL_SCRIPT \
        --model $JUDGE_MODEL \
        --input_path "$tmpdir" \
        --out_path "$OUT_BASE" \
        --principle $principle \
        --batch_size $BATCH_SIZE

    rm -rf "$tmpdir"
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


# 所有文件都需要评估fluency
# 含有expertise的文件需要评估expert
# 含有informativeness的文件需要评估informative
# 含有style的文件需要评估style
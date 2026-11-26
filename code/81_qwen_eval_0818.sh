#!/bin/bash
# Evaluate qwen with_preferences generation results (partial1/partial2 steer) with 80_llm_score_with_ci.py
# Rules:
#   - all files: fluency
#   - files containing Expertise: expert
#   - files containing Informativeness: informative
#   - files containing Style: style
# Grouped by principle (one judge-model load per principle; 80_llm_score_with_ci.py loops over all files in input_path).
# NOTE: partial1/partial2 produce identical filenames -> prefix with mode when staging to avoid overwrite.
set -u
PYTHON="<REPO_ROOT>/.venv/bin/python"
source "<REPO_ROOT>"/.venv/bin/activate
cd "<REPO_ROOT>"/code

export CUDA_VISIBLE_DEVICES=3,0
export NCCL_P2P_DISABLE=0
export VLLM_ENABLE_V1_MULTIPROCESSING=0

EVAL_SCRIPT="80_llm_score_with_ci.py"
JUDGE_MODEL="<MODELS_DIR>/Qwen2.5-72B-Instruct-AWQ"
BATCH_SIZE=4
OUT_BASE="./t-test_results_qwen_mmlu/res"

BASE="<REPO_ROOT>/res/qwen/psoups"

# Collect all generated answer files as "mode|path" pairs
ITEMS=()
for MODE in partial1 partial2; do
  DIR="$BASE/answer_with_preferences_${MODE}_steer"
  for F in "$DIR"/*/*.json; do
    [ -f "$F" ] && ITEMS+=("$MODE|$F")
  done
done
echo "[INFO] total files to evaluate: ${#ITEMS[@]}"

evaluate_principle() {
    local principle=$1
    shift
    local items=("$@")
    if [ ${#items[@]} -eq 0 ]; then
        echo "[$(date +%H:%M:%S)] no files for principle: $principle"
        return
    fi
    local tmpdir=$(mktemp -d)
    local n=0
    for item in "${items[@]}"; do
        local mode="${item%%|*}"
        local f="${item#*|}"
        cp "$f" "$tmpdir/${mode}__$(basename "$f")"
        n=$((n+1))
    done
    echo "[$(date +%H:%M:%S)] Evaluating principle: $principle | files: $n"
    local ok=0
    # Attempt 1: 2 GPUs (3,0), batch_size 4
    export CUDA_VISIBLE_DEVICES=3,0
    $PYTHON $EVAL_SCRIPT \
        --model $JUDGE_MODEL \
        --input_path "$tmpdir" \
        --out_path "$OUT_BASE" \
        --principle $principle \
        --batch_size 4
    if [ $? -eq 0 ]; then
        ok=1
    else
        echo "[$(date +%H:%M:%S)] WARNING: principle $principle failed with 2 GPUs (batch 4); falling back to 1 GPU (batch_size 1)..."
        sleep 30
        # Attempt 2: 1 GPU (3), batch_size 1
        export CUDA_VISIBLE_DEVICES=3
        $PYTHON $EVAL_SCRIPT \
            --model $JUDGE_MODEL \
            --input_path "$tmpdir" \
            --out_path "$OUT_BASE" \
            --principle $principle \
            --batch_size 1
        if [ $? -eq 0 ]; then
            ok=1
        fi
    fi
    if [ $ok -ne 1 ]; then
        echo "[$(date +%H:%M:%S)] ERROR: principle $principle failed on both 2-GPU and 1-GPU attempts; aborting"
        find "$tmpdir" -type f -delete 2>/dev/null || true
        rmdir "$tmpdir" 2>/dev/null || true
        exit 1
    fi
    echo "[$(date +%H:%M:%S)] done principle: $principle"
    # safe cleanup (no rm -rf per user policy)
    find "$tmpdir" -type f -delete 2>/dev/null || true
    rmdir "$tmpdir" 2>/dev/null || true
}

FLUENCY=(); EXPERT=(); INFORMATIVE=(); STYLE=()
for item in "${ITEMS[@]}"; do
    NAME=$(basename "${item#*|}")
    FLUENCY+=("$item")
    [[ "$NAME" == *Expertise* ]] && EXPERT+=("$item")
    [[ "$NAME" == *Informativeness* ]] && INFORMATIVE+=("$item")
    [[ "$NAME" == *Style* ]] && STYLE+=("$item")
done

echo "[INFO] fluency: ${#FLUENCY[@]} | expert: ${#EXPERT[@]} | informative: ${#INFORMATIVE[@]} | style: ${#STYLE[@]}"

evaluate_principle "fluency" "${FLUENCY[@]}"
evaluate_principle "expert" "${EXPERT[@]}"
evaluate_principle "informative" "${INFORMATIVE[@]}"
evaluate_principle "style" "${STYLE[@]}"

echo ""
echo "[$(date +%H:%M:%S)] All qwen evaluations complete!"

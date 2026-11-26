#!/bin/bash
# Evaluate Scheme-A partialA1/partialA2 steer answers (qwen) with 80_llm_score_with_ci.py.
# Rules: all files -> fluency; Expertise -> expert; Informativeness -> informative; Style -> style.
# GPUs 1,3 (TP=2 batch 4) with fallback to 1 GPU + batch_size 1 if needed.
set -u
MODEL_NAME="qwen"
PYTHON="<REPO_ROOT>/.venv/bin/python"
source "<REPO_ROOT>"/.venv/bin/activate
cd "<REPO_ROOT>"/code

export NCCL_P2P_DISABLE=0
export VLLM_ENABLE_V1_MULTIPROCESSING=0

EVAL_SCRIPT="80_llm_score_with_ci.py"
JUDGE_MODEL="<MODELS_DIR>/Qwen2.5-72B-Instruct-AWQ"
OUT_BASE="./t-test_results_qwen_mmlu/res"

BASE="<REPO_ROOT>/res/qwen/psoups"

ITEMS=()
for MODE in dual; do
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
    export CUDA_VISIBLE_DEVICES=2,7
    $PYTHON $EVAL_SCRIPT \
        --model $JUDGE_MODEL \
        --input_path "$tmpdir" \
        --out_path "$OUT_BASE" \
        --principle $principle \
        --batch_size 4
    if [ $? -eq 0 ]; then
        ok=1
    else
        echo "[$(date +%H:%M:%S)] WARNING: $principle failed with 2 GPUs; falling back to 1 GPU batch_size 1..."
        sleep 30
        export CUDA_VISIBLE_DEVICES=2
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
        echo "[$(date +%H:%M:%S)] ERROR: $principle failed on both attempts; aborting"
        find "$tmpdir" -type f -delete 2>/dev/null || true
        rmdir "$tmpdir" 2>/dev/null || true
        exit 1
    fi
    echo "[$(date +%H:%M:%S)] done principle: $principle"
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
echo "[$(date +%H:%M:%S)] All A1/A2 evaluations complete!"

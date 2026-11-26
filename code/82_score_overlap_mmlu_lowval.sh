#!/bin/bash
# Score the low-strength mmlu overlap answers (staged in /tmp/overlap_score).
# Per preference: preference dimension + fluency.
set -e
PYTHON="<REPO_ROOT>/.venv/bin/python"
source "<REPO_ROOT>"/.venv/bin/activate
export CUDA_VISIBLE_DEVICES=4
export VLLM_ENABLE_V1_MULTIPROCESSING=0

JUDGE="<MODELS_DIR>/Qwen2.5-72B-Instruct-AWQ"
SCORER="<REPO_ROOT>/code/80_llm_score_memfix.py"
OUT_BASE="<REPO_ROOT>/LLM_results/qwen_overlap_mmlu_lowval"
IN_ROOT="/tmp/overlap_score"

declare -A PRINCIPLE=(
  [Expertise_high]="expert"
  [Informativeness_low]="informative"
  [Style_high]="style"
  [Style_low]="style"
)

cd "<REPO_ROOT>"/code
for pref in Expertise_high Informativeness_low Style_high Style_low; do
  dir="${IN_ROOT}/${pref}"
  [ -d "$dir" ] || { echo "MISSING $dir"; continue; }
  for p in "${PRINCIPLE[$pref]}" "fluency"; do
    echo "[$(date '+%H:%M:%S')] Scoring ${pref} with principle ${p}"
    $PYTHON $SCORER \
        --model $JUDGE \
        --input_path "$dir" \
        --out_path "$OUT_BASE" \
        --principle "$p" \
        --batch_size 8 \
        --num_runs 1
  done
done
echo "DONE low-val mmlu overlap scoring"

#!/bin/bash
# Score overlap-steer answers with preference-specific principles.
#   Expert  -> principle 'expert'
#   Info    -> principle 'informative'
#   Style   -> principle 'style'
# Runs after generation finishes (uses GPU 6).
set -e
PYTHON="<REPO_ROOT>/.venv/bin/python"
source "<REPO_ROOT>"/.venv/bin/activate
export CUDA_VISIBLE_DEVICES=6
export VLLM_ENABLE_V1_MULTIPROCESSING=0

JUDGE="<MODELS_DIR>/Qwen2.5-72B-Instruct-AWQ"
SCORER="<REPO_ROOT>/code/80_llm_score_memfix.py"
OUT_BASE="<REPO_ROOT>/LLM_results/qwen_overlap_mmlu"

# mmlu overlap answers (delta=4.0, one val file per pref)
GEN_DIR="<REPO_ROOT>/qwen/mmlu_test/answer_mmlu_neuron_overlap_4.0"

declare -A PRINCIPLE=(
  [Expertise_high]="expert"
  [Expertise_low]="expert"
  [Informativeness_high]="informative"
  [Informativeness_low]="informative"
  [Style_high]="style"
  [Style_low]="style"
)

cd "<REPO_ROOT>"/code
for pref in Expertise_high Expertise_low Informativeness_high Informativeness_low Style_high Style_low; do
  dir="${GEN_DIR}/${pref}"
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
echo "DONE mmlu overlap scoring"

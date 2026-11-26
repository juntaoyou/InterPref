#!/bin/bash
# Score the neutral (None prompt) baselines for both psoups (49q) and mmlu_test (570q)
# with the 4 principles, so deltas can be computed consistently.
set -e
PYTHON="<REPO_ROOT>/.venv/bin/python"
source "<REPO_ROOT>"/.venv/bin/activate
export CUDA_VISIBLE_DEVICES=7
export VLLM_ENABLE_V1_MULTIPROCESSING=0

JUDGE="<MODELS_DIR>/Qwen2.5-72B-Instruct-AWQ"
SCORER="<REPO_ROOT>/code/80_llm_score_memfix.py"
IN_ROOT="/tmp/neutral_score"

for ds in psoups mmlu; do
  dir="${IN_ROOT}/${ds}"
  out="<REPO_ROOT>/LLM_results/neutral_${ds}"
  for p in expert informative style fluency; do
    echo "[$(date '+%H:%M:%S')] Neutral ${ds} principle ${p}"
    $PYTHON $SCORER \
        --model $JUDGE \
        --input_path "$dir" \
        --out_path "$out" \
        --principle "$p" \
        --batch_size 8 \
        --num_runs 1
  done
done
echo "DONE neutral scoring"

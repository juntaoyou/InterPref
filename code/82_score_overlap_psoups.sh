#!/bin/bash
# Score the existing psoups overlap answers (eval/qwen/Overlap_psoups, 49q)
# as the psoups reference, using the same principles as the mmlu overlap.
# llm_score_memfix scores ALL json files in input_path with one principle,
# so we run it once per principle and pick the per-preference file after.
set -e
PYTHON="<REPO_ROOT>/.venv/bin/python"
source "<REPO_ROOT>"/.venv/bin/activate
export CUDA_VISIBLE_DEVICES=6
export VLLM_ENABLE_V1_MULTIPROCESSING=0

JUDGE="<MODELS_DIR>/Qwen2.5-72B-Instruct-AWQ"
SCORER="<REPO_ROOT>/code/80_llm_score_memfix.py"
OUT_BASE="<REPO_ROOT>/LLM_results/qwen_overlap_psoups"
IN_DIR="<REPO_ROOT>/eval/qwen/Overlap_psoups"

cd "<REPO_ROOT>"/code
for p in expert informative style fluency; do
  echo "[$(date '+%H:%M:%S')] Scoring Overlap_psoups with principle ${p}"
  $PYTHON $SCORER \
      --model $JUDGE \
      --input_path "$IN_DIR" \
      --out_path "$OUT_BASE" \
      --principle "$p" \
      --batch_size 8 \
      --num_runs 1
done
echo "DONE psoups overlap scoring"

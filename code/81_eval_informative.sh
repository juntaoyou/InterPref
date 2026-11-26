#!/bin/bash
PYTHON="<REPO_ROOT>/.venv/bin/python"
source "<REPO_ROOT>"/.venv/bin/activate

# 用 Qwen2.5-72B-Instruct-AWQ 评估 informativeness
# 需要 2 张 GPU（~36GB 权重 + overhead）

export CUDA_VISIBLE_DEVICES=0,1
export VLLM_ENABLE_V1_MULTIPROCESSING=0

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EVAL_SCRIPT="${SCRIPT_DIR}/80_llm_score_with_ci.py"

INPUT_DIR="<REPO_ROOT>/tmp_eval/informative_eval"
OUTPUT_DIR="<REPO_ROOT>/tmp_eval/informative_eval_results"
MODEL="<MODELS_DIR>/Qwen2.5-72B-Instruct-AWQ"

mkdir -p "${INPUT_DIR}"

# 复制要评估的文件（仅 4.0）
cp "<REPO_ROOT>"/qwen/mmlu_test/answer_mmlu_neuron_mmlu_4.0/Informativeness_low/Informativeness_low_4.0.json "${INPUT_DIR}/"

echo "[INFO] Judge model: ${MODEL}"
echo "[INFO] Input: ${INPUT_DIR}"
echo "[INFO] Principle: informative"
echo "[INFO] GPU: ${CUDA_VISIBLE_DEVICES}"

$PYTHON "${EVAL_SCRIPT}" \
  --model "${MODEL}" \
  --input_path "${INPUT_DIR}" \
  --out_path "${OUTPUT_DIR}" \
  --batch_size 4 \
  --principle informative \
  --num_runs 3

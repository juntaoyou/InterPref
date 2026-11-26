#!/bin/bash
export CUDA_VISIBLE_DEVICES=6
export VLLM_ENABLE_V1_MULTIPROCESSING=0
# MODEL_PATH="<MODELS_DIR>/Qwen3-8B"
# MODEL_PATH="<MODELS_DIR>/gemma-2-9b-it"
MODEL_PATH="<MODELS_DIR>/Llama-3.1-8B-Instruct"
model='llama'
# MODEL_PATH="/home/<user>/models/Llama-3.1-8B-Instruct"
# NEURON_DIR="<REPO_ROOT>/qwen/psoups_for_dataset_compare"
NEURON_DIR="<REPO_ROOT>/${model}/psoups_for_dataset_compare"
NEURON_MULTI_DIR="<REPO_ROOT>"/${model}/psoups_multi_detect


BATCH_SIZE=32
PYTHON_FILE="./53_generation_multi3.py"  # 指定你的 Python 文件名
DATASET="<DATA_ROOT>/psoups_eval"
# DATASET="<DATA_ROOT>/my_alpaca_eval"
PERCENTS=0.5

MODE='partial2'
OUTPUT_DIR="<REPO_ROOT>"/${model}/with_preferences_answer_${MODE}_steer_${PERCENTS}

echo "[INFO] model: ${MODEL_PATH}"
echo "[INFO] neuron path: ${NEURON_DIR}"
echo "[INFO] output path: ${OUTPUT_DIR}"
echo "[INFO] mode: ${MODE}"


# 执行 Python 脚本，并传递参数
python $PYTHON_FILE \
  --model $MODEL_PATH \
  --neuron_dir $NEURON_DIR \
  --neuron_multi_dir $NEURON_MULTI_DIR \
  --output_dir $OUTPUT_DIR \
  --batch_size $BATCH_SIZE \
  --dataset ${DATASET} \
  --percent ${PERCENTS} \
  --mode ${MODE}



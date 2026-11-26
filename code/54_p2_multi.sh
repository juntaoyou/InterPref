#!/bin/bash
export CUDA_VISIBLE_DEVICES=5
export VLLM_ENABLE_V1_MULTIPROCESSING=0
# MODEL_PATH="NAS/<user>/models/Qwen3-8B"
MODEL_PATH="<MODELS_DIR>/Llama-3.1-8B-Instruct"
# MODEL_PATH="NAS/<user>/models/gemma-2-9b-it"
BATCH_SIZE=32

PYTHON_FILE="./54_p2_multi.py"  # 指定你的 Python 文件名
OUTPUT_DIR='../llama/alpaca/multi_P2_psoups'
DATASET="<DATA_ROOT>/psoups_eval"
# DATASET="NAS/<user>/Mydatasets/my_alpaca_eval"
# 执行 Python 脚本，并传递参数


python $PYTHON_FILE \
  --model $MODEL_PATH \
  --output_dir $OUTPUT_DIR \
  --batch_size $BATCH_SIZE \
  --dataset $DATASET

OUTPUT_DIR='../llama/alpaca/multi_P2_alpaca'
DATASET="<DATA_ROOT>/my_alpaca_eval"
python $PYTHON_FILE \
  --model $MODEL_PATH \
  --output_dir $OUTPUT_DIR \
  --batch_size $BATCH_SIZE \
  --dataset $DATASET

MODEL_PATH="<MODELS_DIR>/gemma-2-9b-it"
OUTPUT_DIR='../gemma/alpaca/multi_P2_psoups'
DATASET="<DATA_ROOT>/psoups_eval"
# DATASET="NAS/<user>/Mydatasets/my_alpaca_eval"
# 执行 Python 脚本，并传递参数


python $PYTHON_FILE \
  --model $MODEL_PATH \
  --output_dir $OUTPUT_DIR \
  --batch_size $BATCH_SIZE \
  --dataset $DATASET

OUTPUT_DIR='../gemma/alpaca/multi_P2_alpaca'
DATASET="<DATA_ROOT>/my_alpaca_eval"
python $PYTHON_FILE \
  --model $MODEL_PATH \
  --output_dir $OUTPUT_DIR \
  --batch_size $BATCH_SIZE \
  --dataset $DATASET
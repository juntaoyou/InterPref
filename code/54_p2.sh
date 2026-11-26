#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

MODEL_PATH="<MODELS_DIR>/Qwen3-8B"
#MODEL_PATH="/home/<user>/models/Llama-3.1-8B-Instruct"
#MODEL_PATH="/home/<user>/models/gemma-2-9b-it"
BATCH_SIZE=32
OUTPUT_DIR='./qwen3-8B'
PYTHON_FILE="./54_p2.py"  # 指定你的 Python 文件名
DATASET="<DATA_ROOT>/psoups_eval"
#DATASET="/home/<user>/Mydatasets/my_alpaca_eval"
# 执行 Python 脚本，并传递参数
python $PYTHON_FILE \
  --model $MODEL_PATH \
  --output_dir $OUTPUT_DIR \
  --batch_size $BATCH_SIZE \
  --dataset $DATASET 

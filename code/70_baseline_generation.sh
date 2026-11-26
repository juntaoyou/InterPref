#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export VLLM_ENABLE_V1_MULTIPROCESSING=0
MODEL_PATH="<MODELS_DIR>/Qwen3-8B"
model='qwen'
#MODEL_PATH="<MODELS_DIR>/Llama-3.1-8B-Instruct"
#MODEL_PATH="<MODELS_DIR>"/gemma-2-9b-it
#NEURON_DIR="./neuron_results"
OUTPUT_DIR="../${model}/mmlu_test/answer_results_baseline"
BATCH_SIZE=32
PYTHON_FILE="./70_baseline_generation.py"
DATASET="<DATA_ROOT>/mmlu_test"
#DATASET="/home/<user>/Mydatasets/my_alpaca_eval"
#DATASET="/home/<user>/Mydatasets/PSoups_queries"
# 执行 Python 脚本，并传递参数
python $PYTHON_FILE \
  --model $MODEL_PATH \
  --output_dir $OUTPUT_DIR \
  --batch_size $BATCH_SIZE \
  --dataset $DATASET

# MODEL_PATH="<MODELS_DIR>/Qwen3-8B"
# model='qwen'
MODEL_PATH="<MODELS_DIR>/Llama-3.1-8B-Instruct"
model="llama"
#MODEL_PATH="<MODELS_DIR>"/gemma-2-9b-it
#NEURON_DIR="./neuron_results"
OUTPUT_DIR="../${model}/mmlu_test/answer_results_baseline"
BATCH_SIZE=32
PYTHON_FILE="./70_baseline_generation.py"
DATASET="<DATA_ROOT>/mmlu_test"
#DATASET="/home/<user>/Mydatasets/my_alpaca_eval"
#DATASET="/home/<user>/Mydatasets/PSoups_queries"
# 执行 Python 脚本，并传递参数
python $PYTHON_FILE \
  --model $MODEL_PATH \
  --output_dir $OUTPUT_DIR \
  --batch_size $BATCH_SIZE \
  --dataset $DATASET

# MODEL_PATH="<MODELS_DIR>/Qwen3-8B"
# model='qwen'
#MODEL_PATH="<MODELS_DIR>/Llama-3.1-8B-Instruct"
MODEL_PATH="<MODELS_DIR>"/gemma-2-9b-it
model="gemma"
#NEURON_DIR="./neuron_results"
OUTPUT_DIR="../${model}/mmlu_test/answer_results_baseline"
BATCH_SIZE=32
PYTHON_FILE="./70_baseline_generation.py"
DATASET="<DATA_ROOT>/mmlu_test"
#DATASET="/home/<user>/Mydatasets/my_alpaca_eval"
#DATASET="/home/<user>/Mydatasets/PSoups_queries"
# 执行 Python 脚本，并传递参数
python $PYTHON_FILE \
  --model $MODEL_PATH \
  --output_dir $OUTPUT_DIR \
  --batch_size $BATCH_SIZE \
  --dataset $DATASET

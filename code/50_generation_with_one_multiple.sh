#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
export VLLM_ENABLE_V1_MULTIPROCESSING=0
MODEL_PATH="<MODELS_DIR>/Qwen3-8B"
# MODEL_PATH="<MODELS_DIR>/Llama-3.1-8B-Instruct"
# MODEL_PATH="<MODELS_DIR>/gemma-2-9b-it"
model="qwen"



NEURON_DIR="<REPO_ROOT>/${model}/mmlu_psoups_merge"
# NEURON_DIR="<REPO_ROOT>/${model}/mmlu_for_dataset_compare"
# OUTPUT_DIR="<REPO_ROOT>/${model}/answer_alpaca_neuron_mmlu"
OUTPUT_DIR="<REPO_ROOT>/${model}/answer_alpaca_neuron_merge"
BATCH_SIZE=32

PYTHON_FILE="./50_generation_with_one.py"  
# DATASET="<DATA_ROOT>/psoups_eval"
DATASET="<DATA_ROOT>/my_alpaca_eval"

python $PYTHON_FILE \
  --model $MODEL_PATH \
  --neuron_dir $NEURON_DIR \
  --output_dir $OUTPUT_DIR \
  --batch_size $BATCH_SIZE \
  --dataset $DATASET \
  --num_generations 5 \
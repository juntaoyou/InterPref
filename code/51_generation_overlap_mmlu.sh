#!/bin/bash
# Generate overlap-steer answers on mmlu_test (570 q).
# Overlap neurons = intersection of psoups- and mmlu-detected neurons
# (results/res/qwen/mmlu_psoups_merge), delta = 4.0.
# Per-preference steering strength (val) matches the psoups overlap run.
PYTHON="<REPO_ROOT>/.venv/bin/python"
source "<REPO_ROOT>"/.venv/bin/activate
export CUDA_VISIBLE_DEVICES=6
export VLLM_ENABLE_V1_MULTIPROCESSING=0

MODEL_PATH="<MODELS_DIR>/Qwen3-8B"
NEURON_DIR="<REPO_ROOT>/results/res/qwen/mmlu_psoups_merge"
OUTPUT_DIR="<REPO_ROOT>/qwen/mmlu_test/answer_mmlu_neuron_overlap"
BATCH_SIZE=32
DATASET="<DATA_ROOT>/mmlu_test"
DELTA=4.0

cd "<REPO_ROOT>"/code
$PYTHON 51_generation_with_one_overlap.py \
  --model $MODEL_PATH \
  --neuron_dir $NEURON_DIR \
  --output_dir $OUTPUT_DIR \
  --batch_size $BATCH_SIZE \
  --dataset $DATASET \
  --delta $DELTA

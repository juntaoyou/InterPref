#!/bin/bash
# Generate 'sdiff' steered answers: change = (single(p0)-multi) U (single(p1)-multi).
set -u
cd "<REPO_ROOT>"/code
source "<REPO_ROOT>"/.venv/bin/activate
export CUDA_VISIBLE_DEVICES=${GPU_ID:-2}
export VLLM_ENABLE_V1_MULTIPROCESSING=0
MODEL_PATH="<MODELS_DIR>/Qwen3-8B"
MODEL="qwen"
NEURON_DIR="<REPO_ROOT>"/res/${MODEL}/psoups_for_dataset_compare
NEURON_MULTI_DIR="<REPO_ROOT>"/res/${MODEL}/psoups_multi_detect
BATCH_SIZE=32
PYTHON_FILE="./60_with_preferences_generation_multi_multi_single.py"
DATASET="<DATA_ROOT>/psoups_eval"
CONFIG_FILE="<REPO_ROOT>/eval/qwen/configs/4failed_config.json"
OUTPUT_DIR="<REPO_ROOT>"/res/${MODEL}/psoups/answer_with_preferences_sdiff_steer
echo "[INFO] mode: sdiff | output: $OUTPUT_DIR"
python $PYTHON_FILE \
    --model $MODEL_PATH --neuron_dir $NEURON_DIR --neuron_multi_dir $NEURON_MULTI_DIR \
    --output_dir $OUTPUT_DIR --batch_size $BATCH_SIZE --dataset ${DATASET} \
    --mode sdiff --percent 2.0 --config_file ${CONFIG_FILE}
echo "[INFO] sdiff generation done, exit: $?"

#!/bin/bash
# Generate 'dual' steered answers: change = (N(p0)-N(p1)) U (N(p1)-N(p0)),
# deactivate = (N(rev0)-N(rev1)) U (N(rev1)-N(rev0))  -> high higher + low lower.
set -u
cd "<REPO_ROOT>"/code
source "<REPO_ROOT>"/.venv/bin/activate

export CUDA_VISIBLE_DEVICES=${GPU_ID:-2}
export VLLM_ENABLE_V1_MULTIPROCESSING=0

MODEL_PATH="<MODELS_DIR>/Qwen3-8B"
MODEL="qwen"
NEURON_DIR="<REPO_ROOT>"/res/${MODEL}/psoups_for_dataset_compare
NEURON_MULTI_DIR="<REPO_ROOT>"/res/${MODEL}/psoups_multi_detect
CONFIG_DIR="<REPO_ROOT>"/eval/${MODEL}/configs

BATCH_SIZE=32
PYTHON_FILE="./60_with_preferences_generation_multi_multi_single.py"
DATASET="<DATA_ROOT>/psoups_eval"
MODE="dual"
CONFIG_FILE="${CONFIG_DIR}/with_preferences_steer_partial1_config_lowflu.json"
OUTPUT_DIR="<REPO_ROOT>"/res/${MODEL}/psoups/answer_with_preferences_${MODE}_lowflu_steer

echo "[INFO] mode: ${MODE} | config: ${CONFIG_FILE}"
echo "[INFO] output: ${OUTPUT_DIR}"
python $PYTHON_FILE \
    --model $MODEL_PATH \
    --neuron_dir $NEURON_DIR \
    --neuron_multi_dir $NEURON_MULTI_DIR \
    --output_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE \
    --dataset ${DATASET} \
    --mode ${MODE} \
    --percent 2.0 \
    --config_file ${CONFIG_FILE}
echo "[INFO] dual generation done, exit: $?"

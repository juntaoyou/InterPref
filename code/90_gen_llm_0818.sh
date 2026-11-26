#!/bin/bash
# Generate partial1/partial2 steered answers for a given model (0818 pipeline).
# Usage: 90_gen_llm_0818.sh <model_name> <model_path> <config_dir>
set -u
MODEL_NAME=$1
MODEL_PATH=$2
CONFIG_DIR=$3
cd "<REPO_ROOT>"/code
source "<REPO_ROOT>"/.venv/bin/activate

export CUDA_VISIBLE_DEVICES=3
export VLLM_ENABLE_V1_MULTIPROCESSING=0

NEURON_DIR="<REPO_ROOT>"/res/${MODEL_NAME}/psoups_for_dataset_compare
NEURON_MULTI_DIR="<REPO_ROOT>"/res/${MODEL_NAME}/psoups_multi_detect

BATCH_SIZE=32
PYTHON_FILE="./60_with_preferences_generation_multi.py"
DATASET="<DATA_ROOT>/psoups_eval"
MODES=("partial1" "partial2")

echo "[INFO] model: ${MODEL_PATH} | name: ${MODEL_NAME}"
for MODE in "${MODES[@]}"; do
    CONFIG_FILE="${CONFIG_DIR}/with_preferences_steer_${MODE}_config.json"
    OUTPUT_DIR="<REPO_ROOT>"/res/${MODEL_NAME}/psoups/answer_with_preferences_${MODE}_steer
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
        --config_file ${CONFIG_FILE} || { echo "[ERROR] generation failed for ${MODEL_NAME} mode ${MODE}"; exit 1; }
done
echo "[INFO] generation done for ${MODEL_NAME}"

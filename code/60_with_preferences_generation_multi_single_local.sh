#!/bin/bash
# Generate partialA1/A2/B1/B2 steered answers (multi-preference minus single-preference neurons).
# Scheme A: partial1=multi-single(pref1), partial2=multi-single(pref2)
# Scheme B: partial1=multi-single(pref2), partial2=multi-single(pref1)
set -u
cd "<REPO_ROOT>"/code
source "<REPO_ROOT>"/.venv/bin/activate

export CUDA_VISIBLE_DEVICES=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0

MODEL_PATH="<MODELS_DIR>/Qwen3-8B"
MODEL="qwen"

NEURON_DIR="<REPO_ROOT>"/res/${MODEL}/psoups_for_dataset_compare
NEURON_MULTI_DIR="<REPO_ROOT>"/res/${MODEL}/psoups_multi_detect
CONFIG_DIR="<REPO_ROOT>"/eval/${MODEL}/configs

BATCH_SIZE=${BATCH_SIZE:-32}
PYTHON_FILE="./60_with_preferences_generation_multi_multi_single.py"
DATASET="<DATA_ROOT>/psoups_eval"
MODES=("partialA1" "partialA2")

for MODE in "${MODES[@]}"; do
    SUFFIX="${MODE##*[A-Za-z]}"  # trailing 1 or 2 -> config suffix
    CONFIG_FILE="${CONFIG_DIR}/with_preferences_steer_partial${SUFFIX}_config.json"
    OUTPUT_DIR="<REPO_ROOT>"/res/${MODEL}/psoups/answer_with_preferences_${MODE}_steer
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
        --config_file ${CONFIG_FILE} || { echo "[ERROR] generation failed for mode ${MODE}"; exit 1; }
done
echo "[INFO] all modes done"

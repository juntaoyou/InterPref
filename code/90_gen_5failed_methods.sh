#!/bin/bash
# Try different neuron steering methods on the 5 never-succeeded preference pairs.
# Variants: union, both, dual@4.0, deact_only@2.0, deact_only@4.0
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
CONFIG_FILE="<REPO_ROOT>/eval/qwen/configs/with_preferences_steer_5failed_config.json"

# method|mode|percent|extra_args|output_suffix
METHODS=(
    "union|union|2.0||union"
    "both|both|2.0||both"
    "dual@4|dual|4.0||dual_4"
    "deact_only@2|dual|2.0|--no_steer|deact_only"
    "deact_only@4|dual|4.0|--no_steer|deact_only_4"
)

for entry in "${METHODS[@]}"; do
    IFS='|' read -r label mode percent extra suffix <<< "$entry"
    OUTPUT_DIR="<REPO_ROOT>"/res/${MODEL}/psoups/answer_with_preferences_${suffix}_steer
    echo "[INFO] === method: $label | mode=$mode percent=$percent ==="
    echo "[INFO] output: $OUTPUT_DIR"
    python $PYTHON_FILE \
        --model $MODEL_PATH \
        --neuron_dir $NEURON_DIR \
        --neuron_multi_dir $NEURON_MULTI_DIR \
        --output_dir $OUTPUT_DIR \
        --batch_size $BATCH_SIZE \
        --dataset ${DATASET} \
        --mode ${mode} \
        --percent ${percent} \
        --config_file ${CONFIG_FILE} $extra || { echo "[ERROR] $label failed"; exit 1; }
    echo "[INFO] === done: $label ==="
done
echo "[INFO] all methods done"

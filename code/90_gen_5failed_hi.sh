#!/bin/bash
# High-ratio targeted experiments for the 4 failing pairs.
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
CFG="<REPO_ROOT>"/eval/qwen/configs

RUNS=(
    "partial2|${CFG}/5failed_hi_config1.json|hi1"
    "partial1|${CFG}/5failed_hi_config2.json|hi2"
    "mint|${CFG}/5failed_hi_config3.json|hi3"
)
for entry in "${RUNS[@]}"; do
    IFS='|' read -r mode cfg suffix <<< "$entry"
    OUTPUT_DIR="<REPO_ROOT>"/res/${MODEL}/psoups/answer_with_preferences_${suffix}_steer
    echo "[INFO] === run: $suffix | mode=$mode ==="
    python $PYTHON_FILE \
        --model $MODEL_PATH --neuron_dir $NEURON_DIR --neuron_multi_dir $NEURON_MULTI_DIR \
        --output_dir $OUTPUT_DIR --batch_size $BATCH_SIZE --dataset ${DATASET} \
        --mode ${mode} --percent 2.0 --config_file ${cfg} || { echo "[ERROR] $suffix failed"; exit 1; }
    echo "[INFO] === done: $suffix ==="
done
echo "[INFO] all high-ratio runs done"

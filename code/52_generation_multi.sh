#!/bin/bash
PYTHON="<REPO_ROOT>/.venv/bin/python"
source "<REPO_ROOT>"/.venv/bin/activate

export CUDA_VISIBLE_DEVICES=5,7
export NCCL_P2P_DISABLE=0
export VLLM_ENABLE_V1_MULTIPROCESSING=0

MODEL_DIR="<MODELS_DIR>"
FILE_DIR="<REPO_ROOT>"/results/res
DATA_DIR="<DATA_ROOT>"

MODEL_PATH="${MODEL_DIR}/gemma-2-9b-it"
model='gemma'

NEURON_DIR=${FILE_DIR}/${model}/mmlu_for_dataset_compare
NEURON_MULTI_DIR=${FILE_DIR}/${model}/mmlu_multi_detect
CONFIG_DIR=${FILE_DIR}/${model}/configs
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_FILE="${SCRIPT_DIR}/52_generation_multi.py"
DATASET="${DATA_DIR}/mmlu_test"

BATCH_SIZE=32
MODE='both'

echo "[INFO] model: ${MODEL_PATH}"
echo "[INFO] neuron path: ${NEURON_DIR}"
echo "[INFO] neuron_multi path: ${NEURON_MULTI_DIR}"
echo "[INFO] dataset: ${DATASET}"
echo "[INFO] mode: ${MODE}"

mkdir -p ${CONFIG_DIR}

# Create symlinks for dict neuron files to expected names
for pct in 1.0 2.0; do
    for pref in "Expertise_low.Style_high" "Expertise_high.Style_low"; do
        dict_file="${NEURON_MULTI_DIR}/${pref}_dict_${pct}.json"
        expected_file="${NEURON_MULTI_DIR}/${pref}_${pct}.json"
        if [ -f "$dict_file" ] && [ ! -f "$expected_file" ]; then
            ln -s "$dict_file" "$expected_file"
            echo "[INFO] Created symlink: ${expected_file} → ${dict_file}"
        fi
    done
done

# Generate with percent=1.0
PERCENT="1.0"
PREFERENCE="Expertise_low.Style_high"

echo "[INFO] === Generating with percent=${PERCENT}, preference=${PREFERENCE} ==="

CONFIG_FILE="${CONFIG_DIR}/deactivate_both_${PREFERENCE}_${PERCENT}.json"
OUTPUT_DIR="${FILE_DIR}/${model}/answer_results_multi_deactivate_both_preference_neurons_${PERCENT}"

cat > ${CONFIG_FILE} << EOF
{
    "${PREFERENCE}": [20, 22],
    "percent": ${PERCENT}
}
EOF

echo "[INFO] config file: ${CONFIG_FILE}"
echo "[INFO] output path: ${OUTPUT_DIR}"

$PYTHON $PYTHON_FILE \
    --model $MODEL_PATH \
    --neuron_dir $NEURON_DIR \
    --neuron_multi_dir $NEURON_MULTI_DIR \
    --output_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE \
    --dataset ${DATASET} \
    --mode ${MODE} \
    --config_file ${CONFIG_FILE} \
    --percent ${PERCENT}

echo "[INFO] === percent=${PERCENT} done ==="

# Generate with percent=2.0
PERCENT="2.0"

echo "[INFO] === Generating with percent=${PERCENT}, preference=${PREFERENCE} ==="

CONFIG_FILE="${CONFIG_DIR}/deactivate_both_${PREFERENCE}_${PERCENT}.json"
OUTPUT_DIR="${FILE_DIR}/${model}/answer_results_multi_deactivate_both_preference_neurons_${PERCENT}"

cat > ${CONFIG_FILE} << EOF
{
    "${PREFERENCE}": [20, 22],
    "percent": ${PERCENT}
}
EOF

echo "[INFO] config file: ${CONFIG_FILE}"
echo "[INFO] output path: ${OUTPUT_DIR}"

$PYTHON $PYTHON_FILE \
    --model $MODEL_PATH \
    --neuron_dir $NEURON_DIR \
    --neuron_multi_dir $NEURON_MULTI_DIR \
    --output_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE \
    --dataset ${DATASET} \
    --mode ${MODE} \
    --config_file ${CONFIG_FILE} \
    --percent ${PERCENT}

echo "[INFO] === percent=${PERCENT} done ==="

echo "[INFO] All done!"

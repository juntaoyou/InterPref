#!/bin/bash
export CUDA_VISIBLE_DEVICES=4
export VLLM_ENABLE_V1_MULTIPROCESSING=0

PYTHON="<REPO_ROOT>/.venv/bin/python"
PYTHON_FILE="<REPO_ROOT>/code/41_deactivate_multi.py"
MODEL_PATH="<MODELS_DIR>/gemma-2-9b-it"
model="gemma"

DATASET="<DATA_ROOT>/psoups_eval"
NEURON_DIR="<REPO_ROOT>/results/res/${model}/psoups_for_dataset_compare"
NEURON_DIR_MULTI="<REPO_ROOT>/results/res/${model}/psoups_multi_detect"
OUTPUT_DIR="<REPO_ROOT>/results/res/psoups_eval/${model}/deactivate_both"

BATCH_SIZE=32
MODE='both'

echo "[INFO] model: ${MODEL_PATH}"
echo "[INFO] dataset: ${DATASET}"
echo "[INFO] mode: ${MODE}"
echo "[INFO] output: ${OUTPUT_DIR}"

mkdir -p "${OUTPUT_DIR}"

for percent in 4.0; do
    PREFERENCE="Informativeness_high.Style_high"
    CONFIG_FILE="${OUTPUT_DIR}/config_${PREFERENCE}_${percent}.json"

    cat > ${CONFIG_FILE} << EOF
{
    "${PREFERENCE}": ${percent}
}
EOF

    echo ""
    echo "[INFO] === percent=${percent} ==="
    echo "[INFO] config: ${CONFIG_FILE}"

    $PYTHON $PYTHON_FILE \
        --model $MODEL_PATH \
        --neuron_dir $NEURON_DIR \
        --neuron_dir_multi $NEURON_DIR_MULTI \
        --output_dir $OUTPUT_DIR \
        --batch_size $BATCH_SIZE \
        --dataset ${DATASET} \
        --mode ${MODE} \
        --config ${CONFIG_FILE}

    echo "[INFO] === percent=${percent} done ==="
done

echo ""
echo "[INFO] All done!"

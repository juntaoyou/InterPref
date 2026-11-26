#!/bin/bash
# Swapped Preference Order Experiment (MMLU + PSoups)
# Tests whether the order of preferences in the prompt affects expression.
set -e

source "<REPO_ROOT>"/.venv/bin/activate
cd "<REPO_ROOT>"/code

export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_V1=0

BATCH_SIZE=32
PYTHON_FILE="./72_baseline_generation_multi_swapped.py"

DATASETS=(
    "<DATA_ROOT>/mmlu_test"
    "<DATA_ROOT>/psoups_eval"
)
DATASET_NAMES=("mmlu_test" "psoups_eval")

run_model() {
    local MODEL_PATH=$1
    local model=$2
    local gpu=$3

    for i in "${!DATASETS[@]}"; do
        ds_path="${DATASETS[$i]}"
        ds_name="${DATASET_NAMES[$i]}"
        OUTPUT_DIR="../${model}/${ds_name}/answer_results_multi_swapped"
        mkdir -p "$OUTPUT_DIR"

        echo "=========================================="
        echo "${model} — ${ds_name} — Swapped Preference Order"
        echo "=========================================="
        CUDA_VISIBLE_DEVICES=$gpu python $PYTHON_FILE \
            --model $MODEL_PATH \
            --output_dir $OUTPUT_DIR \
            --batch_size $BATCH_SIZE \
            --dataset ${ds_path}
    done
}

# 三个模型依次在 GPU 2 上运行
run_model "<MODELS_DIR>/Qwen3-8B"               "qwen"   2
run_model "<MODELS_DIR>/Llama-3.1-8B-Instruct"  "llama"  2
run_model "<MODELS_DIR>/gemma-2-9b-it"          "gemma"  2

echo ""
echo "=========================================="
echo "全部模型 × 全部数据集 swapped 实验完成"
echo "=========================================="

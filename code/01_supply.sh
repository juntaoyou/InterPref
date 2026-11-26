#!/bin/bash
# 补充缺失的数据文件（本地路径版）
# 按本地路径组织数据补齐流程

source "<REPO_ROOT>"/.venv/bin/activate
cd "<REPO_ROOT>"/code

export CUDA_VISIBLE_DEVICES=4
export NCCL_P2P_DISABLE=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
MODEL_DIR="<MODELS_DIR>"
BASE_DIR="<REPO_ROOT>"
DATA_DIR="<DATA_ROOT>"
CONFIG_DIR="<REPO_ROOT>"/detect_with_two/eval
# 定义模型路径
declare -A MODEL_PATHS
MODEL_PATHS=(
    ["qwen"]="${MODEL_DIR}/Qwen3-8B"
    ["llama"]="${MODEL_DIR}/Llama-3.1-8B-Instruct"
    ["gemma"]="${MODEL_DIR}/gemma-2-9b-it"
)

# 偏好对列表
preferences=("Expertise.Informativeness" "Expertise.Style" "Informativeness.Style")
Batch_size=32

echo "=========================================="
echo "开始补充缺失的数据文件（本地路径版）"
echo "=========================================="

# 函数：运行 MultiSteer Both/Union 生成实验
run_experiment() {
    local model=$1
    local mode=$2
    local preference=$3
    local extra_config=$4

    local MODEL_PATH=${MODEL_PATHS[$model]}
    local NEURON_DIR=${BASE_DIR}/${model}/psoups_for_dataset_compare
    local NEURON_MULTI_DIR=${BASE_DIR}/${model}/psoups_multi_detect
    local CONFIG_FILE="${CONFIG_DIR}/${model}/configs/steer_${mode}_config.json"
    local OUTPUT_DIR="${BASE_DIR}/${model}/answer_${mode}_steer"
    local DATASET="${DATA_DIR}/mmlu_test"

    echo ""
    echo "----------------------------------------"
    echo "运行实验: ${model} / ${mode} (52_generation_multi.py)"
    echo "模型路径: ${MODEL_PATH}"
    echo "输出: ${OUTPUT_DIR}"
    echo "----------------------------------------"

    if [ -n "$extra_config" ]; then
        echo "  使用自定义 config: ${extra_config}"
        python 52_generation_multi.py \
            --model "$MODEL_PATH" \
            --neuron_dir "$NEURON_DIR" \
            --neuron_multi_dir "$NEURON_MULTI_DIR" \
            --output_dir "$OUTPUT_DIR" \
            --batch_size $Batch_size \
            --dataset "$DATASET" \
            --mode "$mode" \
            --config "$extra_config"
    else
        if [ ! -f "$CONFIG_FILE" ]; then
            echo "⚠️  警告: 配置文件不存在: ${CONFIG_FILE}"
            return 1
        fi
        echo "  配置: ${CONFIG_FILE}"
        python 52_generation_multi.py \
            --model "$MODEL_PATH" \
            --neuron_dir "$NEURON_DIR" \
            --neuron_multi_dir "$NEURON_MULTI_DIR" \
            --output_dir "$OUTPUT_DIR" \
            --batch_size $Batch_size \
            --dataset "$DATASET" \
            --mode "$mode" \
            --config_file "$CONFIG_FILE"
    fi

    echo "✓ 完成: ${model} / ${mode}"
}

# 函数：运行 Partial_steer 生成实验
run_partial_experiment() {
    local model=$1
    local mode=$2
    local preference=$3

    local MODEL_PATH=${MODEL_PATHS[$model]}
    local NEURON_DIR=${BASE_DIR}/${model}/psoups_for_dataset_compare
    local NEURON_MULTI_DIR=${BASE_DIR}/${model}/psoups_multi_detect
    local CONFIG_FILE="${CONFIG_DIR}/${model}/configs/with_preferences_steer_${mode}_config.json"
    local OUTPUT_DIR="${BASE_DIR}/mmlu_test/${model}/answer_with_preferences_${mode}_steer"
    local DATASET="${DATA_DIR}/mmlu_test"

    echo ""
    echo "----------------------------------------"
    echo "运行实验: ${model} / ${mode} (60_with_preferences_generation_multi.py)"
    echo "模型路径: ${MODEL_PATH}"
    echo "配置: ${CONFIG_FILE}"
    echo "输出: ${OUTPUT_DIR}"
    echo "----------------------------------------"

    if [ ! -f "$CONFIG_FILE" ]; then
        echo "⚠️  警告: 配置文件不存在: ${CONFIG_FILE}"
        return 1
    fi

    python 60_with_preferences_generation_multi.py \
        --model "$MODEL_PATH" \
        --neuron_dir "$NEURON_DIR" \
        --neuron_multi_dir "$NEURON_MULTI_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --batch_size $Batch_size \
        --dataset "$DATASET" \
        --mode "$mode" \
        --percent 6.0 \
        --config_file "$CONFIG_FILE"

    echo "✓ 完成: ${model} / ${mode}"
}

# 函数：运行 Deactivate 生成实验
run_deactivate_experiment() {
    local model=$1
    local mode=$2
    local preference=$3

    local MODEL_PATH=${MODEL_PATHS[$model]}
    local NEURON_DIR=${BASE_DIR}/${model}/psoups_for_dataset_compare
    local NEURON_MULTI_DIR=${BASE_DIR}/${model}/psoups_multi_detect
    local DEACTIVATE="preference_neurons"
    local OUTPUT_DIR="${BASE_DIR}/mmlu_test/${model}/answer_results_multi_deactivate_${mode}_${DEACTIVATE}_6.0"
    local DATASET="${DATA_DIR}/mmlu_test"
    local CONFIG_FILE="${CONFIG_DIR}/${model}/configs/deactivate_${mode}_config.json"

    echo ""
    echo "----------------------------------------"
    echo "运行实验: ${model} / deactivate_${mode} (41_deactivate_multi.py)"
    echo "模型路径: ${MODEL_PATH}"
    echo "配置: ${CONFIG_FILE}"
    echo "输出: ${OUTPUT_DIR}"
    echo "----------------------------------------"


    python 41_deactivate_multi.py \
        --model "$MODEL_PATH" \
        --neuron_dir "$NEURON_DIR" \
        --neuron_dir_multi "$NEURON_MULTI_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --batch_size $Batch_size \
        --deactivate "$DEACTIVATE" \
        --mode "$mode" \
        --dataset "$DATASET" \
        --config_file "$CONFIG_FILE"

    echo "✓ 完成: ${model} / deactivate_${mode}"
}

# # ==========================================
# # 第一部分：Deactivate 数据补充
# # ==========================================
# echo ""
# echo "=========================================="
# echo "第一部分：补充 Deactivate 数据"
# echo "=========================================="

# # Qwen deactivate — both + union
# echo ""
# echo "--- Qwen deactivate ---"
# run_deactivate_experiment "qwen" "both" ""
# run_deactivate_experiment "qwen" "union" ""

# # Llama deactivate — both + union
# echo ""
# echo "--- Llama deactivate ---"
# run_deactivate_experiment "llama" "both" ""
# run_deactivate_experiment "llama" "union" ""

# # Gemma deactivate — both + union
# echo ""
# echo "--- Gemma deactivate ---"
# run_deactivate_experiment "gemma" "both" ""
# run_deactivate_experiment "gemma" "union" ""

# # ==========================================
# # 第二部分：MultiSteer 数据补充
# # ==========================================
# echo ""
# echo "=========================================="
# echo "第二部分：补充 MultiSteer 数据"
# echo "=========================================="

# # Llama/Both — 仅 Informativeness.Style 缺 4 个（每 combo 缺 1 个系数值）
# echo ""
# echo "--- Llama MultiSteer/Both (仅 I.S.) ---"
# run_experiment "llama" "both" "Informativeness.Style" \
#     '{"percent": 6.0, "Informativeness_low.Style_low": [19, 21], "Informativeness_low.Style_high": [20, 22], "Informativeness_high.Style_low": [25, 27], "Informativeness_high.Style_high": [10, 12]}'

# # Gemma/Both — 每分组缺 4 个，共 12 个
# echo ""
# echo "--- Gemma MultiSteer/Both ---"
# run_experiment "gemma" "both" ""

# # Gemma/Union — 每分组缺 4 个，共 12 个
# echo ""
# echo "--- Gemma MultiSteer/Union ---"
# run_experiment "gemma" "union" ""

# ==========================================
# 第三部分：Partial_steer 数据补充
# ==========================================
echo ""
echo "=========================================="
echo "第三部分：补充 Partial_steer 数据"
echo "=========================================="

# Llama partial — partial1 + partial2
echo ""
echo "--- Llama partial ---"
run_partial_experiment "llama" "partial1" ""
run_partial_experiment "llama" "partial2" ""

# Gemma partial — partial1 + partial2
echo ""
echo "--- Gemma partial ---"
run_partial_experiment "gemma" "partial1" ""
run_partial_experiment "gemma" "partial2" ""

echo ""
echo "=========================================="
echo "所有数据补充完成！"
echo "=========================================="
echo ""
echo "生成的数据位置："
echo "  Deactivate:  ${BASE_DIR}/{model}/answer_results_multi_deactivate_{both,union}_preference_neurons_6.0/"
echo "  MultiSteer:  ${BASE_DIR}/{model}/answer_{both,union}_steer/"
echo "  Partial:     ${BASE_DIR}/{model}/answer_with_preferences_partial{1,2}_steer/"
echo ""

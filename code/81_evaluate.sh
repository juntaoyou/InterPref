#!/bin/bash
# ================================================================
# 评估脚本: 评估 deactivate_both / deactivate_union 的生成结果
# 使用 80_llm_score_with_ci.py 进行多次评估，计算置信区间与 3σ 误差
#
# 目录结构:
#   .../deactivate_both_preference_neurons_6.0/Expertise.Informativeness/
#   .../deactivate_both_preference_neurons_6.0/Expertise.Style/
#   .../deactivate_both_preference_neurons_6.0/Informativeness.Style/
#   .../deactivate_union_preference_neurons_6.0/相同结构/
#
# 根据子目录名自动匹配评估维度:
#   Expertise.Informativeness → expert + informative
#   Expertise.Style           → expert + style
#   Informativeness.Style     → informative + style
#   所有文件                   → fluency
# ================================================================
source "<REPO_ROOT>"/.venv/bin/activate
cd "<REPO_ROOT>"/code

export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5}"

MODEL_DIR="<MODELS_DIR>"
FILE_DIR="<PROJECT_ROOT>"/eval

# 评判模型
JUDGE_MODEL="${MODEL_DIR}/Qwen2.5-72B-Instruct-AWQ"

# 评估配置
NUM_RUNS=3
BATCH_SIZE=4

# 待评估的目录对
declare -A DIR_PAIRS=(
    ["deactivate_both"]  "${FILE_DIR}/llama/answer_results_multi_deactivate_both_preference_neurons_6.0"
    ["deactivate_union"] "${FILE_DIR}/llama/answer_results_multi_deactivate_union_preference_neurons_6.0"
)

# 子目录 → principle 映射
declare -A SUBDIR_PRINCIPLES=(
    ["Expertise.Informativeness"]  "expert,informative"
    ["Expertise.Style"]            "expert,style"
    ["Informativeness.Style"]      "informative,style"
)

for TAG in "${!DIR_PAIRS[@]}"; do
    INPUT_DIR="${DIR_PAIRS[$TAG]}"
    OUTPUT_BASE="${FILE_DIR}/llama/eval_${TAG}_neurons_6.0"

    echo ""
    echo "================================================"
    echo " Tag:      ${TAG}"
    echo " Input:    ${INPUT_DIR}"
    echo " Output:   ${OUTPUT_BASE}"
    echo " Judge:    ${JUDGE_MODEL}"
    echo " Runs:     ${NUM_RUNS}"
    echo "================================================"

    if [ ! -d "$INPUT_DIR" ]; then
        echo "[WARNING] Input directory does not exist: ${INPUT_DIR}"
        continue
    fi

    # =========================================================
    #  1. Fluency → 评估所有文件（递归收集，平铺到临时目录）
    # =========================================================
    echo ""
    echo "  >>> [fluency] 构建临时目录 (所有文件) ..."
    TMP_FLUENCY="/tmp/eval_${TAG}_fluency"
    rm -rf "$TMP_FLUENCY"
    mkdir -p "$TMP_FLUENCY"
    find "$INPUT_DIR" -name '*.json' -type f -exec ln -sf {} "$TMP_FLUENCY/" \;

    if [ -n "$(ls -A "$TMP_FLUENCY")" ]; then
        echo "  >>> [fluency] 评估所有文件 (共 $(ls "$TMP_FLUENCY" | wc -l) 个) ..."
        python 80_llm_score_with_ci.py \
            --model "${JUDGE_MODEL}" \
            --input_path "${TMP_FLUENCY}" \
            --out_path "${OUTPUT_BASE}" \
            --batch_size ${BATCH_SIZE} \
            --principle "fluency" \
            --num_runs ${NUM_RUNS}
    else
        echo "  >>> [fluency] 无文件, 跳过"
    fi
    rm -rf "$TMP_FLUENCY"

    # =========================================================
    #  2. 按子目录名匹配合适的维度
    # =========================================================
    for SUBDIR in "${!SUBDIR_PRINCIPLES[@]}"; do
        PRINCIPLES="${SUBDIR_PRINCIPLES[$SUBDIR]}"
        SUBDIR_PATH="${INPUT_DIR}/${SUBDIR}"

        if [ ! -d "$SUBDIR_PATH" ]; then
            echo "  >>> [${SUBDIR}] 目录不存在, 跳过"
            continue
        fi

        # 将子目录中的文件平铺到临时目录
        TMP_DIR="/tmp/eval_${TAG}_${SUBDIR}"
        rm -rf "$TMP_DIR"
        mkdir -p "$TMP_DIR"
        find "$SUBDIR_PATH" -name '*.json' -type f -exec ln -sf {} "$TMP_DIR/" \;

        FILE_COUNT=$(ls "$TMP_DIR" 2>/dev/null | wc -l)
        if [ "$FILE_COUNT" -eq 0 ]; then
            echo "  >>> [${SUBDIR}] 无 JSON 文件, 跳过"
            rm -rf "$TMP_DIR"
            continue
        fi

        # 对每个 principle 分别评估
        IFS=',' read -ra PRINCE_ARRAY <<< "$PRINCIPLES"
        for PRINCIPLE in "${PRINCE_ARRAY[@]}"; do
            echo ""
            echo "  >>> [${SUBDIR}] 评估维度: ${PRINCIPLE} (${FILE_COUNT} 个文件) ..."
            python 80_llm_score_with_ci.py \
                --model "${JUDGE_MODEL}" \
                --input_path "${TMP_DIR}" \
                --out_path "${OUTPUT_BASE}" \
                --batch_size ${BATCH_SIZE} \
                --principle "${PRINCIPLE}" \
                --num_runs ${NUM_RUNS}
        done

        rm -rf "$TMP_DIR"
    done

    echo ""
    echo "  All evaluations done for ${TAG}."
    echo "  Results saved to: ${OUTPUT_BASE}"
done

echo ""
echo "================================================"
echo "  All evaluations complete!"
echo "================================================"

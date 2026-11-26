#!/bin/bash
# ================================================================
# 评估脚本: 对 60_with_preferences_generation_multi.sh 生成的输出
# 使用 80_llm_score_with_ci.py 进行多次评估，计算置信区间与 3σ 误差
#
# 根据文件名自动匹配评估维度:
#   文件名含 expertise      → 评估 expert
#   文件名含 informativeness → 评估 informative
#   文件名含 style           → 评估 style
#   所有文件                 → 评估 fluency
# ================================================================
source "<REPO_ROOT>"/.venv/bin/activate
cd "<REPO_ROOT>"/code

export NCCL_P2P_DISABLE=1

MODEL_DIR="<MODELS_DIR>"
ROOT_DIR="<PROJECT_ROOT>"/eval

# 评判模型 (用于打分的 LLM)
JUDGE_MODEL="${MODEL_DIR}/Qwen2.5-72B-Instruct-AWQ"

# 待评估的生成模型 (需与生成脚本一致)
MODEL_PATHS=(
    "${MODEL_DIR}/Qwen3-8B"
    # "${MODEL_DIR}/Llama3.1-8B"
    # "${MODEL_DIR}/gemma-2-9b-it"
)

# 评估配置
NUM_RUNS=3             # 每个文件运行次数
CONFIDENCE_LEVEL=0.95  # 置信水平
SIGMA_MULTIPLIER=3.0   # 3σ 误差
BATCH_SIZE=4           # 评判模型批大小

# 生成模式 (需与生成脚本一致)
MODES=('partial2' 'partial1')

for MODEL_PATH in "${MODEL_PATHS[@]}"; do

    if [[ "$MODEL_PATH" == "${MODEL_DIR}/Qwen3-8B" ]]; then
        model='qwen'
    elif [[ "$MODEL_PATH" == "${MODEL_DIR}/Llama3.1-8B" ]]; then
        model='llama'
    elif [[ "$MODEL_PATH" == "${MODEL_DIR}/gemma-2-9b-it" ]]; then
        model='gemma'
    else
        echo "Unknown model path: $MODEL_PATH"
        exit 1
    fi

    for MODE in "${MODES[@]}"; do
        INPUT_DIR="${ROOT_DIR}/${model}/answer_with_preferences_${MODE}_steer"
        OUTPUT_BASE="${ROOT_DIR}/${model}/eval_ci_with_preferences_${MODE}"

        echo ""
        echo "================================================"
        echo " Model:     ${model}"
        echo " Mode:      ${MODE}"
        echo " Input:     ${INPUT_DIR}"
        echo " Output:    ${OUTPUT_BASE}"
        echo " Judge:     ${JUDGE_MODEL}"
        echo " Runs:      ${NUM_RUNS}"
        echo "================================================"

        if [ ! -d "$INPUT_DIR" ]; then
            echo "[WARNING] Input directory does not exist: ${INPUT_DIR}"
            echo "          Run 60_with_preferences_generation_multi.sh first!"
            continue
        fi

        # =========================================================
        #  1. Fluency → 评估所有文件
        #    (先用 find 递归搜出所有 .json，平铺到临时目录，因为
        #    80_llm_score_with_ci.py 用 os.listdir 不递归)
        # =========================================================
        echo ""
        echo "  >>> [fluency] 构建临时目录 (所有文件) ..."
        TMP_FLUENCY="${INPUT_DIR}_eval_fluency"
        mkdir -p "$TMP_FLUENCY"
        find "$INPUT_DIR" -name '*.json' -type f -exec ln -sf {} "$TMP_FLUENCY/" \;

        echo "  >>> [fluency] 评估所有文件 ..."
        python 80_llm_score_with_ci.py \
            --model "${JUDGE_MODEL}" \
            --input_path "${TMP_FLUENCY}" \
            --out_path "${OUTPUT_BASE}" \
            --batch_size ${BATCH_SIZE} \
            --principle "fluency" \
            --num_runs ${NUM_RUNS} \
            --confidence_level ${CONFIDENCE_LEVEL} \
            --sigma_multiplier ${SIGMA_MULTIPLIER}
        rm -rf "$TMP_FLUENCY"

        # =========================================================
        #  2. 按文件名关键词匹配合适的维度
        # =========================================================
        # keyword -> principle 映射
        declare -A PRINCIPLE_MAP=(
            ["expertise"]="expert"
            ["informativeness"]="informative"
            ["style"]="style"
        )

        for KEYWORD in "${!PRINCIPLE_MAP[@]}"; do
            PRINCIPLE="${PRINCIPLE_MAP[$KEYWORD]}"

            # 递归搜索匹配的文件
            MATCHED=($(find "$INPUT_DIR" -name "*${KEYWORD}*.json" -type f))
            if [ ${#MATCHED[@]} -eq 0 ]; then
                echo "  >>> [${PRINCIPLE}] 无匹配文件 (keyword: ${KEYWORD}), 跳过"
                continue
            fi

            # 创建临时目录，将匹配的文件平铺软链进去
            TMP_DIR="${INPUT_DIR}_eval_${PRINCIPLE}"
            mkdir -p "$TMP_DIR"

            for f in "${MATCHED[@]}"; do
                bn=$(basename "$f")
                ln -sf "$f" "$TMP_DIR/$bn"
            done

            echo ""
            echo "  >>> [${PRINCIPLE}] 评估 ${#MATCHED[@]} 个文件 (keyword: ${KEYWORD}) ..."
            python 80_llm_score_with_ci.py \
                --model "${JUDGE_MODEL}" \
                --input_path "${TMP_DIR}" \
                --out_path "${OUTPUT_BASE}" \
                --batch_size ${BATCH_SIZE} \
                --principle "${PRINCIPLE}" \
                --num_runs ${NUM_RUNS} \
                --confidence_level ${CONFIDENCE_LEVEL} \
                --sigma_multiplier ${SIGMA_MULTIPLIER}

            # 清理临时目录
            rm -rf "$TMP_DIR"
        done

        echo ""
        echo "  All evaluations done for ${model} / ${MODE}."
        echo "  Results saved to: ${OUTPUT_BASE}_*"
    done
done

echo ""
echo "================================================"
echo "  All evaluations complete!"
echo "================================================"

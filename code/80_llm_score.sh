#!/bin/bash
source "<REPO_ROOT>"/.venv/bin/activate
cd "<REPO_ROOT>"/code
PYTHON="<REPO_ROOT>/.venv/bin/python"
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_P2P_DISABLE=1
PYTHON_FILE="80_llm_score_with_ci.py"
# model='qwen' 
MODEL_DIR="<MODELS_DIR>"
FILE_DIR="<REPO_ROOT>"/res
# MODEL_PATH="/home/<user>/preference/model/Qwen3-8B"
# MODEL_PATH="/home/<user>/preference/model/Llama3.1-8B"
#MODEL_PATH="/home/<user>/preference/model/gemma-2-9b-it"

MODEL_PATHS=(
    # "${MODEL_DIR}/Qwen3-8B"
    # "${MODEL_DIR}/Llama3.1-8B"
    "${MODEL_DIR}/gemma-2-9b-it"
)

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


OUTPUT_DIR=./t-test_results_${model}_mmlu/res
Batch_size=4

echo "=========================================="
echo "开始处理模型: ${model}"
echo "=========================================="

# ==========================================
# answer_both_steer 实验
# ==========================================
echo ""
echo "=========================================="
echo "运行 answer_both_steer 实验..."
echo "=========================================="

STEER_DIR="${FILE_DIR}/${model}/answer_both_steer"
if [ -d "$STEER_DIR" ]; then
    for p_dir in "$STEER_DIR"/*/; do
        [ -d "$p_dir" ] || continue
        p=$(basename "$p_dir")

        echo ""
        echo "运行: answer_both_steer / ${p}"
        echo "输入路径: ${p_dir}"

        # 先运行 fluency（基线）
        $PYTHON $PYTHON_FILE \
                --input_path "$p_dir" \
                --principle fluency \
                --batch_size $Batch_size \
                --out_path "${OUTPUT_DIR}"

        # 确定 principle 列表：根据目录名中的维度关键词
        principles=()
        if [[ "$p" == *Expertise* ]]; then
            principles+=("expert")
        fi
        if [[ "$p" == *Informativeness* ]]; then
            principles+=("informative")
        fi
        if [[ "$p" == *Style* ]]; then
            principles+=("style")
        fi

        # 运行各个 principle
        for principle in "${principles[@]}"; do
            echo "  运行 principle: ${principle}"
            $PYTHON $PYTHON_FILE \
                    --input_path "$p_dir" \
                    --principle $principle \
                    --batch_size $Batch_size \
                    --out_path "${OUTPUT_DIR}"
        done

        echo "✓ 完成: answer_both_steer / ${p}"
    done
else
    echo "⚠️  路径不存在: ${STEER_DIR}"
fi

# ==========================================
# answer_union_steer 实验
# ==========================================
echo ""
echo "=========================================="
echo "运行 answer_union_steer 实验..."
echo "=========================================="

STEER_DIR="${FILE_DIR}/${model}/answer_union_steer"
if [ -d "$STEER_DIR" ]; then
    for p_dir in "$STEER_DIR"/*/; do
        [ -d "$p_dir" ] || continue
        p=$(basename "$p_dir")

        echo ""
        echo "运行: answer_union_steer / ${p}"
        echo "输入路径: ${p_dir}"

        # 先运行 fluency（基线）
        $PYTHON $PYTHON_FILE \
                --input_path "$p_dir" \
                --principle fluency \
                --batch_size $Batch_size \
                --out_path "${OUTPUT_DIR}"

        # 确定 principle 列表
        principles=()
        if [[ "$p" == *Expertise* ]]; then
            principles+=("expert")
        fi
        if [[ "$p" == *Informativeness* ]]; then
            principles+=("informative")
        fi
        if [[ "$p" == *Style* ]]; then
            principles+=("style")
        fi

        # 运行各个 principle
        for principle in "${principles[@]}"; do
            echo "  运行 principle: ${principle}"
            $PYTHON $PYTHON_FILE \
                    --input_path "$p_dir" \
                    --principle $principle \
                    --batch_size $Batch_size \
                    --out_path "${OUTPUT_DIR}"
        done

        echo "✓ 完成: answer_union_steer / ${p}"
    done
else
    echo "⚠️  路径不存在: ${STEER_DIR}"
fi

echo ""
echo "=========================================="
echo "模型 ${model} 所有实验完成！"
echo "=========================================="

done

echo ""
echo "=========================================="
echo "所有模型实验全部完成！"
echo "=========================================="

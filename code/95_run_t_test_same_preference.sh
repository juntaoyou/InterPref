#!/bin/bash

PYTHON_FILE="95_t_test_analysis.py"
BASE_DIR="<REPO_ROOT>/code/t-test-files"
OUTPUT_BASE_DIR="<REPO_ROOT>/code/t-test-res-files"
MODELS=("qwen" "llama" "gemma")

echo "=== 开始t检验分析 ==="
echo ""

for model in "${MODELS[@]}"; do
    echo "=========================================="
    echo "处理模型: $model"
    echo "=========================================="
    echo ""

    MODEL_DIR="$BASE_DIR/${model}_mmlu"

    if [ ! -d "$MODEL_DIR" ]; then
        echo "目录 $MODEL_DIR 不存在，跳过"
        echo ""
        continue
    fi

    for subfolder in "$MODEL_DIR"/*; do
        if [ ! -d "$subfolder" ]; then
            continue
        fi

        SUBDIR_NAME=$(basename "$subfolder")
        echo "--- 子文件夹: $SUBDIR_NAME ---"

        OUTPUT_DIR="$OUTPUT_BASE_DIR/${model}_mmlu/$SUBDIR_NAME"
        mkdir -p "$OUTPUT_DIR"

        # 找到该子文件夹中的所有json文件
        files=("$subfolder"/*.json)

        if [ ${#files[@]} -lt 2 ]; then
            echo "  文件数量不足2个，跳过"
            echo ""
            continue
        fi

        FILE1="${files[0]}"
        FILE2="${files[1]}"

        echo "  文件1: $(basename "$FILE1")"
        echo "  文件2: $(basename "$FILE2")"

        python "$PYTHON_FILE" --file1 "$FILE1" --file2 "$FILE2" --output-dir "$OUTPUT_DIR"

        echo ""
    done

    echo ""
done

echo "=== 所有分析完成 ==="
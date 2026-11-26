#!/bin/bash

PYTHON_FILE="95_t_test_analysis.py"
model="qwen"
DATA_DIR="<REPO_ROOT>/code/t-test-files/$model/deactivate_vary/pref"
OUTPUT_DIR="<REPO_ROOT>/code/t-test-res-files/$model/deactivate_vary/pref"

mkdir -p "$OUTPUT_DIR"

echo "=== 开始deactivate_vary偏好文件的t检验分析 ==="
echo ""

declare -A preference_types=(
    ["Expertise_high"]="Expertise_high_prompt.json"
    ["Expertise_low"]="Expertise_low_prompt.json"
    ["Informativeness_high"]="Informativeness_high_prompt.json"
    ["Informativeness_low"]="Informativeness_low_prompt.json"
    ["Style_high"]="Style_high_prompt.json"
    ["Style_low"]="Style_low_prompt.json"
)

ratios=(2.0 4.0 6.0 8.0 10.0 12.0 14.0 16.0 18.0 20.0)

echo "--- 不同比例scores文件与对应prompt文件的比较 ---"
echo ""

for pref_type in "${!preference_types[@]}"; do
    prompt_file="${preference_types[$pref_type]}"
    FILE1="$DATA_DIR/$prompt_file"
    
    if [ ! -f "$FILE1" ]; then
        echo "  错误: Prompt文件 $FILE1 不存在，跳过 $pref_type"
        echo ""
        continue
    fi
    
    echo "----------------------------------------"
    echo "偏好类型: $pref_type"
    echo "  Prompt文件: $prompt_file"
    echo ""
    
    for ratio in "${ratios[@]}"; do
        score_file="${pref_type}_${ratio}.average_1.scores.json"
        FILE2="$DATA_DIR/$score_file"
        
        if [ ! -f "$FILE2" ]; then
            echo "  警告: Scores文件 $score_file 不存在，跳过"
            continue
        fi
        
        echo "  比较比例: $ratio"
        echo "    Prompt vs ${score_file}"
        
        python "$PYTHON_FILE" --file1 "$FILE1" --file2 "$FILE2" --output-dir "$OUTPUT_DIR"
        
        echo ""
    done
done

echo "=== 所有分析完成 ==="
echo "结果已保存至: $OUTPUT_DIR"
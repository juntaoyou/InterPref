#!/bin/bash
# Judge Robustness: 用 Llama-3.1-72B-Instruct 评估 None.json 的 4 个 principle
# 用法: bash 81_evaluate_mmlu.sh

set -e

source "<REPO_ROOT>"/.venv/bin/activate
cd "<REPO_ROOT>"/code

export NCCL_P2P_DISABLE=1

# ---------- 路径变量 ----------
MODEL_DIR="<MODELS_DIR>"
FILE_DIR="<PROJECT_ROOT>"/eval
DATA_DIR="<PROJECT_ROOT>"

# ---------- 配置 ----------
EVAL_BASE=${DATA_DIR}/mmlu_test/single
SCRIPT=80_llm_score_with_ci.py
JUDGE_MODEL=${MODEL_DIR}/Llama-3.1-72B-Instruct
BATCH_SIZE=4
NUM_RUNS=3
OUTPUT_BASE=${FILE_DIR}/eval_results/mmlu_test_robustness
mkdir -p "$OUTPUT_BASE"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5}"

echo "=========================================="
echo "Judge Robustness — Llama-3.1-72B 评估 None.json"
echo "法官模型: ${JUDGE_MODEL}"
echo "评估次数: ${NUM_RUNS}"
echo "批大小:   ${BATCH_SIZE}"
echo "GPU:      ${CUDA_VISIBLE_DEVICES}"
echo "=========================================="

# ---- 只评估 3 个模型的 None.json ----
FILES=()
for model in qwen llama gemma; do
    f="${EVAL_BASE}/${model}/Base/None.json"
    if [ -f "$f" ]; then
        FILES+=("$f")
        echo "  ✓ $f"
    else
        echo "  ⚠ 未找到: $f"
    fi
done

total=${#FILES[@]}
echo ""
echo "共 ${total} 个 None.json 文件"
echo ""

if [ "$total" -eq 0 ]; then
    echo "❌ 未找到任何 None.json，退出。"
    exit 1
fi

# ---- 评估 4 个 principle ----
PRINCIPLES=("expert" "fluency" "informative" "style")

for PRINCIPLE in "${PRINCIPLES[@]}"; do
    echo ""
    echo "========================================"
    echo "评估 ${PRINCIPLE}"
    echo "========================================"

    temp_dir="/tmp/eval_robustness_${PRINCIPLE}"
    mkdir -p "$temp_dir"
    rm -f "$temp_dir"/*.json

    for f in "${FILES[@]}"; do
        model_name=$(basename "$(dirname "$(dirname "$f")")")
        ln -sf "$f" "${temp_dir}/${model_name}_None.json"
    done

    out_dir="${OUTPUT_BASE}/${PRINCIPLE}"
    mkdir -p "$out_dir"

    python "$SCRIPT" \
        --model "$JUDGE_MODEL" \
        --input_path "$temp_dir" \
        --out_path "$out_dir" \
        --principle "$PRINCIPLE" \
        --batch_size "$BATCH_SIZE" \
        --num_runs "$NUM_RUNS"

    rm -rf "$temp_dir"
    echo "✓ ${PRINCIPLE} 评估完成"
done

echo ""
echo "=========================================="
echo "🎉 评估完成！"
echo "结果保存至: ${OUTPUT_BASE}"
echo "=========================================="
echo ""
echo "下一步: 运行 85_compute_judge_robustness.py 计算 Pearson r"

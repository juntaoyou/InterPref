#!/bin/bash
# Generate multi-preference neuron data for mmlu dataset
# 1. 30_search_multi.py: run inference to collect neuron activations → .pt files
# 2. 20_process_multi.py: extract top% neurons → .json files

PYTHON="<REPO_ROOT>/.venv/bin/python"
CODE_DIR="<REPO_ROOT>/code"
BASE_DIR="<REPO_ROOT>/res"
DATASET="<DATA_ROOT>/mmlu_sampled_by_psoups"

# Models: (short_name, model_path, model_key_for_search)
# 30_search_multi.sh originally uses these paths
MODELS=(
    "gemma:<MODELS_DIR>/gemma-2-9b-it"
    "llama:<MODELS_DIR>/Llama-3.1-8B-Instruct"
    "qwen:<MODELS_DIR>/Qwen3-8B"
)

# Percentages for 20_process_multi.py
PERCENTS=(0.01 0.05 0.1 0.5 1.0 1.5 2.0 4.0 6.0 8.0 10.0 12.0 14.0 16.0 18.0 20.0)

for model_entry in "${MODELS[@]}"; do
    model_short="${model_entry%%:*}"
    model_path="${model_entry##*:}"

    ANSWER_DIR="${BASE_DIR}/${model_short}/mmlu_answer_results_multi"
    NEURON_DIR="${BASE_DIR}/${model_short}/neuron_results_multi_on_mmlu"
    OUT_DIR="${BASE_DIR}/${model_short}/mmlu_multi_detect"

    mkdir -p "$ANSWER_DIR" "$NEURON_DIR" "$OUT_DIR"

    echo ""
    echo "============================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Processing model: ${model_short}"
    echo "  Model path: ${model_path}"
    echo "  ANSWER_DIR: ${ANSWER_DIR}"
    echo "  NEURON_DIR: ${NEURON_DIR}"
    echo "  OUT_DIR: ${OUT_DIR}"
    echo "============================================================"

    # Step 1: Run 30_search_multi.py to generate .pt neuron files
    # This does model inference with vLLM
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Step 1: Running 30_search_multi.py (model inference)..."
    echo "  This will take ~1.5-2.5 hours per model"

    cd "$CODE_DIR"
    export CUDA_VISIBLE_DEVICES=2
    $PYTHON ./30_search_multi.py \
        --model "$model_path" \
        --answer_path "$ANSWER_DIR" \
        --neuron_path "$NEURON_DIR" \
        --batch_size 32 \
        --dataset "$DATASET"

    # Check if 30_search_multi.py succeeded (should have .pt files)
    pt_count=$(ls "$NEURON_DIR"/*.pt 2>/dev/null | wc -l)
    if [ "$pt_count" -eq 0 ]; then
        echo "[ERROR] No .pt files generated for ${model_short}! Skipping process_multi."
        continue
    fi
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Step 1 complete: ${pt_count} .pt files in ${NEURON_DIR}"

    # Step 2: Run 20_process_multi.py for each percentage
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Step 2: Running 20_process_multi.py for all percentages..."

    DELTA=0.5
    for percent in "${PERCENTS[@]}"; do
        echo "  [$(date '+%H:%M:%S')] Processing ${model_short}: top ${percent}%"
        $PYTHON ./20_process_multi.py \
            --neuron_dir "$NEURON_DIR" \
            --out_dir "$OUT_DIR" \
            --delta ${DELTA} \
            --save_percent \
            --top_percent ${percent}
    done

    # Step 3: Rename _top_{percent}percent.json → _{percent}.json
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Step 3: Renaming files for compatibility..."
    cd "$OUT_DIR"
    for f in *_top_*.json; do
        newname=$(echo "$f" | sed 's/_top_/_/;s/percent//')
        mv "$f" "$newname"
        echo "  ${f} -> ${newname}"
    done
    # Also rename .txt files if needed
    for f in *_top_*.txt; do
        newname=$(echo "$f" | sed 's/_top_/_/;s/percent//')
        mv "$f" "$newname"
        echo "  ${f} -> ${newname}"
    done

    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done with model: ${model_short}"
    echo "  Results saved to: ${OUT_DIR}"
done

echo ""
echo "============================================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ALL MODELS COMPLETE!"
echo "============================================================"

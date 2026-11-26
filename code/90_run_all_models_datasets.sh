export CUDA_VISIBLE_DEVICES=2
PYTHON="<REPO_ROOT>/.venv/bin/python"

# Models: (short_name, model_path)
MODELS=(
    "gemma:<MODELS_DIR>/gemma-2-9b-it"
    "llama:<MODELS_DIR>/Llama-3.1-8B-Instruct"
    "qwen:<MODELS_DIR>/Qwen3-8B"
)

# Datasets: (short_name, neuron_subdir, out_subdir)
# Using existing neuron results directories
DATASETS=(
    "mmlu:mmlu_neuron_results:mmlu_for_dataset_compare"
    "psoups:psoups_neuron_results:psoups_for_dataset_compare"
)

PREFERENCES=("Informativeness_high")
DELTA=0.1
PERCENTS=(0.01 0.1 0.05 0.5)
BASE_DIR="<REPO_ROOT>/res"

for model_entry in "${MODELS[@]}"; do
    model_short="${model_entry%%:*}"
    model_path="${model_entry##*:}"

    for dataset_entry in "${DATASETS[@]}"; do
        IFS=':' read -r ds_short neuron_subdir out_subdir <<< "$dataset_entry"

        NEURON_DIR="${BASE_DIR}/${model_short}/${neuron_subdir}"
        OUT_DIR="${BASE_DIR}/${model_short}/${out_subdir}"

        echo "========================================="
        echo "Model: ${model_short} (${model_path})"
        echo "Dataset: ${ds_short}"
        echo "NEURON_DIR: ${NEURON_DIR}"
        echo "OUT_DIR: ${OUT_DIR}"
        echo "========================================="

        mkdir -p "$OUT_DIR"

        for p in "${PREFERENCES[@]}"; do
            for percent in "${PERCENTS[@]}"; do
                echo "  [$(date '+%H:%M:%S')] Processing ${model_short}/${ds_short}: preference=${p}, percent=${percent}%"
                $PYTHON ./20_process_with_one.py \
                    --neuron_dir "$NEURON_DIR" \
                    --out_dir "$OUT_DIR" \
                    --delta $DELTA \
                    --preference $p \
                    --save_percent \
                    --top_percent ${percent}
            done
        done
    done
done

echo ""
echo "All done! $(date)"

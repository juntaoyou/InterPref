export CUDA_VISIBLE_DEVICES=1
# Define parameters
# MODEL_PATH="<MODELS_DIR>/Qwen3-8B"
# MODEL_PATH="<MODELS_DIR>/Llama-3.1-8B-Instruct"
MODEL_PATH="<MODELS_DIR>/gemma-2-9b-it"
model="gemma"
Dataset="<DATA_ROOT>/mmlu_sampled_by_psoups"

NEURON_DIR="<REPO_ROOT>/${model}/mmlu_neuron_results"
BATCH_SIZE=32
OUT_DIR="<REPO_ROOT>/${model}/mmlu_for_dataset_compare"
PREFERENCES=( "Informativeness_high")
# PREFERENCES=("Style_low")
DELTA=0.1
# PERCENTS=(0.05 0.1 2.0 4.0 8.0 10.0 12.0 14.0 16.0 18.0 20.0)
PERCENTS=(0.01 0.5)
# python ./10_search.py \
#    --model "$MODEL_PATH" \
#    --answer_path "$ANSWER_DIR" \
#    --neuron_path "$NEURON_DIR" \
#    --batch_size $BATCH_SIZE \
#    --dataset $Dataset

for p in "${PREFERENCES[@]}"; do
    for percent in "${PERCENTS[@]}"; do
        python ./20_process_with_one.py\
            --neuron_dir "$NEURON_DIR" \
            --out_dir "$OUT_DIR" \
            --delta $DELTA \
            --preference $p \
            --save_percent \
            --top_percent ${percent}
    done
done

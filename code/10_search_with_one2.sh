export CUDA_VISIBLE_DEVICES=0
# Define parameters
#MODEL_PATH="/home/<user>/models/Qwen3-8B"
MODEL_PATH="<MODELS_DIR>/Llama-3.1-8B-Instruct"
#MODEL_PATH="/home/<user>/models/gemma-2-9b-it"
ANSWER_DIR="./llama/answer_results"
NEURON_DIR="./llama/neuron_results_new"
BATCH_SIZE=32
OUT_DIR="./llama/neuron_results_with_one"
PREFERENCE=("Informativeness_low" "Informativeness_high")
DELTA=0.08
# Run the Python script with arguments
#types=("Expertise" "Informativeness" "Style")
#for PREFERENCE in "${types[@]}"; do
#python ./10_search.py \
#    --model "$MODEL_PATH" \
#    --answer_path "$ANSWER_DIR" \
#    --neuron_path "$NEURON_DIR" \
#    --batch_size $BATCH_SIZE \
# done
for p in "${PREFERENCE[@]}"; do
python ./20_process_with_one.py\
    --neuron_dir "$NEURON_DIR" \
    --out_dir "$OUT_DIR" \
    --delta $DELTA \
    --preference $p
done

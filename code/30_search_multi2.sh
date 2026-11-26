export CUDA_VISIBLE_DEVICES=0
# Define parameters
#MODEL_PATH="/home/<user>/models/Llama-3.1-8B-Instruct"
MODEL_PATH="<MODELS_DIR>"/Qwen3-8B
#MODEL_PATH=/home/<user>/models/gemma-2-9b-it
ANSWER_DIR="./qwen/analysis/answer_results"
NEURON_DIR="./qwen/analysis/neuron_results"
BATCH_SIZE=32
# OUT_DIR="./qwen/analysis/neuron_results_with_one_new"
# PREFERENCE="Expertise"

# Run the Python script with arguments
#types=("Expertise" "Informativeness" "Style")
#for PREFERENCE in "${types[@]}"; do
python ./30_search_multi2.py \
    --model "$MODEL_PATH" \
    --answer_path "$ANSWER_DIR" \
    --neuron_path "$NEURON_DIR" \
    --batch_size $BATCH_SIZE \
#    --preference ${PREFERENCE}
#done

#python ./20_process_with_one.py\
#    --neuron_dir "$NEURON_DIR" \
#    --out_dir "$OUT_DIR"

export CUDA_VISIBLE_DEVICES=6
# Define parameters
# MODEL_PATH="/home/<user>/models/Qwen3-8B"
MODEL_PATH="<MODELS_DIR>/gemma-2-9b-it"
# MODEL_PATH="<MODELS_DIR>/Llama-3.1-8B-Instruct"
model="gemma"
ANSWER_DIR="../${model}/answer_results_multi"
NEURON_DIR="../${model}/neuron_results_multi_on_psoups"
BATCH_SIZE=32
OUT_DIR="../${model}/psoups_multi_detect"
DATASET="<DATA_ROOT>"/PSoups_queries
# python ./30_search_multi.py \
#    --model $MODEL_PATH \
#    --answer_path $ANSWER_DIR \
#    --neuron_path $NEURON_DIR \
#    --batch_size $BATCH_SIZE \
#    --dataset $DATASET
echo ${NEURON_DIR}
echo ${OUT_DIR}
DELTA=0.5
# PERCENTS=(2.0 4.0 6.0 8.0 10.0 12.0 14.0 16.0 18.0 20.0)
PERCENTS=(0.5 1.0)
for percent in "${PERCENTS[@]}"; do
python ./20_process_multi.py\
    --neuron_dir "$NEURON_DIR" \
    --out_dir $OUT_DIR \
    --delta ${DELTA} \
    --save_percent \
    --top_percent ${percent}
done


# MODEL_PATH="<MODELS_DIR>/gemma-2-9b-it"
MODEL_PATH="<MODELS_DIR>/Llama-3.1-8B-Instruct"
model="llama"
ANSWER_DIR="../${model}/answer_results_multi"
NEURON_DIR="../${model}/neuron_results_multi_on_psoups"
BATCH_SIZE=32
OUT_DIR="../${model}/psoups_multi_detect"
for percent in "${PERCENTS[@]}"; do
python ./20_process_multi.py\
    --neuron_dir "$NEURON_DIR" \
    --out_dir $OUT_DIR \
    --delta ${DELTA} \
    --save_percent \
    --top_percent ${percent}
done
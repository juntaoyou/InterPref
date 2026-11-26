#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
export VLLM_ENABLE_V1_MULTIPROCESSING=0
MODEL_PATH="<MODELS_DIR>/Qwen3-8B"
# MODEL_PATH="<MODELS_DIR>/Llama-3.1-8B-Instruct"
# MODEL_PATH="<MODELS_DIR>/gemma-2-9b-it"
if [[ "$MODEL_PATH" == "<MODELS_DIR>/Qwen3-8B" ]]; then
    model='qwen'
elif [[ "$MODEL_PATH" == "<MODELS_DIR>/Llama-3.1-8B-Instruct" ]]; then
    model='llama'
elif [[ "$MODEL_PATH" == "<MODELS_DIR>/gemma-2-9b-it" ]]; then
    model='gemma'
else
    echo "Unknown model path: $MODEL_PATH"
    exit 1
fi

d="psoups"
NEURON_DIR="<REPO_ROOT>/${model}/${d}_for_dataset_compare"
# OUTPUT_DIR="./answer_results_deactivate"
BATCH_SIZE=32
DEACTIVATE="preference_neurons"
# DEACTIVATE="random"
PYTHON_FILE="./40_deactivate.py"
OUTPUT_DIR=../${model}/mmlu_test/answers_deactivate_${DEACTIVATE}_${d}
# DATASET="<DATA_ROOT>/psoups_eval"
DATASET="<DATA_ROOT>/mmlu_test"
# PERCENTS=(2.0 4.0 6.0 8.0 10.0 12.0 14.0 16.0 18.0 20.0)
PERCENTS=(6.0)
# 执行 Python 脚本，并传递参数
DELTA_CONFIG='{"Expertise_low": 6.0, "Expertise_high": 2.0, "Informativeness_high": 0.5, "Style_high": 0.1}'
# for percent in "${PERCENTS[@]}"; do
python $PYTHON_FILE \
  --model $MODEL_PATH \
  --neuron_dir $NEURON_DIR \
  --output_dir $OUTPUT_DIR \
  --batch_size $BATCH_SIZE \
  --deactivate ${DEACTIVATE} \
  --dataset ${DATASET} \
  --delta_config '{"Style_high":12.0}'
# done

# MODEL_PATH="<MODELS_DIR>/Llama-3.1-8B-Instruct"
# model='llama'
# # MODEL_PATH="<MODELS_DIR>/Llama-3.1-8B-Instruct"
# # MODEL_PATH="<MODELS_DIR>/gemma-2-9b-it"

# NEURON_DIR="<REPO_ROOT>/${model}/${d}_for_dataset_compare"
# # OUTPUT_DIR="./answer_results_deactivate"
# BATCH_SIZE=32
# DEACTIVATE="preference_neurons"
# # DEACTIVATE="random"
# PYTHON_FILE="./40_deactivate.py"
# OUTPUT_DIR=../${model}/mmlu_test/answers_deactivate_${DEACTIVATE}_${d}
# # DATASET="<DATA_ROOT>/psoups_eval"
# DATASET="<DATA_ROOT>/mmlu_test"
# # PERCENTS=(2.0 4.0 6.0 8.0 10.0 12.0 14.0 16.0 18.0 20.0)
# PERCENTS=(6.0)
# # 执行 Python 脚本，并传递参数
# DELTA_CONFIG='{"Expertise_low": 8.0, "Expertise_high": 10.0, "Informativeness_low": 10.0, "Informativeness_high": 10.0, "Style_low": 10.0, "Style_high": 10.0}'
# # for percent in "${PERCENTS[@]}"; do
# python $PYTHON_FILE \
#   --model $MODEL_PATH \
#   --neuron_dir $NEURON_DIR \
#   --output_dir $OUTPUT_DIR \
#   --batch_size $BATCH_SIZE \
#   --deactivate ${DEACTIVATE} \
#   --dataset ${DATASET} \
#   --delta_config '{"Expertise_low": 8.0, "Expertise_high": 10.0, "Informativeness_low": 10.0, "Informativeness_high": 10.0, "Style_low": 10.0, "Style_high": 10.0}'
# done

# MODEL_PATH="<MODELS_DIR>/gemma-2-9b-it"
# model='gemma'
# # MODEL_PATH="<MODELS_DIR>/Llama-3.1-8B-Instruct"
# # MODEL_PATH="<MODELS_DIR>/gemma-2-9b-it"

# NEURON_DIR="<REPO_ROOT>/${model}/psoups_for_dataset_compare"
# # OUTPUT_DIR="./answer_results_deactivate"
# BATCH_SIZE=32
# DEACTIVATE="preference_neurons"
# # DEACTIVATE="random"
# PYTHON_FILE="./40_deactivate.py"
# OUTPUT_DIR=../${model}/mmlu_test/answers_deactivate_${DEACTIVATE}
# # DATASET="<DATA_ROOT>/psoups_eval"
# DATASET="<DATA_ROOT>/mmlu_test"
# PERCENTS=(2.0 4.0 6.0 8.0 10.0 12.0 14.0 16.0 18.0 20.0)
# # 执行 Python 脚本，并传递参数

# for percent in "${PERCENTS[@]}"; do
# python $PYTHON_FILE \
#   --model $MODEL_PATH \
#   --neuron_dir $NEURON_DIR \
#   --output_dir $OUTPUT_DIR \
#   --batch_size $BATCH_SIZE \
#   --deactivate ${DEACTIVATE} \
#   --dataset ${DATASET} \
#   --delta ${percent}
# done
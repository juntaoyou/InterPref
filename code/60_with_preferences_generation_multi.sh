#!/bin/bash
source "<REPO_ROOT>"/.venv/bin/activate
cd "<REPO_ROOT>"/code

export CUDA_VISIBLE_DEVICES=4
export NCCL_P2P_DISABLE=1
# model='qwen' 
MODEL_DIR="<MODELS_DIR>"
FILE_DIR="<PROJECT_ROOT>"
CONFIG_DIR="<PROJECT_ROOT>"
DATA_DIR="<PROJECT_ROOT>"/data
NEURONS_DIR="<PROJECT_ROOT>"
ROOT_DIR="<PROJECT_ROOT>"/eval
MODEL_PATHS=(
    "${MODEL_DIR}/Qwen3-8B"
    # "${MODEL_DIR}/Llama3.1-8B"
    # "${MODEL_DIR}/gemma-2-9b-it"
)

for MODEL_PATH in "${MODEL_PATHS[@]}"; do

if [[ "$MODEL_PATH" == "${MODEL_DIR}/Qwen3-8B" ]]; then
    model='qwen'
elif [[ "$MODEL_PATH" == "${MODEL_DIR}/Llama3.1-8B" ]]; then
    model='llama'
elif [[ "$MODEL_PATH" == "${MODEL_DIR}/gemma-2-9b-it" ]]; then
    model='gemma'
else
    echo "Unknown model path: $MODEL_PATH"
    exit 1
fi

# MODEL_PATH="/home/<user>/models/Llama-3.1-8B-Instruct"
NEURON_DIR=${FILE_DIR}/${model}/psoups_for_dataset_compare
NEURON_MULTI_DIR=${FILE_DIR}/${model}/psoups_multi_detect
CONFIG_DIR=${CONFIG_DIR}/eval/${model}/configs

BATCH_SIZE=32
PYTHON_FILE="./60_with_preferences_generation_multi.py"  # 指定你的 Python 文件名
DATASET="${DATA_DIR}/psoups_eval"
# DATASET="${ROOT_DIR}/data/mmlu_test"
MODES=('partial2' 'partial1')
# OUTPUT_DIR="<REPO_ROOT>"/gemma/answer_${MODE}_steer_${PERCENTS}
echo "[INFO] model: ${MODEL_PATH}"
echo "[INFO] neuron path: ${NEURON_DIR}"
echo "[INFO] config dir: ${CONFIG_DIR}"

# 执行 Python 脚本，并传递参数
for MODE in "${MODES[@]}"; do
  CONFIG_FILE="${CONFIG_DIR}/with_preferences_steer_${MODE}_config.json"
  OUTPUT_DIR="${ROOT_DIR}/${model}/answer_with_preferences_${MODE}_steer"
  echo "[INFO] config file: ${CONFIG_FILE}"
  echo "[INFO] output path: ${OUTPUT_DIR}"
  echo "[INFO] mode: ${MODE}"
  
  python $PYTHON_FILE \
    --model $MODEL_PATH \
    --neuron_dir $NEURON_DIR \
    --neuron_multi_dir $NEURON_MULTI_DIR \
    --output_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE \
    --dataset ${DATASET} \
    --mode ${MODE} \
    --percent 6.0 \
    --config_file ${CONFIG_FILE}
done

done

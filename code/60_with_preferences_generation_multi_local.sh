#!/bin/bash
cd "<REPO_ROOT>"/code & source "<REPO_ROOT>"/.venv/bin/activate

export CUDA_VISIBLE_DEVICES=3
export VLLM_ENABLE_V1_MULTIPROCESSING=0

# MODEL_PATHS and MODEL_NAMES
MODEL_PATHS=(
    "<MODELS_DIR>/Qwen3-8B"
    # "<MODELS_DIR>/gemma-2-9b-it"
    # "<MODELS_DIR>/Llama-3.1-8B-Instruct"
)
# MODEL_NAMES=("qwen" "gemma" "llama")
MODEL_NAMES=("qwen")
for i in "${!MODEL_PATHS[@]}"; do
    MODEL_PATH="${MODEL_PATHS[$i]}"
    MODEL="${MODEL_NAMES[$i]}"
    
    NEURON_DIR="<REPO_ROOT>"/res/${MODEL}/psoups_for_dataset_compare
    NEURON_MULTI_DIR="<REPO_ROOT>"/res/${MODEL}/psoups_multi_detect
    CONFIG_DIR="<REPO_ROOT>"/eval/${MODEL}/configs

    BATCH_SIZE=32
    PYTHON_FILE="./60_with_preferences_generation_multi.py"
    DATASET="<DATA_ROOT>/psoups_eval"
    # DATASET="<DATA_ROOT>/mmlu_test"
    MODES=("partial1" "partial2")
    # MODES=("partial2")
    
    echo "[INFO] model: ${MODEL_PATH}"
    echo "[INFO] model name: ${MODEL}"
    echo "[INFO] neuron path: ${NEURON_DIR}"
    echo "[INFO] config dir: ${CONFIG_DIR}"

    # 执行 Python 脚本，并传递参数
    for MODE in "${MODES[@]}"; do
        CONFIG_FILE="${CONFIG_DIR}/with_preferences_steer_${MODE}_config.json"
        OUTPUT_DIR="<REPO_ROOT>"/res/${MODEL}/psoups/answer_with_preferences_${MODE}_steer
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
            --percent 2.0 \
            --config_file ${CONFIG_FILE}
    done
done
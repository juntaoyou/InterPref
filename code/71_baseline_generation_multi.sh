#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
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

BATCH_SIZE=32
PYTHON_FILE="./71_baseline_generation_multi.py"  # 指定你的 Python 文件名
OUTPUT_DIR="../${model}/mmlu_test/answer_results_multi_baseline"
# DATASET="<DATA_ROOT>/psoups_eval"
DATASET="<DATA_ROOT>/mmlu_test"
# 执行 Python 脚本，并传递参数
python $PYTHON_FILE \
  --model $MODEL_PATH \
  --output_dir $OUTPUT_DIR \
  --batch_size $BATCH_SIZE \
  --dataset ${DATASET}

# MODEL_PATH="<MODELS_DIR>/Qwen3-8B"
MODEL_PATH="<MODELS_DIR>/Llama-3.1-8B-Instruct"
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

BATCH_SIZE=32
PYTHON_FILE="./71_baseline_generation_multi.py"  # 指定你的 Python 文件名
OUTPUT_DIR="../${model}/mmlu_test/answer_results_multi_baseline"
# DATASET="<DATA_ROOT>/psoups_eval"
DATASET="<DATA_ROOT>/mmlu_test"
# 执行 Python 脚本，并传递参数
python $PYTHON_FILE \
  --model $MODEL_PATH \
  --output_dir $OUTPUT_DIR \
  --batch_size $BATCH_SIZE \
  --dataset ${DATASET}

# MODEL_PATH="<MODELS_DIR>/Qwen3-8B"
# MODEL_PATH="<MODELS_DIR>/Llama-3.1-8B-Instruct"
MODEL_PATH="<MODELS_DIR>/gemma-2-9b-it"
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

BATCH_SIZE=32
PYTHON_FILE="./71_baseline_generation_multi.py"  # 指定你的 Python 文件名
OUTPUT_DIR="../${model}/mmlu_test/answer_results_multi_baseline"
# DATASET="<DATA_ROOT>/psoups_eval"
DATASET="<DATA_ROOT>/mmlu_test"
# 执行 Python 脚本，并传递参数
python $PYTHON_FILE \
  --model $MODEL_PATH \
  --output_dir $OUTPUT_DIR \
  --batch_size $BATCH_SIZE \
  --dataset ${DATASET}
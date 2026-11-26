#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
export VLLM_ENABLE_V1_MULTIPROCESSING=0
# export NCCL_P2P_DISABLE=1

DATASET="<DATA_ROOT>/psoups_eval"
PERCENTS=(0.5 1.0)
# DATASET="<DATA_ROOT>/my_alpaca_eval"
# MODEL_PATH="<MODELS_DIR>/Qwen3-8B"
# # MODEL_PATH="<MODELS_DIR>/gemma-2-9b-it"
# # MODEL_PATH="<MODELS_DIR>/Llama-3.1-8B-Instruct"
# model="qwen"
# NEURON_DIR="<REPO_ROOT>/${model}/psoups_for_dataset_compare"
# # NEURON_DIR="<REPO_ROOT>/llama/psoups_for_dataset_compare"
# # NEURON_DIR="<REPO_ROOT>/gemma/psoups_for_dataset_compare"
# # OUTPUT_DIR="./answer_results_deactivate"
# BATCH_SIZE=32
# DEACTIVATE="preference_neurons"
# PYTHON_FILE="./41_deactivate_multi.py"
# PERCENTS=(6.0)
# # for percent in "${PERCENTS[@]}"; do
# # MODES=('union' 'both' 'intersect' 'partial1' 'partial2')
# MODES=('overlap')
# NEURON_DIR_MULTI="<REPO_ROOT>"/${model}/psoups_multi_detect
# # NEURON_DIR_MULTI="<REPO_ROOT>"/llama/psoups_multi_detect
# # NEURON_DIR_MULTI="<REPO_ROOT>"/gemma/psoups_multi_detect
# # OUTPUT_DIR=./union_both_intersect/psoups/answer_results_multi_with_one_deactivate_${MODE}_${DEACTIVATE}_alpaca


# # DATASET="<DATA_ROOT>/psoups_eval"

# # 执行 Python 脚本，并传递参数
# for mode in "${MODES[@]}"; do
#   for percent in "${PERCENTS[@]}"; do
#   OUTPUT_DIR="<REPO_ROOT>"/${model}/answer_results_multi_deactivate_${mode}_${DEACTIVATE}_${percent}
#   echo "[INFO] mode: ${mode}"
#   echo "[INFO] output path: ${OUTPUT_DIR}"
#   python $PYTHON_FILE \
#     --model $MODEL_PATH \
#     --neuron_dir_multi $NEURON_DIR_MULTI \
#     --neuron_dir $NEURON_DIR \
#     --output_dir $OUTPUT_DIR \
#     --batch_size $BATCH_SIZE \
#     --deactivate ${DEACTIVATE} \
#     --mode ${mode} \
#     --dataset ${DATASET} \
#     --percent ${percent}
#   done
# done



MODEL_PATH="<MODELS_DIR>/Llama-3.1-8B-Instruct"
model="llama"
NEURON_DIR="<REPO_ROOT>/${model}/psoups_for_dataset_compare"
BATCH_SIZE=16
DEACTIVATE="preference_neurons"
PYTHON_FILE="./41_deactivate_multi.py"

# for percent in "${PERCENTS[@]}"; do
# MODES=('union' 'both' 'intersect' 'partial1' 'partial2')
MODES=('overlap')
NEURON_DIR_MULTI="<REPO_ROOT>"/${model}/psoups_multi_detect

for mode in "${MODES[@]}"; do
  for percent in "${PERCENTS[@]}"; do
  OUTPUT_DIR="<REPO_ROOT>"/${model}/answer_results_multi_deactivate_${mode}_${DEACTIVATE}_${percent}
  echo "[INFO] mode: ${mode}"
  echo "[INFO] output path: ${OUTPUT_DIR}"
  python $PYTHON_FILE \
    --model $MODEL_PATH \
    --neuron_dir_multi $NEURON_DIR_MULTI \
    --neuron_dir $NEURON_DIR \
    --output_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE \
    --deactivate ${DEACTIVATE} \
    --mode ${mode} \
    --dataset ${DATASET} \
    --percent ${percent}
  done
done

# MODEL_PATH="<MODELS_DIR>/Qwen3-8B"
MODEL_PATH="<MODELS_DIR>/gemma-2-9b-it"
# MODEL_PATH="<MODELS_DIR>/Llama-3.1-8B-Instruct"
model="gemma"
NEURON_DIR="<REPO_ROOT>/${model}/psoups_for_dataset_compare"

BATCH_SIZE=16
DEACTIVATE="preference_neurons"
PYTHON_FILE="./41_deactivate_multi.py"

# for percent in "${PERCENTS[@]}"; do
# MODES=('union' 'both' 'intersect' 'partial1' 'partial2')
MODES=('overlap')
NEURON_DIR_MULTI="<REPO_ROOT>"/${model}/psoups_multi_detect


# DATASET="<DATA_ROOT>/my_alpaca_eval"
# 执行 Python 脚本，并传递参数
for mode in "${MODES[@]}"; do
  for percent in "${PERCENTS[@]}"; do
  OUTPUT_DIR="<REPO_ROOT>"/${model}/answer_results_multi_deactivate_${mode}_${DEACTIVATE}_${percent}
  echo "[INFO] mode: ${mode}"
  echo "[INFO] output path: ${OUTPUT_DIR}"
  python $PYTHON_FILE \
    --model $MODEL_PATH \
    --neuron_dir_multi $NEURON_DIR_MULTI \
    --neuron_dir $NEURON_DIR \
    --output_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE \
    --deactivate ${DEACTIVATE} \
    --mode ${mode} \
    --dataset ${DATASET} \
    --percent ${percent}
  done
done



# DATASET="<DATA_ROOT>/psoups_eval"
PERCENTS=(0.5 1.0)
DATASET="<DATA_ROOT>/my_alpaca_eval"
# MODEL_PATH="<MODELS_DIR>/Qwen3-8B"
# # MODEL_PATH="<MODELS_DIR>/gemma-2-9b-it"
# # MODEL_PATH="<MODELS_DIR>/Llama-3.1-8B-Instruct"
# model="qwen"
# NEURON_DIR="<REPO_ROOT>/${model}/psoups_for_dataset_compare"
# # NEURON_DIR="<REPO_ROOT>/llama/psoups_for_dataset_compare"
# # NEURON_DIR="<REPO_ROOT>/gemma/psoups_for_dataset_compare"
# # OUTPUT_DIR="./answer_results_deactivate"
# BATCH_SIZE=32
# DEACTIVATE="preference_neurons"
# PYTHON_FILE="./41_deactivate_multi.py"
# PERCENTS=(6.0)
# # for percent in "${PERCENTS[@]}"; do
# # MODES=('union' 'both' 'intersect' 'partial1' 'partial2')
# MODES=('overlap')
# NEURON_DIR_MULTI="<REPO_ROOT>"/${model}/psoups_multi_detect
# # NEURON_DIR_MULTI="<REPO_ROOT>"/llama/psoups_multi_detect
# # NEURON_DIR_MULTI="<REPO_ROOT>"/gemma/psoups_multi_detect
# # OUTPUT_DIR=./union_both_intersect/psoups/answer_results_multi_with_one_deactivate_${MODE}_${DEACTIVATE}_alpaca


# # DATASET="<DATA_ROOT>/psoups_eval"

# # 执行 Python 脚本，并传递参数
# for mode in "${MODES[@]}"; do
#   for percent in "${PERCENTS[@]}"; do
#   OUTPUT_DIR="<REPO_ROOT>"/${model}/answer_results_multi_deactivate_${mode}_${DEACTIVATE}_${percent}
#   echo "[INFO] mode: ${mode}"
#   echo "[INFO] output path: ${OUTPUT_DIR}"
#   python $PYTHON_FILE \
#     --model $MODEL_PATH \
#     --neuron_dir_multi $NEURON_DIR_MULTI \
#     --neuron_dir $NEURON_DIR \
#     --output_dir $OUTPUT_DIR \
#     --batch_size $BATCH_SIZE \
#     --deactivate ${DEACTIVATE} \
#     --mode ${mode} \
#     --dataset ${DATASET} \
#     --percent ${percent}
#   done
# done



MODEL_PATH="<MODELS_DIR>/Llama-3.1-8B-Instruct"
model="llama"
NEURON_DIR="<REPO_ROOT>/${model}/psoups_for_dataset_compare"
BATCH_SIZE=16
DEACTIVATE="preference_neurons"
PYTHON_FILE="./41_deactivate_multi.py"

# for percent in "${PERCENTS[@]}"; do
# MODES=('union' 'both' 'intersect' 'partial1' 'partial2')
MODES=('overlap')
NEURON_DIR_MULTI="<REPO_ROOT>"/${model}/psoups_multi_detect

for mode in "${MODES[@]}"; do
  for percent in "${PERCENTS[@]}"; do
  OUTPUT_DIR="<REPO_ROOT>"/${model}/answer_results_multi_deactivate_${mode}_${DEACTIVATE}_${percent}_alpaca
  echo "[INFO] mode: ${mode}"
  echo "[INFO] output path: ${OUTPUT_DIR}"
  python $PYTHON_FILE \
    --model $MODEL_PATH \
    --neuron_dir_multi $NEURON_DIR_MULTI \
    --neuron_dir $NEURON_DIR \
    --output_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE \
    --deactivate ${DEACTIVATE} \
    --mode ${mode} \
    --dataset ${DATASET} \
    --percent ${percent}
  done
done

# MODEL_PATH="<MODELS_DIR>/Qwen3-8B"
MODEL_PATH="<MODELS_DIR>/gemma-2-9b-it"
# MODEL_PATH="<MODELS_DIR>/Llama-3.1-8B-Instruct"
model="gemma"
NEURON_DIR="<REPO_ROOT>/${model}/psoups_for_dataset_compare"

BATCH_SIZE=16
DEACTIVATE="preference_neurons"
PYTHON_FILE="./41_deactivate_multi.py"

# for percent in "${PERCENTS[@]}"; do
# MODES=('union' 'both' 'intersect' 'partial1' 'partial2')
MODES=('overlap')
NEURON_DIR_MULTI="<REPO_ROOT>"/${model}/psoups_multi_detect


# DATASET="<DATA_ROOT>/my_alpaca_eval"
# 执行 Python 脚本，并传递参数
for mode in "${MODES[@]}"; do
  for percent in "${PERCENTS[@]}"; do
  OUTPUT_DIR="<REPO_ROOT>"/${model}/answer_results_multi_deactivate_${mode}_${DEACTIVATE}_${percent}_alpaca
  echo "[INFO] mode: ${mode}"
  echo "[INFO] output path: ${OUTPUT_DIR}"
  python $PYTHON_FILE \
    --model $MODEL_PATH \
    --neuron_dir_multi $NEURON_DIR_MULTI \
    --neuron_dir $NEURON_DIR \
    --output_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE \
    --deactivate ${DEACTIVATE} \
    --mode ${mode} \
    --dataset ${DATASET} \
    --percent ${percent}
  done
done
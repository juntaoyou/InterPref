#!/bin/bash
source "<REPO_ROOT>"/.venv/bin/activate
cd "<REPO_ROOT>"/code

export NCCL_P2P_DISABLE=1

MODEL_DIR="<MODELS_DIR>"
FILE_DIR="<PROJECT_ROOT>"
DATASET="${FILE_DIR}/data/mmlu_test"

BATCH_SIZE=32

MODEL_PATHS=(
    "${MODEL_DIR}/Qwen3-8B"
    "${MODEL_DIR}/Llama3.1-8B"
    "${MODEL_DIR}/gemma-2-9b-it"
)

echo "=========================================="
echo " 开始生成所有缺失文件"
echo "=========================================="

for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    if [[ "$MODEL_PATH" == "${MODEL_DIR}/Qwen3-8B" ]]; then
        model='qwen'
    elif [[ "$MODEL_PATH" == "${MODEL_DIR}/Llama3.1-8B" ]]; then
        model='llama'
    elif [[ "$MODEL_PATH" == "${MODEL_DIR}/gemma-2-9b-it" ]]; then
        model='gemma'
    else
        echo "Unknown model path: $MODEL_PATH"
        continue
    fi

    NEURON_DIR=${FILE_DIR}/${model}/psoups_for_dataset_compare
    NEURON_MULTI_DIR=${FILE_DIR}/${model}/psoups_multi_detect
    CONFIG_DIR=${FILE_DIR}/${model}/configs

    echo ""
    echo "------------------------------------------"
    echo " 处理模型: ${model}"
    echo "------------------------------------------"

    echo ""
    echo "[Step 1/3] 生成 MultiSteer/Both 和 MultiSteer/Union"
    echo "---------------------------------------------------"
    
    PYTHON_FILE="./52_generation_multi.py"
    MODES=('both' 'union')
    
    for MODE in "${MODES[@]}"; do
        CONFIG_FILE="${CONFIG_DIR}/steer_${MODE}_config.json"
        OUTPUT_DIR="${FILE_DIR}/${model}/answer_${MODE}_steer"
        
        echo ""
        echo "[INFO] 模型: ${model}, 模式: ${MODE}"
        echo "[INFO] 输出目录: ${OUTPUT_DIR}"
        echo "[INFO] 配置文件: ${CONFIG_FILE}"
        
        python $PYTHON_FILE \
            --model $MODEL_PATH \
            --neuron_dir $NEURON_DIR \
            --neuron_multi_dir $NEURON_MULTI_DIR \
            --output_dir $OUTPUT_DIR \
            --batch_size $BATCH_SIZE \
            --dataset $DATASET \
            --mode $MODE \
            --config_file $CONFIG_FILE
        
        if [ $? -eq 0 ]; then
            echo "[SUCCESS] ${model}_${MODE} 生成完成"
        else
            echo "[ERROR] ${model}_${MODE} 生成失败"
        fi
    done

    echo ""
    echo "[Step 2/3] 生成 deactivate 系列 (both, union, overlap)"
    echo "------------------------------------------------------"
    
    PYTHON_FILE="./41_deactivate_multi.py"
    BATCH_SIZE_DEACTIVATE=16
    DEACTIVATE="preference_neurons"
    
    if [[ "$model" == "qwen" ]]; then
        DEACTIVATE_MODES=('both' 'union')
    else
        DEACTIVATE_MODES=('both' 'union' 'overlap')
    fi
    
    for mode in "${DEACTIVATE_MODES[@]}"; do
        CONFIG_FILE="${CONFIG_DIR}/steer_${mode}_config.json"
        OUTPUT_DIR="${FILE_DIR}/${model}/answer_results_multi_deactivate_${mode}_${DEACTIVATE}"
        
        echo ""
        echo "[INFO] 模型: ${model}, 去激活模式: ${mode}"
        echo "[INFO] 输出目录: ${OUTPUT_DIR}"
        echo "[INFO] 配置文件: ${CONFIG_FILE}"
        
        python $PYTHON_FILE \
            --model $MODEL_PATH \
            --neuron_dir_multi $NEURON_MULTI_DIR \
            --neuron_dir $NEURON_DIR \
            --output_dir $OUTPUT_DIR \
            --batch_size $BATCH_SIZE_DEACTIVATE \
            --deactivate $DEACTIVATE \
            --mode $mode \
            --dataset $DATASET \
            --config_file $CONFIG_FILE
        
        if [ $? -eq 0 ]; then
            echo "[SUCCESS] ${model}_deactivate_${mode} 生成完成"
        else
            echo "[ERROR] ${model}_deactivate_${mode} 生成失败"
        fi
    done

    echo ""
    echo "[Step 3/3] 生成 partial steer 系列"
    echo "----------------------------------"
    
    if [[ "$model" != "qwen" ]]; then
        PARTIAL_MODES=('partial1' 'partial2')
        
        for mode in "${PARTIAL_MODES[@]}"; do
            CONFIG_FILE="${CONFIG_DIR}/with_preferences_steer_${mode}_config.json"
            OUTPUT_DIR="${FILE_DIR}/${model}/answer_results_multi_steer_${mode}"
            
            echo ""
            echo "[INFO] 模型: ${model}, Partial模式: ${mode}"
            echo "[INFO] 输出目录: ${OUTPUT_DIR}"
            echo "[INFO] 配置文件: ${CONFIG_FILE}"
            
            python $PYTHON_FILE \
                --model $MODEL_PATH \
                --neuron_dir_multi $NEURON_MULTI_DIR \
                --neuron_dir $NEURON_DIR \
                --output_dir $OUTPUT_DIR \
                --batch_size $BATCH_SIZE_DEACTIVATE \
                --deactivate $DEACTIVATE \
                --mode $mode \
                --dataset $DATASET \
                --config_file $CONFIG_FILE
            
            if [ $? -eq 0 ]; then
                echo "[SUCCESS] ${model}_partial_${mode} 生成完成"
            else
                echo "[ERROR] ${model}_partial_${mode} 生成失败"
            fi
        done
    else
        echo "[SKIP] qwen 的 partial1_steer 和 partial2_steer 已完整"
    fi

done

echo ""
echo "=========================================="
echo " 所有缺失文件生成任务已完成"
echo "=========================================="
export CUDA_VISIBLE_DEVICES=6,7
export NCCL_P2P_DISABLE=1
PYTHON_FILE="80_llm_score2.py"
# model='qwen' 

model='qwen'
OUTPUT_DIR=./t-test_results_${model}_mmlu/res
preferences=("Expertise_high" "Expertise_low" "Informativeness_high" "Informativeness_low" "Style_high" "Style_low") 
# preferences=("Expertise_high" "Expertise_low" "Style_low" "Style_high" "")
Batch_size=4

INPUT_PATH="<REPO_ROOT>"/qwen/mmlu_test/answer_mmlu_neuron_psoups_10.0/Style_high
echo "[INFO] Input_path: ${INPUT_PATH}"
python $PYTHON_FILE \
        --input_path $INPUT_PATH \
        --principle fluency \
        --batch_size $Batch_size \
        --out_path ${OUTPUT_DIR} 
if [[ "$p" == Expertise_* ]]; then principle="expert"; elif [[ "$p" == Informativeness_* ]]; then principle="informative"; else principle="style"; fi
python $PYTHON_FILE \
        --input_path $INPUT_PATH \
        --principle style \
        --batch_size $Batch_size \
        --out_path ${OUTPUT_DIR} 
# done


export CUDA_VISIBLE_DEVICES=5,7
export NCCL_P2P_DISABLE=1
PYTHON_FILE="80_llm_score.py"
# model='qwen' 

model='qwen'
OUTPUT_DIR=./t-test_results_${model}_mmlu/res
# preferences=("Expertise_high" "Expertise_low" "Informativeness_high" "Informativeness_low" "Style_high" "Style_low") 
preferences=("Expertise" "Informativeness" "Style")
Batch_size=4
for p in "${preferences[@]}"; do
INPUT_PATH="<REPO_ROOT>"/${model}/mmlu_test/answers_deactivate_preference_neurons_mmlu/${p}
echo "[INFO] Input_path: ${INPUT_PATH}"
python $PYTHON_FILE \
        --input_path $INPUT_PATH \
        --principle fluency \
        --batch_size $Batch_size \
        --out_path ${OUTPUT_DIR} 
if [[ "$p" == Expertise_* ]]; then principle="expert"; elif [[ "$p" == Informativeness_* ]]; then principle="informative"; else principle="style"; fi
python $PYTHON_FILE \
        --input_path $INPUT_PATH \
        --principle $principle \
        --batch_size $Batch_size \
        --out_path ${OUTPUT_DIR} 
done

model='llama'
OUTPUT_DIR=./t-test_results_${model}_mmlu/res
preferences=("Expertise_high" "Expertise_low" "Informativeness_high" "Informativeness_low" "Style_high" "Style_low") 
# preferences=("Expertise_high" )
Batch_size=4
for p in "${preferences[@]}"; do
INPUT_PATH="<REPO_ROOT>"/${model}/mmlu_test/answers_deactivate_preference_neurons_mmlu/${p}
echo "[INFO] Input_path: ${INPUT_PATH}"
python $PYTHON_FILE \
        --input_path $INPUT_PATH \
        --principle fluency \
        --batch_size $Batch_size \
        --out_path ${OUTPUT_DIR} 
if [[ "$p" == Expertise_* ]]; then principle="expert"; elif [[ "$p" == Informativeness_* ]]; then principle="informative"; else principle="style"; fi
python $PYTHON_FILE \
        --input_path $INPUT_PATH \
        --principle $principle \
        --batch_size $Batch_size \
        --out_path ${OUTPUT_DIR} 
done

model='gemma'
OUTPUT_DIR=./t-test_results_${model}_mmlu/res
preferences=("Expertise_high" "Expertise_low" "Informativeness_high" "Informativeness_low" "Style_high" "Style_low") 
# preferences=("Expertise_high" )
Batch_size=4
for p in "${preferences[@]}"; do
INPUT_PATH="<REPO_ROOT>"/${model}/mmlu_test/answers_deactivate_preference_neurons_mmlu/${p}
echo "[INFO] Input_path: ${INPUT_PATH}"
python $PYTHON_FILE \
        --input_path $INPUT_PATH \
        --principle fluency \
        --batch_size $Batch_size \
        --out_path ${OUTPUT_DIR} 
if [[ "$p" == Expertise_* ]]; then principle="expert"; elif [[ "$p" == Informativeness_* ]]; then principle="informative"; else principle="style"; fi
python $PYTHON_FILE \
        --input_path $INPUT_PATH \
        --principle $principle \
        --batch_size $Batch_size \
        --out_path ${OUTPUT_DIR} 
done

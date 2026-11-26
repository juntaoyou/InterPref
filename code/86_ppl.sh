export CUDA_VISIBLE_DEVICES=3
# MODEL="<MODELS_DIR>"/Qwen3-8B-Base
MODEL="<MODELS_DIR>"/gemma-2-9b
# MODEL="<MODELS_DIR>"/Llama-3.1-8B
INPUT_PATH="<REPO_ROOT>"/detect_with_two/eval/gemma_alpaca/test
python 86_ppl.py \
    --input_path $INPUT_PATH \
    --model $MODEL


import argparse
from types import MethodType
import torch
import json
import numpy as np
from tqdm import tqdm
import random
import os
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import math
import ast
from datasets import load_from_disk
from lib_data_utils import load_queries  # packaged query pools (JSONL/CSV or save_to_disk)
import sys
sys.path.append('../')
from lib_instructions import instructions  # descriptions variant retired; instruction templates are used
# TEMPLATE="""
# You are given a preference followed by a question. Write a response that reflects the preference as clearly as possible. Don’t worry about correctness or structure. Focus on alignment with the preference.
# ###Preference requirement:{preference}

# ###Question:{question}

# ###Response:
# """
TEMPLATE="""        
You are given two preferences followed by a question. Write a short response (under 300 words) that reflects both preferences as clearly as possible. Don’t worry about correctness or structure. Focus on alignment with the preferences. The preferences should be equally expressed.
###Preference requirement1:{preference1}

###Preference requirement2:{preference2}

###Question:{question}

###Response:
"""
#!!!!!!!!!!!!!!!!
parser = argparse.ArgumentParser()
parser.add_argument( "--model", type=str, required=True, help="Path to the model")  # 模型路径
parser.add_argument( "--output_dir", type=str, required=True, help="Path to output directory")
parser.add_argument( "--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--dataset", type=str)
args = parser.parse_args()

output_dir = args.output_dir
batch_size = args.batch_size
dataset_path = args.dataset
os.makedirs(output_dir,exist_ok=True)
model = LLM(model=args.model, tensor_parallel_size=torch.cuda.device_count(), enforce_eager=True)  # 加载模型
tokenizer = AutoTokenizer.from_pretrained(args.model)  # 加载分词器
num_layers = model.llm_engine.model_config.hf_config.num_hidden_layers


is_llama = is_qwen = is_gemma = False
if "Llama" in args.model:
    is_llama = True

if "gemma" in args.model:
    is_gemma = True

if "Qwen" in args.model:
    is_qwen = True
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#读取要改的神经元
def load_neuron_to_change(neuron_to_change_path):
    with open(neuron_to_change_path, 'r', encoding='utf-8') as file:
        neuron_to_change = json.loads(file.read())
        for t in neuron_to_change:
            neuron_to_change[t]=torch.tensor(neuron_to_change[t]).to(device)
    return neuron_to_change

data = load_queries(dataset_path)
total_len = len(data)
def normal_forward():
    def qwen_forward(self, x):
        x, _ = self.gate_up_proj(x)
        x = self.act_fn(x)
        x, _ = self.down_proj(x)
        return x

    def llama_forward(self, x):
        x, _ = self.gate_up_proj(x)
        x = self.act_fn(x)
        x, _ = self.down_proj(x)
        return x

    if is_llama or is_gemma:
        return llama_forward
    elif is_qwen:
        return qwen_forward


for i in range(num_layers):
    obj = model.llm_engine.model_executor.driver_worker.model_runner.model.model.layers[i].mlp
    obj.forward = MethodType(normal_forward(), obj)

preference_list = [
"Expertise_high.Style_high", "Expertise_high.Style_low", "Expertise_low.Style_low","Expertise_low.Style_high",
"Expertise_low.Informativeness_low", "Expertise_high.Informativeness_high","Expertise_low.Informativeness_high", "Expertise_high.Informativeness_low",
"Informativeness_low.Style_low", "Informativeness_high.Style_low","Informativeness_low.Style_high","Informativeness_high.Style_high",
]

for preference in preference_list:
    preferences = preference.split(".")
    output_dir_bfi=f'{output_dir}/{preference}'
    os.makedirs(output_dir_bfi, exist_ok=True)
    output_file=f'{output_dir_bfi}/{preference}.json'
    with open(output_file, 'w', encoding='utf-8') as output_file:
        for i in tqdm(range(0, total_len, batch_size)):
            num_of_words=0
            end_idx = min(total_len, i + batch_size)
            batch_questions = data[i: end_idx]
            input_texts = []
            for question in batch_questions:
                input_text =TEMPLATE.format(preference1=instructions[preferences[0]],
                                            preference2=instructions[preferences[1]],
                                            question=question)
                input_texts.append([{"role": "user", "content": input_text}])
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            input_ids = tokenizer.apply_chat_template(input_texts, add_generation_prompt=True, tokenize=False, enable_thinking=False)
            sampling_params = SamplingParams(
                max_tokens=2048,
                temperature=0.0,
                repetition_penalty=1.15
            )
            output = model.generate(input_ids, sampling_params=sampling_params)
            json_lines = [json.dumps({"question": batch_questions[i], "answer": output[i].outputs[0].text}) + '\n' 
                for i in range(len(batch_questions))]
            output_file.writelines(json_lines)

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
import datetime
date_string = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

TEMPLATE="""
You are given a preference followed by a question. Write a short response (under 300 words) that reflects the preference as clearly as possible. Don't worry about correctness or structure. Focus on alignment with the preference.
###Preference requirement:None

###Question:{question}

###Response:
"""

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="Path to the model")  # 模型路径
parser.add_argument("--neuron_dir", type=str, required=True, help="Path to neuron results directory")
parser.add_argument("--output_dir", type=str, required=True, help="Path to output directory")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--dataset", type=str, required=True, help="Path to Dataset")
parser.add_argument("--delta", type=float, default=6.0, help="Number of generations per question")
args = parser.parse_args()
neuron_dir = args.neuron_dir
output_dir = args.output_dir
batch_size = args.batch_size
dataset_path = args.dataset
delta = args.delta

is_llama = is_qwen = is_gemma = False
if "Llama" in args.model:
    is_llama = True

if "gemma" in args.model:
    is_gemma = True

if "Qwen" in args.model:
    is_qwen = True

os.makedirs(output_dir,exist_ok=True)
model = LLM(model=args.model, 
            tensor_parallel_size=torch.cuda.device_count(), 
            enforce_eager=True,
            gpu_memory_utilization=0.8,
            )  # 加载模型
tokenizer = AutoTokenizer.from_pretrained(args.model)  # 加载分词器
num_layers = model.llm_engine.model_config.hf_config.num_hidden_layers

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_neuron_to_change(neuron_to_change_path):
    with open(neuron_to_change_path, 'r', encoding='utf-8') as file:
        neuron_to_change = json.loads(file.read())
        for t in neuron_to_change:
            neuron_to_change[t]=torch.tensor(neuron_to_change[t]).to(device)
    return neuron_to_change

def deduplicate_tensors_by_dim(T1, T2):
    mask1 = ~torch.isin(T1[:, 1], T2[:,1])
    mask2 = ~torch.isin(T2[:, 1], T1[:,1])
    T1_final = T1[mask1]
    T2_final = T2[mask2]
    return T1_final,T2_final
            

def factory(idx): 
    def qwen_forward(self, x): 
        gate_up, _ = self.gate_up_proj(x) 
        i = gate_up.size(-1)
        j=x.shape[0]
        gate_up[:, : i // 2] = torch.nn.SiLU()(gate_up[:, : i // 2])
        if j <= batch_size and str(idx) in neuron_to_change:
            elements = neuron_to_change[str(idx)]
            indices = elements[:, 1].long()
            # Amplification : the selected preference-associated neurons are
            # scaled up under the matched neutral prompt: a_j -> a_j * (1 + val),
            # with no sigmoid gating, quantile weighting or random dropout
            # .
            gate_up[:, indices] = gate_up[:, indices] * (1.0 + val)
        x = gate_up[:, : i // 2] * gate_up[:, i // 2:]
        x, _ = self.down_proj(x)
        return x

    def llama_forward(self, x):
        gate_up, _ = self.gate_up_proj(x) 
        i = gate_up.size(-1)
        j=x.shape[0]
        gate_up[:, : i // 2] = torch.nn.SiLU()(gate_up[:, : i // 2])
        if j <= batch_size and str(idx) in neuron_to_change:
            elements = neuron_to_change[str(idx)]
            indices = elements[:, 1].long()
            # Amplification : the selected preference-associated neurons are
            # scaled up under the matched neutral prompt: a_j -> a_j * (1 + val),
            # with no sigmoid gating, quantile weighting or random dropout
            # .
            gate_up[:, indices] = gate_up[:, indices] * (1.0 + val)
        x = gate_up[:, : i // 2] * gate_up[:, i // 2:]
        x, _ = self.down_proj(x)
        return x
    
    def gemma_forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        i = gate_up.size(-1)
        j=x.shape[0]
        gate_up[:, : i // 2] = torch.nn.GELU()(gate_up[:, : i // 2])
        if j <= batch_size and str(idx) in neuron_to_change:
            elements = neuron_to_change[str(idx)]
            indices = elements[:, 1].long()
            # Amplification : the selected preference-associated neurons are
            # scaled up under the matched neutral prompt: a_j -> a_j * (1 + val),
            # with no sigmoid gating, quantile weighting or random dropout
            # .
            gate_up[:, indices] = gate_up[:, indices] * (1.0 + val)
        x = gate_up[:, : i // 2] * gate_up[:, i // 2:]
        x, _ = self.down_proj(x)
        return x

    if is_llama:
        return llama_forward
    elif is_gemma:
        return gemma_forward
    elif is_qwen:
        return qwen_forward
    
data = load_queries(dataset_path)
total_len = len(data)
def normal_forward():
    def qwen_forward(self,x):
        x, _ = self.gate_up_proj(x)
        x = self.act_fn(x)
        x, _ = self.down_proj(x)
        return x
    
    def llama_forward(self,x):
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


for i in range(num_layers):
    obj = model.llm_engine.model_executor.driver_worker.model_runner.model.model.layers[i].mlp
    obj.forward = MethodType(factory(i), obj)

# delta = 10.0
interval = {"Informativeness_low": [14, 17],
            "Informativeness_high":[61, 65],
            "Expertise_high":[51, 55],
            "Expertise_low":[11, 15],
            "Style_low":[26, 30], 
            "Style_high":[18, 22]}
preference_list = ["Expertise_high", "Expertise_low", "Style_high", "Style_low"]
# preference_list = ["Informativeness_low", "Informativeness_high", "Expertise_high", "Expertise_low", "Style_high", "Style_low"]

for preference in preference_list:
    neuron_to_change_path=f'{neuron_dir}/{preference}_{delta}.json'
    if 'low' in preference:
        reverse_preference = preference.replace('low', 'high')
    elif 'high' in preference:
         reverse_preference = preference.replace('high', 'low')
    neuron_to_deactivate_path=f'{neuron_dir}/{reverse_preference}_{delta}.json'
    neuron_to_change=load_neuron_to_change(neuron_to_change_path)
    neuron_to_deactivate=load_neuron_to_change(neuron_to_deactivate_path)
    output_dir_bfi=f'{output_dir}_{delta}/{preference}'
    os.makedirs(output_dir_bfi, exist_ok=True)
    ratios = list(map(lambda x: round(x * 0.1,1), list(range(interval[preference][0],interval[preference][1]))))
    
    for val in ratios:
        output_file=f'{output_dir_bfi}/{preference}_{val}.json'
        with open(output_file, 'w', encoding='utf-8') as output_file:
            for i in tqdm(range(0, total_len, batch_size)):
                end_idx = min(total_len, i + batch_size)
                batch_questions = data[i: end_idx]
                input_texts = []
                for question in batch_questions:
                    input_text = TEMPLATE.format(question=question)
                    input_texts.append([{"role": "user", "content": input_text}])
                
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                input_ids = tokenizer.apply_chat_template(input_texts, add_generation_prompt=True, tokenize=False, enable_thinking=False)
                
                sampling_params = SamplingParams(
                    max_tokens=400,
                    temperature=0.0,
                    repetition_penalty=1.15,
                )
                
                output = model.generate(input_ids, sampling_params=sampling_params)
                json_lines = [json.dumps({"question": batch_questions[i], "answer": output[i].outputs[0].text}) + '\n' 
                    for i in range(len(batch_questions))]
                output_file.writelines(json_lines)

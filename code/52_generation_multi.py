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
from lib_instructions import instructions

TEMPLATE="""           
You are given a preference followed by a question. Write a short response (under 300 words) that reflects the preference as clearly as possible. Don’t worry about correctness or structure. Focus on alignment with preference.
###Preference requirement:None

###Question:{question}

###Response:
"""

parser = argparse.ArgumentParser()
parser.add_argument( "--model", type=str, required=True, help="Path to the model")
parser.add_argument( "--neuron_dir", type=str, required=True, help="Path to neuron results directory")
parser.add_argument( "--neuron_multi_dir", type=str, help="Path to neuron results directory")
parser.add_argument( "--output_dir", type=str, required=True, help="Path to output directory")
parser.add_argument( "--batch_size", type=int, default=32, help="Batch size")
parser.add_argument( "--dataset", type=str, help="Path to dataset")
parser.add_argument( "--percent", type=float, default=32.0)
parser.add_argument( "--mode", type=str, default="union")
parser.add_argument( "--config", type=str, 
                    help="JSON string for interval config, e.g. '{\"Expertise_low.Style_high\": [20, 27], ...}'")
parser.add_argument( "--config_file", type=str, 
                    help="JSON file path containing interval config")
args = parser.parse_args()

# Parse interval config
if args.config:
    config = json.loads(args.config)
elif args.config_file:
    with open(args.config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
else:
    raise ValueError("Either --config or --config_file must be provided")

interval = {k: v for k, v in config.items() if isinstance(v, list)}
percent = config.get('percent', args.percent)

neuron_dir = args.neuron_dir
output_dir = args.output_dir
batch_size = args.batch_size
neuron_multi_dir = args.neuron_multi_dir
mode = args.mode


os.makedirs(args.output_dir, exist_ok=True)
model = LLM(model=args.model,
            tensor_parallel_size=torch.cuda.device_count(), 
            enforce_eager=True,
            gpu_memory_utilization=0.8)
tokenizer = AutoTokenizer.from_pretrained(args.model)
num_layers = model.llm_engine.model_config.hf_config.num_hidden_layers

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
is_llama = is_qwen = is_gemma = False
if "Llama" in args.model:
    is_llama = True

if "gemma" in args.model:
    is_gemma = True

if "Qwen" in args.model:
    is_qwen = True

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


def merge(neuron1, neuron2):
    neuron = {}
    for layer in range(num_layers):
        layer_str = str(layer)
        
        if layer_str in neuron1 and layer_str in neuron2:
            concatenated = torch.cat([neuron1[layer_str], neuron2[layer_str]], dim=0)
            
            unique_neuron_indices, inverse_indices = torch.unique(concatenated[:, 1].long(), return_inverse=True, sorted=False)
            
            unique_tensors = []
            for i in range(len(unique_neuron_indices)):
                mask = inverse_indices == i
                if torch.any(mask):
                    first_index = torch.where(mask)[0][0]
                    unique_tensors.append(concatenated[first_index])
            
            if unique_tensors:
                merged_tensor = torch.stack(unique_tensors)
            else:
                merged_tensor = torch.empty((0, concatenated.shape[1])).to(device)
        elif layer_str in neuron1:
                merged_tensor = neuron1[layer_str]
        elif layer_str in neuron2:
                merged_tensor = neuron2[layer_str]
        neuron[layer_str] = merged_tensor
    return neuron
    
def intersect(neuron1, neuron2):
    neuron = {}
    for layer in range(num_layers):
        layer_str = str(layer)
        
        if layer_str in neuron1 and layer_str in neuron2:
            t1 = neuron1[layer_str]
            t2 = neuron2[layer_str]
            
            if t1.numel() == 0 or t2.numel() == 0:
                neuron[layer_str] = torch.empty((0, t1.shape[1] if t1.numel() > 0 else t2.shape[1])).to(device)
                continue
            neuron_indices1 = t1[:, 1].long()
            neuron_indices2 = t2[:, 1].long()
            mask_in_t2 = torch.isin(neuron_indices1, neuron_indices2)
            
            if torch.any(mask_in_t2):
                intersection_tensor = t1[mask_in_t2]
                neuron[layer_str] = intersection_tensor    
    return neuron

def myPartial(neuron1, neuron2):
    neuron = {}
    for layer in range(num_layers):
        layer_str = str(layer)

        if layer_str in neuron1:
            t1 = neuron1[layer_str]
            if layer_str in neuron2:
                t2 = neuron2[layer_str]
                if t1.numel() == 0:
                    neuron[layer_str] = torch.empty((0, t1.shape[1])).to(device) if t1.shape[1] > 0 else torch.empty((0, 2)).to(device)
                    continue
                if t2.numel() == 0:
                    neuron[layer_str] = t1
                    continue
                neuron_indices1 = t1[:, 1].long()
                neuron_indices2 = t2[:, 1].long()
                mask_not_in_t2 = ~torch.isin(neuron_indices1, neuron_indices2)

                if torch.any(mask_not_in_t2):
                    difference_tensor = t1[mask_not_in_t2]
                    neuron[layer_str] = difference_tensor
            else:
                neuron[layer_str] = t1
    return neuron  

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
    
data = load_queries(args.dataset)
total_len = len(data)

def normal_forward():
    def forward(self,x):
        x, _ = self.gate_up_proj(x)
        x = self.act_fn(x)
        x, _ = self.down_proj(x)
        return x

    return forward

for i in range(num_layers):
    obj = model.llm_engine.model_executor.driver_worker.model_runner.model.model.layers[i].mlp
    obj.forward = MethodType(normal_forward(), obj)


for i in range(num_layers):
    obj = model.llm_engine.model_executor.driver_worker.model_runner.model.model.layers[i].mlp ##
    obj.forward = MethodType(factory(i), obj)

# Get preference list from interval config keys
preference_list = list(interval.keys())

def load_union_neuron(preferences):
    if len(preferences) == 1:
        # Single attribute mode
        neuron_to_change_path = f'{neuron_dir}/{preferences[0]}_{percent}.json'
        if 'low' in preferences[0]:
            reverse_p = preferences[0].replace("low","high")
        elif 'high' in preferences[0]:
            reverse_p = preferences[0].replace("high","low")
        neuron_to_change = load_neuron_to_change(neuron_to_change_path)
        neuron_to_deactivate_path = f'{neuron_dir}/{reverse_p}_{percent}.json'
        neuron_to_deactivate = load_neuron_to_change(neuron_to_deactivate_path)
        return neuron_to_change, neuron_to_deactivate

    neuron_to_change_path1 = f'{neuron_dir}/{preferences[0]}_{percent}.json'
    neuron_to_change_path2 = f'{neuron_dir}/{preferences[1]}_{percent}.json'
    if 'low' in preferences[0]:
        reverse_p0 =  preferences[0].replace("low","high")
    elif 'high' in preferences[0]:
        reverse_p0 =  preferences[0].replace("high","low")

    if 'low' in preferences[1]:
        reverse_p1 =  preferences[1].replace("low","high")
    elif 'high' in preferences[1]:
        reverse_p1 =  preferences[1].replace("high","low")
    neuron_to_change1 = load_neuron_to_change(neuron_to_change_path1)
    neuron_to_change2 = load_neuron_to_change(neuron_to_change_path2)

    neuron_to_deactivate_path1 = f'{neuron_dir}/{reverse_p0}_{percent}.json'
    neuron_to_deactivate_path2 = f'{neuron_dir}/{reverse_p1}_{percent}.json'
    neuron_to_deactivate1 = load_neuron_to_change(neuron_to_deactivate_path1)
    neuron_to_deactivate2 = load_neuron_to_change(neuron_to_deactivate_path2)
    neuron_to_change = merge(neuron_to_change1, neuron_to_change2)
    neuron_to_deactivate = merge(neuron_to_deactivate1, neuron_to_deactivate2)

    return neuron_to_change, neuron_to_deactivate

def load_both_neuron(preference):
    neuron_to_change_path = f'{neuron_multi_dir}/{preference}_{percent}.json'
    if len(preferences) == 1:
        # Single attribute: use single neuron dir (multi dir doesn't have single attributes)
        neuron_to_change_path = f'{neuron_dir}/{preferences[0]}_{percent}.json'
        if 'low' in preferences[0]:
            reverse_p = preferences[0].replace("low","high")
        elif 'high' in preferences[0]:
            reverse_p = preferences[0].replace("high","low")
        neuron_to_change = load_neuron_to_change(neuron_to_change_path)
        neuron_to_deactivate_path = f'{neuron_dir}/{reverse_p}_{percent}.json'
        neuron_to_deactivate = load_neuron_to_change(neuron_to_deactivate_path)
        return neuron_to_change, neuron_to_deactivate

    if 'low' in preferences[0]:
        reverse_p0 =  preferences[0].replace("low","high")
    elif 'high' in preferences[0]:
        reverse_p0 =  preferences[0].replace("high","low")

    if 'low' in preferences[1]:
        reverse_p1 =  preferences[1].replace("low","high")
    elif 'high' in preferences[1]:
        reverse_p1 =  preferences[1].replace("high","low")
    reverse_p = reverse_p0 + "." + reverse_p1
    neuron_to_deactivate_path = f'{neuron_multi_dir}/{reverse_p}_{percent}.json'
    neuron_to_change = load_neuron_to_change(neuron_to_change_path)
    neuron_to_deactivate = load_neuron_to_change(neuron_to_deactivate_path)
    return neuron_to_change, neuron_to_deactivate


for preference in preference_list:
    preferences = preference.split(".")
    ratios = list(map(lambda x: round(x * 0.1,1), list(range(interval[preference][0],interval[preference][1])))) 
    for val in ratios:
        if mode == 'union':
            neuron_to_change, neuron_to_deactivate = load_union_neuron(preferences)
        elif mode == 'both':
            neuron_to_change, neuron_to_deactivate = load_both_neuron(preference)
        elif mode == 'overlap':
            neuron_to_change1, neuron_to_deactivate1 = load_union_neuron(preferences)
            neuron_to_change2, neuron_to_deactivate2 = load_both_neuron(preference)
            neuron_to_change = intersect(neuron_to_change1, neuron_to_change2)
            neuron_to_deactivate = intersect(neuron_to_deactivate1, neuron_to_deactivate2)
        elif mode == 'partial1' or mode == 'partial2':
            neuron_to_change_path1 = f'{neuron_dir}/{preferences[0]}_{percent}.json'
            neuron_to_change_path2 = f'{neuron_dir}/{preferences[1]}_{percent}.json'
            reversed_p1 = preferences[0].replace("low","high") if 'low' in preferences[0] else preferences[0].replace("high","low")
            reversed_p2 = preferences[1].replace("low","high") if 'low' in preferences[1] else preferences[1].replace("high","low")
            neuron_to_deactivate_path1 = f'{neuron_dir}/{reversed_p1}_{percent}.json'
            neuron_to_deactivate_path2 = f'{neuron_dir}/{reversed_p2}_{percent}.json'
            if mode == 'partial1':
                neuron_to_change = myPartial(load_neuron_to_change(neuron_to_change_path1), load_neuron_to_change(neuron_to_change_path2))
                neuron_to_deactivate = myPartial(load_neuron_to_change(neuron_to_deactivate_path1), load_neuron_to_change(neuron_to_deactivate_path2))
            elif mode == 'partial2':                                                                                                    
                neuron_to_change = myPartial(load_neuron_to_change(neuron_to_change_path2), load_neuron_to_change(neuron_to_change_path1))
                neuron_to_deactivate = myPartial(load_neuron_to_change(neuron_to_deactivate_path2), load_neuron_to_change(neuron_to_deactivate_path1))
                
        output_dir_bfi=f'{output_dir}/{preference}'
        os.makedirs(output_dir_bfi, exist_ok=True)
        output_file=f'{output_dir_bfi}/{preference}_{val}.json'
        with open(output_file, 'w', encoding='utf-8') as output_file:
            for i in tqdm(range(0, total_len, batch_size)):
                num_of_words=0
                end_idx = min(total_len, i + batch_size)
                batch_questions = data[i: end_idx]
                input_texts = []
                for question in batch_questions:
                    input_text =TEMPLATE.format(question=question)
                    input_texts.append([{"role": "user", "content": input_text}])
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                input_ids = tokenizer.apply_chat_template(input_texts, add_generation_prompt=True, tokenize=False, enable_thinking=False)
                sampling_params = SamplingParams(
                    max_tokens=400,
                    temperature=0.0,
                    repetition_penalty=1.15
                )
                output = model.generate(input_ids, sampling_params=sampling_params)
                json_lines = [json.dumps({"question": batch_questions[i], "answer": output[i].outputs[0].text}) + '\n' 
                    for i in range(len(batch_questions))]
                output_file.writelines(json_lines)

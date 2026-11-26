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
import datetime
date_string = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
TEMPLATE="""        
You are given two preferences followed by a question. Write a short response (under 300 words) that reflects both preferences as clearly as possible. Don't worry about correctness or structure. Focus on alignment with the preferences. The preferences should be equally expressed.
###Preference requirement1:{preference1}

###Preference requirement2:{preference2}

###Question:{question}

###Response:
"""
parser = argparse.ArgumentParser()
parser.add_argument( "--model", type=str, required=True, help="Path to the model")
parser.add_argument( "--neuron_dir", type=str, required=True, help="Path to neuron results directory")
parser.add_argument( "--neuron_dir_multi", type=str, required=True, help="Path to neuron results directory")
parser.add_argument( "--output_dir", type=str, required=True, help="Path to output directory")
parser.add_argument( "--batch_size", type=int, default=32, help="Batch size")
parser.add_argument( "--deactivate", type=str, default="preference_neurons", help="Way to deactivate")
parser.add_argument( "--mode", type=str, default="union", help="Union or Two")
parser.add_argument( "--dataset", type=str)
parser.add_argument( "--config", type=str, help="JSON config file with preference:percent mappings")
parser.add_argument( "--config_file", type=str, 
                    help="JSON file path containing interval config")
args = parser.parse_args()

# Load config from JSON file
if args.config:
    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)
    preference_list = list(config.keys())
elif args.config_file:
    with open(args.config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    preference_list = list(config.keys())
else:
    raise ValueError("Either --config or --config_file must be provided")

neuron_dir = args.neuron_dir
output_dir = args.output_dir
batch_size = args.batch_size
neuron_dir_multi = args.neuron_dir_multi
mode = args.mode
os.makedirs(args.output_dir, exist_ok=True)

import logging
os.makedirs("./logs", exist_ok=True)
logging.basicConfig(filename=f"./logs/log_{date_string}.txt", filemode="w", format="%(asctime)s %(name)s:%(levelname)s:%(message)s", datefmt="%d-%M-%Y %H:%M:%S", level=logging.DEBUG)
logging.info(f"File: {os.path.basename(__file__)}")
logging.info(f'Model: {args.model}')
logging.info(f'Dataset: {args.dataset}')
logging.info(f'Mode: {mode}')
logging.info(f'Config: {config}')


is_llama = is_qwen = is_gemma = False
if "Llama" in args.model:
    is_llama = True

if "gemma" in args.model:
    is_gemma = True

if "Qwen" in args.model:
    is_qwen = True
model = LLM(model=args.model, 
            tensor_parallel_size=torch.cuda.device_count(), 
            enforce_eager=True, 
            gpu_memory_utilization=0.8,
            
)  # 加载模型
tokenizer = AutoTokenizer.from_pretrained(args.model)  # 加载分词器
num_layers = model.llm_engine.model_config.hf_config.num_hidden_layers
intermediate_size = model.llm_engine.model_config.hf_config.intermediate_size
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_neuron_to_change(neuron_to_change_path):
    with open(neuron_to_change_path, 'r', encoding='utf-8') as file:
        neuron_to_change = json.loads(file.read())
        for t in neuron_to_change:
            neuron_to_change[t]=torch.tensor(neuron_to_change[t]).to(device)
    return neuron_to_change

def merge(e1, e2):
    s1,s2 = e1.tolist(),e2.tolist()
    s = list(set(s1) | set(s2))
    return torch.tensor(s).long()

def intersect(e1, e2):
    s1,s2 = e1.tolist(),e2.tolist()
    s = list(set(s1) & set(s2))
    return torch.tensor(s).long()

def myPartial(e1, e2):
    s1,s2 = e1.tolist(),e2.tolist()
    s = list(set(s1) - set(s2))
    return torch.tensor(s).long()

def factory(idx): 
    def forward(self, x): 
        gate_up, _ = self.gate_up_proj(x)
        i = gate_up.size(-1)
        j=x.shape[0]
        if is_llama or is_qwen:
            gate_up[:, : i // 2] = torch.nn.SiLU()(gate_up[:, : i // 2])
        elif is_gemma:
            gate_up[:, : i // 2] = torch.nn.GELU()(gate_up[:, : i // 2])
        if mode == 'both':
            if j <= batch_size and str(idx) in neuron_to_deactivate:
                elements_to_deactivate=neuron_to_deactivate[str(idx)]
                indices_to_deactivate=elements_to_deactivate[:, 1].long().to(x.device) 
                if args.deactivate == 'preference_neurons':
                    gate_up[:, indices_to_deactivate] = 0.0  # Deactivation : the selected neurons are clamped to zero
                elif args.deactivate == 'random':
                    neuron_num = len(indices_to_deactivate)
                    cur_mask = torch.randperm(intermediate_size)[:neuron_num].to('cuda')
                    gate_up[:, cur_mask] = 0.0  # Random-neuron control: matched-size random neurons clamped to zero
                    
        elif mode == 'union':
            if j <= batch_size and str(idx) in neuron_to_deactivate1 and str(idx) in neuron_to_deactivate2:
                elements_to_deactivate1=neuron_to_deactivate1[str(idx)]
                indices_to_deactivate1=elements_to_deactivate1[:, 1].long().to(x.device) 
                elements_to_deactivate2=neuron_to_deactivate2[str(idx)]
                indices_to_deactivate2=elements_to_deactivate2[:, 1].long().to(x.device) 
                indices_to_deactivate = merge(indices_to_deactivate1, indices_to_deactivate2)
            elif j <= batch_size and str(idx) in neuron_to_deactivate1:
                elements_to_deactivate1=neuron_to_deactivate1[str(idx)]
                indices_to_deactivate=elements_to_deactivate1[:, 1].long().to(x.device) 
            elif j <= batch_size and str(idx) in neuron_to_deactivate2:
                elements_to_deactivate2=neuron_to_deactivate2[str(idx)]
                indices_to_deactivate=elements_to_deactivate2[:, 1].long().to(x.device) 
            else:
                x = gate_up[:, : i // 2] * gate_up[:, i // 2:]
                x, _ = self.down_proj(x)
                return x
              
            if args.deactivate == 'preference_neurons':
                gate_up[:, indices_to_deactivate] = 0.0  # Deactivation : the selected neurons are clamped to zero
            elif args.deactivate == 'random':
                neuron_num = len(indices_to_deactivate)
                cur_mask = torch.randperm(intermediate_size)[:neuron_num].to('cuda')
                gate_up[:, cur_mask] = 0.0  # Random-neuron control: matched-size random neurons clamped to zero
                
        elif mode == 'intersect':
            if j <= batch_size and str(idx) in neuron_to_deactivate1 and str(idx) in neuron_to_deactivate2:
                elements_to_deactivate1=neuron_to_deactivate1[str(idx)]
                indices_to_deactivate1=elements_to_deactivate1[:, 1].long().to(x.device) 
                elements_to_deactivate2=neuron_to_deactivate2[str(idx)]
                indices_to_deactivate2=elements_to_deactivate2[:, 1].long().to(x.device) 
                indices_to_deactivate = intersect(indices_to_deactivate1, indices_to_deactivate2)
                
                if args.deactivate == 'preference_neurons':
                    gate_up[:, indices_to_deactivate] = 0.0  # Deactivation : the selected neurons are clamped to zero
                
                elif args.deactivate == 'random':
                    neuron_num = len(indices_to_deactivate)
                    cur_mask = torch.randperm(intermediate_size)[:neuron_num].to('cuda')
                    gate_up[:, cur_mask] = 0.0  # Random-neuron control: matched-size random neurons clamped to zero
                    
        elif mode == "partial1":
            if j <= batch_size and str(idx) in neuron_to_deactivate1 and str(idx) in neuron_to_deactivate2:
                elements_to_deactivate1=neuron_to_deactivate1[str(idx)]
                indices_to_deactivate1=elements_to_deactivate1[:, 1].long()
                elements_to_deactivate2=neuron_to_deactivate2[str(idx)]
                indices_to_deactivate2=elements_to_deactivate2[:, 1].long()
                indices_to_deactivate = myPartial(indices_to_deactivate1, indices_to_deactivate2)
            elif j <= batch_size and str(idx) in neuron_to_deactivate1:
                elements_to_deactivate1=neuron_to_deactivate1[str(idx)]
                indices_to_deactivate=elements_to_deactivate1[:, 1].long()
            elif j <= batch_size and str(idx) in neuron_to_deactivate2:
                elements_to_deactivate2=neuron_to_deactivate2[str(idx)]
                indices_to_deactivate=elements_to_deactivate2[:, 1].long()
            else:
                x = gate_up[:, : i // 2] * gate_up[:, i // 2:]
                x, _ = self.down_proj(x)
                return x
                
                
            if args.deactivate == 'preference_neurons':
                gate_up[:, indices_to_deactivate] = 0.0  # Deactivation : the selected neurons are clamped to zero
            
            elif args.deactivate == 'random':
                neuron_num = len(indices_to_deactivate)
                cur_mask = torch.randperm(intermediate_size)[:neuron_num].to('cuda')
                gate_up[:, cur_mask] = 0.0  # Random-neuron control: matched-size random neurons clamped to zero
                    
        elif mode == "partial2":
            if j <= batch_size and str(idx) in neuron_to_deactivate1 and str(idx) in neuron_to_deactivate2:
                elements_to_deactivate1=neuron_to_deactivate1[str(idx)]
                indices_to_deactivate1=elements_to_deactivate1[:, 1].long()
                elements_to_deactivate2=neuron_to_deactivate2[str(idx)]
                indices_to_deactivate2=elements_to_deactivate2[:, 1].long()
                indices_to_deactivate = myPartial(indices_to_deactivate2, indices_to_deactivate1)
                
                if args.deactivate == 'preference_neurons':
                    gate_up[:, indices_to_deactivate] = 0.0  # Deactivation : the selected neurons are clamped to zero
                
                if args.deactivate == 'random':
                    neuron_num = len(indices_to_deactivate)
                    cur_mask = torch.randperm(intermediate_size)[:neuron_num].to('cuda')
                    gate_up[:, cur_mask] = 0.0  # Random-neuron control: matched-size random neurons clamped to zero
            
            
        elif mode == 'first':
            if j <= batch_size and str(idx) in neuron_to_deactivate1:
                elements_to_deactivate1=neuron_to_deactivate1[str(idx)]
                indices_to_deactivate1=elements_to_deactivate1[:, 1].long()
                if args.deactivate == 'preference_neurons':
                    gate_up[:, indices_to_deactivate1] = 0.0  # Deactivation : clamped to zero
                elif args.deactivate == 'random':
                    neuron_num = len(indices_to_deactivate1)
                    cur_mask = torch.randperm(intermediate_size)[:neuron_num].to('cuda')
                    gate_up[:, cur_mask] = 0.0  # Random-neuron control: matched-size random neurons clamped to zero

        elif mode == 'second':
            if j <= batch_size and str(idx) in neuron_to_deactivate2:
                elements_to_deactivate2=neuron_to_deactivate2[str(idx)]
                indices_to_deactivate2=elements_to_deactivate2[:, 1].long()
                if args.deactivate == 'preference_neurons':
                    gate_up[:, indices_to_deactivate2] = 0.0  # Deactivation : clamped to zero
                elif args.deactivate == 'random':
                    neuron_num = len(indices_to_deactivate2)
                    cur_mask = torch.randperm(intermediate_size)[:neuron_num].to('cuda')
                    gate_up[:, cur_mask] = 0.0  # Random-neuron control: matched-size random neurons clamped to zero
                    
        elif mode == 'overlap':
            if j <= batch_size and str(idx) in neuron_to_deactivate:
                elements_to_deactivate=neuron_to_deactivate[str(idx)]
                indices_to_deactivate_main=elements_to_deactivate[:, 1].long()
                
                union_indices = []
                if str(idx) in neuron_to_deactivate1:
                    elements_to_deactivate1=neuron_to_deactivate1[str(idx)]
                    union_indices.extend(elements_to_deactivate1[:, 1].long().tolist())
                if str(idx) in neuron_to_deactivate2:
                    elements_to_deactivate2=neuron_to_deactivate2[str(idx)]
                    union_indices.extend(elements_to_deactivate2[:, 1].long().tolist())
                
                union_indices = list(set(union_indices))
                
                if union_indices:
                    union_indices = torch.tensor(union_indices).long()
                    
                    indices_to_deactivate = intersect(union_indices, indices_to_deactivate_main)
                    
                    if len(indices_to_deactivate) > 0:
                        if args.deactivate == 'preference_neurons':
                            gate_up[:, indices_to_deactivate] = 0.0  # Deactivation : the selected neurons are clamped to zero
                        elif args.deactivate == 'random':
                            neuron_num = len(indices_to_deactivate)
                            cur_mask = torch.randperm(intermediate_size)[:neuron_num].to('cuda')
                            gate_up[:, cur_mask] = 0.0  # Random-neuron control: matched-size random neurons clamped to zero
        x = gate_up[:, : i // 2] * gate_up[:, i // 2:]
        x, _ = self.down_proj(x)
        return x
    return forward


data = load_queries(args.dataset)
total_len = len(data)
for i in range(num_layers):
    obj = model.llm_engine.model_executor.driver_worker.model_runner.model.model.layers[i].mlp
    obj.forward = MethodType(factory(i), obj)

save_name = {
"Expertise_high.Style_high":"Expertise.Style", 
"Expertise_high.Style_low":"Expertise.Style", 
"Expertise_low.Style_low":"Expertise.Style", 
"Expertise_low.Style_high":"Expertise.Style", 
"Expertise_low.Informativeness_low":"Expertise.Informativeness", 
"Expertise_high.Informativeness_high":"Expertise.Informativeness", 
"Expertise_high.Informativeness_low":"Expertise.Informativeness", 
"Expertise_low.Informativeness_high":"Expertise.Informativeness", 
"Informativeness_low.Style_low":"Informativeness.Style", 
"Informativeness_high.Style_low":"Informativeness.Style", 
"Informativeness_low.Style_high":"Informativeness.Style", 
"Informativeness_high.Style_high":"Informativeness.Style"
}

for preference in preference_list:
    percent = config[preference]
    preferences = preference.split(".")
    if mode == 'both':
        neuron_to_deactivate_path=f'{neuron_dir_multi}/{preference}_{percent}.json'
        neuron_to_deactivate=load_neuron_to_change(neuron_to_deactivate_path)
    elif mode == 'union' or mode == "intersect" or mode == "partial1" or mode == "partial2":
        neuron_to_deactivate_path1=f'{neuron_dir}/{preferences[0]}_{percent}.json'
        neuron_to_deactivate1=load_neuron_to_change(neuron_to_deactivate_path1)
        
        neuron_to_deactivate_path2=f'{neuron_dir}/{preferences[1]}_{percent}.json'
        neuron_to_deactivate2=load_neuron_to_change(neuron_to_deactivate_path2)
    elif mode == 'first':
        neuron_to_deactivate_path1=f'{neuron_dir}/{preferences[0]}_{percent}.json'  
        neuron_to_deactivate1=load_neuron_to_change(neuron_to_deactivate_path1)

    elif mode == 'second':
        neuron_to_deactivate_path2=f'{neuron_dir}/{preferences[1]}_{percent}.json'          
        neuron_to_deactivate2=load_neuron_to_change(neuron_to_deactivate_path2)
    elif mode == 'overlap':
        neuron_to_deactivate_path=f'{neuron_dir_multi}/{preference}_{percent}.json'
        neuron_to_deactivate=load_neuron_to_change(neuron_to_deactivate_path)
        
        neuron_to_deactivate_path1=f'{neuron_dir}/{preferences[0]}_{percent}.json'
        neuron_to_deactivate1=load_neuron_to_change(neuron_to_deactivate_path1)

        neuron_to_deactivate_path2=f'{neuron_dir}/{preferences[1]}_top_{percent}percent.json'
        neuron_to_deactivate2=load_neuron_to_change(neuron_to_deactivate_path2)
    output_dir_bfi=f'{output_dir}/{save_name[preference]}'
    os.makedirs(output_dir_bfi, exist_ok=True)
    output_file=f'{output_dir_bfi}/{preference}_{percent}.json'
    
    with open(output_file, 'w', encoding='utf-8') as output_file:
        for i in tqdm(range(0, total_len, batch_size)):
            num_of_words = 0
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
                max_tokens=400,
                temperature=0.0,
                repetition_penalty=1.15
            )
            output = model.generate(input_ids, sampling_params=sampling_params)
            json_lines = [json.dumps({"question": batch_questions[i], "answer": output[i].outputs[0].text}) + '\n' 
                for i in range(len(batch_questions))]
            output_file.writelines(json_lines)


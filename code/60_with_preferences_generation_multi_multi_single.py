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


PREFERENCE_TEMPLATE="""        
You are given two preferences followed by a question. Write a short response (under 300 words) that reflects both preferences as clearly as possible. Don’t worry about correctness or structure. Focus on alignment with the preferences. The preferences should be equally expressed.
###Preference requirement1:{preference1}

###Preference requirement2:{preference2}

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
parser.add_argument( "--max_model_len", type=int, default=None,
                    help="Override vLLM max_model_len. If unset, auto-computed from data: "
                         "max input tokens + 400 (output) + 128 margin.")
parser.add_argument( "--no_deactivate", action="store_true",
                    help="Steer only (neuron_to_change), skip the deactivation set (neuron_to_deactivate).")
parser.add_argument( "--no_steer", action="store_true",
                    help="Deactivate only, skip the steer set (neuron_to_change = {}).")
args = parser.parse_args()

# Parse interval config
if args.config:
    interval = json.loads(args.config)
elif args.config_file:
    with open(args.config_file, 'r', encoding='utf-8') as f:
        interval = json.load(f)
else:
    raise ValueError("Either --config or --config_file must be provided")

neuron_dir = args.neuron_dir
output_dir = args.output_dir
batch_size = args.batch_size
percent = args.percent
neuron_multi_dir = args.neuron_multi_dir
mode = args.mode


os.makedirs(args.output_dir, exist_ok=True)

# Load data and tokenizer first so we can compute a safe max_model_len
# (guaranteed to be larger than the longest input + max output tokens).
data = load_queries(args.dataset)
total_len = len(data)
tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

if args.max_model_len is not None:
    max_model_len = args.max_model_len
    print(f"[INFO] max_model_len (override): {max_model_len}")
else:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    max_input_len = 0
    for preference in list(interval.keys()):
        prefs = preference.split(".")
        for question in data:
            input_text = PREFERENCE_TEMPLATE.format(question=question,
                                                    preference1=prefs[0],
                                                    preference2=prefs[1])
            chat = tokenizer.apply_chat_template(
                [{"role": "user", "content": input_text}],
                add_generation_prompt=True, tokenize=False, enable_thinking=False)
            max_input_len = max(max_input_len, len(tokenizer.encode(chat)))
    max_model_len = max_input_len + 400 + 128  # 400 = max_tokens in SamplingParams, 128 margin
    # clamp to the model's native max position embeddings
    try:
        from transformers import AutoConfig
        native = AutoConfig.from_pretrained(args.model, trust_remote_code=True).max_position_embeddings
        max_model_len = min(max_model_len, native)
    except Exception:
        pass
    print(f"[INFO] computed max_model_len = {max_model_len} "
          f"(max_input_len={max_input_len} + 400 + 128)")

model = LLM(model=args.model, tensor_parallel_size=torch.cuda.device_count(), 
            enforce_eager=True, gpu_memory_utilization=0.8,
            max_model_len=max_model_len)
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
            if str(idx) in neuron_to_deactivate:
                elements_to_deactivate=neuron_to_deactivate[str(idx)]
                elements, elements_to_deactivate = deduplicate_tensors_by_dim(elements, elements_to_deactivate)
                indices_to_deactivate=elements_to_deactivate[:, 1].long()
                gate_up[:, indices_to_deactivate] = 0.0  # Deactivation : the selected neurons are clamped to zero
            indices = elements[:, 1].long()
            # Amplification : the selected preference-associated neurons are
            # scaled up under the matched prompt: a_j -> a_j * (1 + val), with no sigmoid
            # gating, quantile weighting or random dropout.
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
            if str(idx) in neuron_to_deactivate:
                elements_to_deactivate=neuron_to_deactivate[str(idx)]
                elements, elements_to_deactivate = deduplicate_tensors_by_dim(elements, elements_to_deactivate)
                indices_to_deactivate=elements_to_deactivate[:, 1].long()
                gate_up[:, indices_to_deactivate] = 0.0  # Deactivation : the selected neurons are clamped to zero
            indices = elements[:, 1].long()
            # Amplification : the selected preference-associated neurons are
            # scaled up under the matched prompt: a_j -> a_j * (1 + val), with no sigmoid
            # gating, quantile weighting or random dropout.
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
            if str(idx) in neuron_to_deactivate:
                elements_to_deactivate=neuron_to_deactivate[str(idx)]
                elements, elements_to_deactivate = deduplicate_tensors_by_dim(elements, elements_to_deactivate)
                indices_to_deactivate=elements_to_deactivate[:, 1].long()
                gate_up[:, indices_to_deactivate] = 0.0  # Deactivation : the selected neurons are clamped to zero
            indices = elements[:, 1].long()
            # Amplification : the selected preference-associated neurons are
            # scaled up under the matched prompt: a_j -> a_j * (1 + val), with no sigmoid
            # gating, quantile weighting or random dropout.
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
    # Check if it's a single value or a range
    if isinstance(interval[preference], list):
        ratios = list(map(lambda x: round(x * 0.1,1), list(range(interval[preference][0],interval[preference][1]))))
    else:
        # Single value - use directly
        ratios = [interval[preference] * 0.1]
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
        elif mode in ('partialA1', 'partialA2', 'partialB1', 'partialB2'):
            # Multi-preference (neuron_multi_dir) minus single-preference (neuron_dir).
            # Scheme A: partial1 -> multi - single(pref1), partial2 -> multi - single(pref2)
            # Scheme B: partial1 -> multi - single(pref2), partial2 -> multi - single(pref1)
            # Deactivation sets follow the same rule with reversed preferences.
            p0, p1 = preferences[0], preferences[1]
            rp0 = p0.replace("low","high") if 'low' in p0 else p0.replace("high","low")
            rp1 = p1.replace("low","high") if 'low' in p1 else p1.replace("high","low")
            neuron_multi_path = f'{neuron_multi_dir}/{preference}_{percent}.json'
            neuron_multi_deact_path = f'{neuron_multi_dir}/{rp0}.{rp1}_{percent}.json'
            single1 = load_neuron_to_change(f'{neuron_dir}/{p0}_{percent}.json')
            single2 = load_neuron_to_change(f'{neuron_dir}/{p1}_{percent}.json')
            single_deact1 = load_neuron_to_change(f'{neuron_dir}/{rp0}_{percent}.json')
            single_deact2 = load_neuron_to_change(f'{neuron_dir}/{rp1}_{percent}.json')
            multi = load_neuron_to_change(neuron_multi_path)
            multi_deact = load_neuron_to_change(neuron_multi_deact_path)
            if mode in ('partialA1', 'partialB2'):
                neuron_to_change = myPartial(multi, single1)
                neuron_to_deactivate = myPartial(multi_deact, single_deact1)
            else:  # partialA2, partialB1
                neuron_to_change = myPartial(multi, single2)
                neuron_to_deactivate = myPartial(multi_deact, single_deact2)
        elif mode == 'dual':
            # Steer BOTH preferences' exclusive neuron sets simultaneously
            # (high-higher + low-lower): change = (N(p0)-N(p1)) U (N(p1)-N(p0)),
            # deactivate = (N(rev0)-N(rev1)) U (N(rev1)-N(rev0)).
            p0, p1 = preferences[0], preferences[1]
            rp0 = p0.replace("low","high") if 'low' in p0 else p0.replace("high","low")
            rp1 = p1.replace("low","high") if 'low' in p1 else p1.replace("high","low")
            s0 = load_neuron_to_change(f'{neuron_dir}/{p0}_{percent}.json')
            s1 = load_neuron_to_change(f'{neuron_dir}/{p1}_{percent}.json')
            sd0 = load_neuron_to_change(f'{neuron_dir}/{rp0}_{percent}.json')
            sd1 = load_neuron_to_change(f'{neuron_dir}/{rp1}_{percent}.json')
            neuron_to_change = merge(myPartial(s0, s1), myPartial(s1, s0))
            neuron_to_deactivate = merge(myPartial(sd0, sd1), myPartial(sd1, sd0))
        elif mode == 'mdiff':
            # Steer the pure multi-preference difference part:
            # change   = multi(pair) - (single(p0) U single(p1))
            # deact    = multi(rev)  - (single(rev0) U single(rev1))
            p0, p1 = preferences[0], preferences[1]
            rp0 = p0.replace("low","high") if 'low' in p0 else p0.replace("high","low")
            rp1 = p1.replace("low","high") if 'low' in p1 else p1.replace("high","low")
            multi = load_neuron_to_change(f'{neuron_multi_dir}/{preference}_{percent}.json')
            multi_deact = load_neuron_to_change(f'{neuron_multi_dir}/{rp0}.{rp1}_{percent}.json')
            s0 = load_neuron_to_change(f'{neuron_dir}/{p0}_{percent}.json')
            s1 = load_neuron_to_change(f'{neuron_dir}/{p1}_{percent}.json')
            sd0 = load_neuron_to_change(f'{neuron_dir}/{rp0}_{percent}.json')
            sd1 = load_neuron_to_change(f'{neuron_dir}/{rp1}_{percent}.json')
            neuron_to_change = myPartial(multi, merge(s0, s1))
            neuron_to_deactivate = myPartial(multi_deact, merge(sd0, sd1))
        elif mode == 'mint':
            # Steer the multi-preference set's INTERSECTION with each single:
            # change = (multi ∩ single(p0)) U (multi ∩ single(p1))
            # deact  = (multi(rev) ∩ single(rev0)) U (multi(rev) ∩ single(rev1))
            p0, p1 = preferences[0], preferences[1]
            rp0 = p0.replace("low","high") if 'low' in p0 else p0.replace("high","low")
            rp1 = p1.replace("low","high") if 'low' in p1 else p1.replace("high","low")
            multi = load_neuron_to_change(f'{neuron_multi_dir}/{preference}_{percent}.json')
            multi_deact = load_neuron_to_change(f'{neuron_multi_dir}/{rp0}.{rp1}_{percent}.json')
            s0 = load_neuron_to_change(f'{neuron_dir}/{p0}_{percent}.json')
            s1 = load_neuron_to_change(f'{neuron_dir}/{p1}_{percent}.json')
            sd0 = load_neuron_to_change(f'{neuron_dir}/{rp0}_{percent}.json')
            sd1 = load_neuron_to_change(f'{neuron_dir}/{rp1}_{percent}.json')
            neuron_to_change = merge(intersect(multi, s0), intersect(multi, s1))
            neuron_to_deactivate = merge(intersect(multi_deact, sd0), intersect(multi_deact, sd1))
        elif mode == 'sdiff':
            # Reverse of A1/A2: single-preference minus multi-preference sets.
            # change = (single(p0) - multi(pair)) U (single(p1) - multi(pair))
            # deact  = (single(rev0) - multi(rev)) U (single(rev1) - multi(rev))
            p0, p1 = preferences[0], preferences[1]
            rp0 = p0.replace("low","high") if 'low' in p0 else p0.replace("high","low")
            rp1 = p1.replace("low","high") if 'low' in p1 else p1.replace("high","low")
            multi = load_neuron_to_change(f'{neuron_multi_dir}/{preference}_{percent}.json')
            multi_deact = load_neuron_to_change(f'{neuron_multi_dir}/{rp0}.{rp1}_{percent}.json')
            s0 = load_neuron_to_change(f'{neuron_dir}/{p0}_{percent}.json')
            s1 = load_neuron_to_change(f'{neuron_dir}/{p1}_{percent}.json')
            sd0 = load_neuron_to_change(f'{neuron_dir}/{rp0}_{percent}.json')
            sd1 = load_neuron_to_change(f'{neuron_dir}/{rp1}_{percent}.json')
            neuron_to_change = merge(myPartial(s0, multi), myPartial(s1, multi))
            neuron_to_deactivate = merge(myPartial(sd0, multi_deact), myPartial(sd1, multi_deact))

        if args.no_deactivate:
            neuron_to_deactivate = {}
        if args.no_steer:
            neuron_to_change = {}
                
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
                    input_text = PREFERENCE_TEMPLATE.format(question=question, preference1=instructions[preferences[0]], preference2=instructions[preferences[1]])
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

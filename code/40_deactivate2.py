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
You are given a preference followed by a question. Write a short response (under 300 words) that reflects the preference as clearly as possible. Don’t worry about correctness or structure. Focus on alignment with the preference.
###Preference requirement:{preference}

###Question:{question}

###Response:
"""
#!!!!!!!!!!!!!!!!
parser = argparse.ArgumentParser()
parser.add_argument( "--model", type=str, required=True, help="Path to the model")  # 模型路径
parser.add_argument( "--neuron_dir", type=str, required=True, help="Path to neuron results directory")
parser.add_argument( "--output_dir", type=str, required=True, help="Path to output directory")
parser.add_argument( "--batch_size", type=int, default=32, help="Batch size")
parser.add_argument( "--deactivate", type=str, default="preference_neurons", help="Way to deactivate")
parser.add_argument("--dataset", type=str)
parser.add_argument("--delta", type=float)
args = parser.parse_args()
neuron_dir = args.neuron_dir
output_dir = args.output_dir
batch_size = args.batch_size
delta = args.delta
model = LLM(model=args.model, tensor_parallel_size=torch.cuda.device_count(), enforce_eager=True)  # 加载模型
tokenizer = AutoTokenizer.from_pretrained(args.model)  # 加载分词器
num_layers = model.llm_engine.model_config.hf_config.num_hidden_layers
intermediate_size = model.llm_engine.model_config.hf_config.intermediate_size
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#读取要改的神经元
def load_neuron_to_change(neuron_to_change_path):
    with open(neuron_to_change_path, 'r', encoding='utf-8') as file:
        neuron_to_change = json.loads(file.read())
        for t in neuron_to_change:
            neuron_to_change[t]=torch.tensor(neuron_to_change[t]).to(device)
    return neuron_to_change

def factory(idx): 
    def qwen_forward(self, x): 
        gate_up, _ = self.gate_up_proj(x)
        i = gate_up.size(-1)
        j=x.shape[0]
        gate_up[:, : i // 2] = torch.nn.SiLU()(gate_up[:, : i // 2])  # 过一个激活函数.neurn=self.act_fn(self.gate_proj(x))
        if  j <= batch_size and str(idx) in neuron_to_deactivate:
            elements_to_deactivate=neuron_to_deactivate[str(idx)]
            indices_to_deactivate=elements_to_deactivate[:, 1].long()
            if args.deactivate == 'preference_neurons':
                gate_up[:, indices_to_deactivate] = 0.0  # Deactivation : the selected neurons are clamped to zero
            elif args.deactivate == 'random':
                neuron_num = len(indices_to_deactivate)
                cur_mask = torch.randperm(intermediate_size)[:neuron_num].to('cuda')
                gate_up[:, cur_mask] = 0.0  # Random-neuron control: matched-size random neurons clamped to zero
        x = gate_up[:, : i // 2] * gate_up[:, i // 2:]
        x, _ = self.down_proj(x)
        return x


    return qwen_forward
data = load_queries(args.dataset)
total_len = len(data)
for i in range(num_layers):
    obj = model.llm_engine.model_executor.driver_worker.model_runner.model.model.layers[i].mlp ##
    obj.forward = MethodType(factory(i), obj)  # 绑定
#preference_list = ["Expertise_low", "Expertise_high","Informativeness_low","Informativeness_high","Style_low","Style_high"]
preference_list = ["Informativeness_high"]
for preference in preference_list:
            neuron_to_deactivate_path=f'{neuron_dir}/{preference}_top_{delta}percent.json'
            output_dir_bfi=f'{output_dir}/{preference}'
            os.makedirs(output_dir_bfi, exist_ok=True)
            output_file=f'{output_dir_bfi}/{preference}_{delta}.json'
            with open(output_file, 'w', encoding='utf-8') as output_file:
                neuron_to_deactivate=load_neuron_to_change(neuron_to_deactivate_path)
                for i in tqdm(range(0, total_len, batch_size)):
                    num_of_words=0
                    end_idx = min(total_len, i + batch_size)
                    batch_questions = data[i: end_idx]
                    input_texts = []
                    for question in batch_questions:
                        input_text =TEMPLATE.format(preference=instructions[preference], question=question)
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

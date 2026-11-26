# Add sum activate over zero
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
# os.environ['CUDA_VISIBLE_DEVICES'] = '6'
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
from vllm import LLM, SamplingParams
import json
from types import MethodType
from datasets import load_dataset, load_from_disk
from lib_data_utils import load_queries  # packaged query pools (JSONL/CSV or save_to_disk)
from lib_instructions import instructions, doubled_instructions, tripled_instructions,Helpful_instructions, Informative_instructions 
from tqdm import tqdm
import argparse
import gc


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="../models/Qwen3-8B")
parser.add_argument("--answer_path", type=str, default="./10_21/answer")
parser.add_argument("--neuron_path", type=str, default="./10_21/neuron")
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--dataset", type=str, default="<PACKAGE_ROOT>/dataset/psoups_localization_2525.jsonl",
                    help="Query pool for induction (JSONL with a 'query' field, or a HF save_to_disk directory)")

args = parser.parse_args()
os.makedirs(args.answer_path, exist_ok=True)
os.makedirs(args.neuron_path, exist_ok=True)
batch_size = args.batch_size

is_llama = is_qwen = is_gemma = False
if "Llama" in args.model:
    is_llama = True
if "gemma" in args.model:
    is_gemma = True

if "Qwen" in args.model:
    is_qwen = True
    
tokenizer = AutoTokenizer.from_pretrained(args.model)
sampling_params = SamplingParams(
    max_tokens=500,
    temperature=0,
    repetition_penalty=1.1
)
model = LLM(model=args.model, tensor_parallel_size=1, enforce_eager=True)  # 加载模型
if tokenizer.pad_token == None:
    tokenizer.pad_token = tokenizer.eos_token



max_model_length = model.llm_engine.model_config.max_model_len
num_layers = model.llm_engine.model_config.hf_config.num_hidden_layers
intermediate_size = model.llm_engine.model_config.hf_config.intermediate_size
hidden_size = model.llm_engine.model_config.hf_config.hidden_size


def factory(idx):
    def qwen_forward(self, x):
        gate_up, _ = self.gate_up_proj(x) 
        i = gate_up.size(-1)
        j = x.shape[0]
        gate_up[:, : i // 2] = torch.nn.SiLU()(gate_up[:, : i // 2])
        activation = gate_up[:, : i // 2].float()  # b,neuron
        if j <= batch_size:
            global token_num
            token_num+=j
            over_zero[idx, :] += (activation > 0).sum(dim=(0))
            sum_activation[idx, :] += activation.sum(dim=(0))
            sum_activation_over_zero[idx, :] += (activation * (activation > 0)).sum(dim=(0))
            bin_indices = (torch.bucketize(activation, bins) - 1).transpose(0, 1)
            increment_tensor = torch.ones_like(bin_indices, dtype=histograms[idx, 0].dtype)
            histograms[idx].scatter_add_(1, bin_indices, increment_tensor)
        x = gate_up[:, : i // 2] * gate_up[:, i // 2:]
        x, _ = self.down_proj(x)
        return x
    
    def llama_forward(self, x):
        gate_up, _ = self.gate_up_proj(x) 
        i = gate_up.size(-1)
        j = x.shape[0]
        gate_up[:, : i // 2] = torch.nn.SiLU()(gate_up[:, : i // 2])
        activation = gate_up[:, : i // 2].float()  # b,neuron
        if j <= batch_size:
            global token_num
            token_num+=j
            over_zero[idx, :] += (activation > 0).sum(dim=(0))
            sum_activation[idx, :] += activation.sum(dim=(0))
            sum_activation_over_zero[idx, :] += (activation * (activation > 0)).sum(dim=(0))
            bin_indices = (torch.bucketize(activation, bins) - 1).transpose(0, 1)
            increment_tensor = torch.ones_like(bin_indices, dtype=histograms[idx, 0].dtype)
            histograms[idx].scatter_add_(1, bin_indices, increment_tensor)
        x = gate_up[:, : i // 2] * gate_up[:, i // 2:]
        x, _ = self.down_proj(x)
        return x
    
    def gemma_forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        i = gate_up.size(-1)
        j = x.shape[0]
        gate_up[:, : i // 2] = torch.nn.GELU()(gate_up[:, : i // 2])
        activation = gate_up[:, : i // 2].float()  # b,neuron
        if j <= batch_size:
            global token_num
            token_num+=j
            over_zero[idx, :] += (activation > 0).sum(dim=(0))
            sum_activation[idx, :] += activation.sum(dim=(0))
            sum_activation_over_zero[idx, :] += (activation * (activation > 0)).sum(dim=(0))
            bin_indices = (torch.bucketize(activation, bins) - 1).transpose(0, 1)
            increment_tensor = torch.ones_like(bin_indices, dtype=histograms[idx, 0].dtype)
            histograms[idx].scatter_add_(1, bin_indices, increment_tensor)
        x = gate_up[:, : i // 2] * gate_up[:, i // 2:]
        x, _ = self.down_proj(x)
        return x
    
    if is_llama: return llama_forward
    elif is_gemma: return gemma_forward
    elif is_qwen: return qwen_forward


TEMPLATE="""
You are given a preference followed by a question. Write a short response (under 300 words) that reflects the preference as clearly as possible. Don’t worry about correctness or structure. Focus on alignment with the preference.
###Preference requirement:{preference}

###Question:{question}

###Response:
"""

data = load_queries(args.dataset)
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
    obj.forward = MethodType(normal_forward(), obj)  # 绑定
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
preference_list = ["Expertise_low","Expertise_high","Style_low","Style_high","None","Informativeness_low","Informativeness_high"]
for preference in preference_list:
    threshold = 400

    for i in range(num_layers):
        obj = model.llm_engine.model_executor.driver_worker.model_runner.model.model.layers[i].mlp
        obj.forward = MethodType(factory(i), obj)  # 绑定
    over_zero = torch.zeros(num_layers, intermediate_size, dtype=torch.int32).to('cuda')
    sum_activation = torch.zeros(num_layers, intermediate_size, dtype=torch.float32).to('cuda')
    sum_activation_over_zero = torch.zeros(num_layers, intermediate_size, dtype=torch.float32).to('cuda')
    token_num = 0 
    ans_list = []
    histograms = torch.zeros((num_layers, intermediate_size, 301)).cuda()
    bins = torch.cat([torch.tensor([-float('inf')]).to('cuda'), torch.arange(0, 3.01, 0.01).to('cuda')])
    bins[-1] = float('inf')
    num_bins = 301
    input_texts = []
    output_file_qa=f'{args.answer_path}/{preference}.Qwen3-8B.json'
    output_file_neuron=f'{args.neuron_path}/{preference}.Qwen3-8B.pt'
    with open(output_file_qa, 'w', encoding='utf-8') as output_file_qa:
        for i in tqdm(range(0, total_len, batch_size)):
            end_idx = min(total_len, i + batch_size)
            batch_questions = data[i: end_idx]
            input_texts = []
            for question in batch_questions:
                input_text = TEMPLATE.format(preference = instructions[preference], question=question)
                input_texts.append([{"role": "user", "content": input_text}])
            if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
            input_ids = tokenizer.apply_chat_template(input_texts, add_generation_prompt=True, tokenize=False, enable_thinking=False)
            sampling_params = SamplingParams(
                    max_tokens=threshold,
                    temperature=0,
                    repetition_penalty=1.1
            )
            output = model.generate(input_ids, sampling_params=sampling_params)
            ans_list = ans_list+[json.dumps({"question": batch_questions[i], "answer": output[i].outputs[0].text}) + '\n' for i in range(len(batch_questions))]
        output_file_qa.writelines(ans_list)
        avg_activation_all = sum_activation / (token_num * num_layers)
        avg_activation_over_zero = torch.zeros_like(sum_activation_over_zero, dtype=torch.float32)
        mask = over_zero > 0
        avg_activation_over_zero[mask] = sum_activation_over_zero[mask] / over_zero[mask]
        output = dict(token_num=token_num/num_layers,question_num=total_len, over_zero=over_zero.to('cpu'),sum_activation=sum_activation.to('cpu'), sum_activation_over_zero=sum_activation_over_zero.to('cpu'), avg_activation_all=avg_activation_all.to('cpu'), avg_activation_over_zero=avg_activation_over_zero.to('cpu'),histograms=histograms.to('cpu'))
        torch.save(output, output_file_neuron) 
    del over_zero, sum_activation, sum_activation_over_zero, histograms, bins, ans_list, output
    torch.cuda.empty_cache()  
    gc.collect()

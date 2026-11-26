import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
from vllm import LLM, SamplingParams
from lib_instructions import coherence_template, helpfulness_template, expertise_template, informativeness_template, correctness_template, user_content, friendly_prompt,fluency_prompt, friendly_prompt2
from utils.utils import evaluate_with_judge_model
from transformers import AutoTokenizer
import json
import torch
import argparse
import datetime
import numpy as np
from scipy import stats
date_string = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="<MODELS_DIR>/Qwen2.5-72B-Instruct-AWQ")
parser.add_argument("--input_path", type = str, default="./res")
parser.add_argument("--out_path", type=str, default=f"../LLM_results/res")
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument("--principle", type=str, default='expert')
parser.add_argument("--average", action="store_true", help="Run evaluation 5 times per file and take average score")
args = parser.parse_args()
out_path = args.out_path + f"_{date_string}"
os.makedirs(out_path, exist_ok=True)
os.makedirs("./logs", exist_ok=True)
with open(os.path.join(out_path, "readme.txt"), 'w') as f:
    f.write(args.input_path)
print(out_path)
print("\n")
principle = args.principle
batch_size = args.batch_size
import logging
logging.basicConfig(filename=f"./logs/log_{date_string}.txt", filemode="w", format="%(asctime)s %(name)s:%(levelname)s:%(message)s", datefmt="%d-%M-%Y %H:%M:%S", level=logging.DEBUG)
logging.info(f"File: {os.path.basename(__file__)}")
logging.info(f'Model: {args.model}')
logging.info(f'Input Path: {args.input_path}')
logging.info(f'Output Path: {out_path}')
logging.info(f'Principle: {principle}')
def get_data(filepath):
    datas = []
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():  # 确保不处理空行
                data = json.loads(line)
                datas.append(data)
    return datas

# model_name = "../models/Qwen2.5-72B-Instruct-AWQ"

# llm = LLM(args.model, gpu_memory_utilization=0.95,tensor_parallel_size=1)
# tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side='left')
# folder_path = args.input_path
# paths = sorted(os.listdir(folder_path))
# for p in paths:
#     file_path = os.path.join(folder_path,p)
#     if os.path.isdir(file_path): continue
#     data = get_data(file_path)

#     scores_list = []
#     save_path = os.path.join(args.out_path, file_path.split("/")[-1].strip(".json") + ".scores.json")
#     # if os.path.exists(save_path): continue
#     prompt_template = user_content
#     if principle == "helpful":
#             system_prompt = helpfulness_template
#     elif principle == "correct":
#             system_prompt = correctness_template
#     elif principle == "coherent":
#             system_prompt = coherence_template
#             prompt_template = "response:{response}\nScore:\n"
#     elif principle == "informative":
#             system_prompt = informativeness_template
#     elif principle == "expert":
#             system_prompt = expertise_template
#     elif principle == "style":
#         system_prompt = friendly_prompt
#     elif principle == "fluency":
#         system_prompt = fluency_prompt
#         prompt_template = "response:{response}\nScore:\n"
#     else:
#             raise ValueError("Invalid Prefence")

#     if principle == 'fluency' or principle == 'coherent':
#         queries = [
#             [
#                 {"role":'system',"content": system_prompt},
#                 {"role":'user', 'content': prompt_template.format_map({'response': d['answer']})}
#             ]
#             for d in data]
#     else:
#          queries = [
#             [
#                 {"role":'system',"content": system_prompt},
#                 {"role":'user', 'content': prompt_template.format_map({'query': d['question'],'response': d['answer']})}
#             ]
#             for d in data]
#     cur_scores, mean_value = evaluate_with_judge_model(llm, sampling_params, queries, batch_size=4, tokenizer = tokenizer)
#     scores_list.append({"principle": principle, "mean_score": mean_value})
#     print(p)
#     print(scores_list)
#     #if principle == 'fluency' and mean_value >= 4.3530:
#     #    print("符合要求")
#     #else:
#     #    print("不符合要求")
#     #print("\n")
#     with open(save_path, 'w') as f:
#         json.dump({'scores': scores_list}, f, ensure_ascii=False, indent=4)
            
#     torch.cuda.empty_cache()


def evaluate_average(count = 5, sampling_params=None, tokenizer=None, llm=None):
    """
    对每个文件运行count次评估，取平均分数
    """
    print(f"\n=== 开始平均评估模式 (每个文件运行{count}次) ===")
    
    # 获取文件列表
    folder_path = args.input_path
    paths = sorted(os.listdir(folder_path))
    
    for p in paths:
        file_path = os.path.join(folder_path, p)
        if os.path.isdir(file_path):
            continue
        
        print(f"\n处理文件: {p}")
        data = get_data(file_path)
        prompt_template = user_content
        if args.principle == "helpful":
                system_prompt = helpfulness_template
        elif args.principle == "correct":
                system_prompt = correctness_template
        elif args.principle == "coherent":
                system_prompt = coherence_template
                prompt_template = "response:{response}\nScore:\n"
        elif args.principle == "informative":
                system_prompt = informativeness_template
        elif args.principle == "expert":
                system_prompt = expertise_template
        elif args.principle == "style":
            system_prompt = friendly_prompt
        elif args.principle == "fluency":
            system_prompt = fluency_prompt
            prompt_template = "response:{response}\nScore:\n"
        else:
                raise ValueError("Invalid Prefence")

        # 构建查询
        if args.principle == 'fluency' or args.principle == 'coherent':
            queries = [
                [
                    {"role":'system',"content": system_prompt},
                    {"role":'user', 'content': prompt_template.format_map({'response': d['answer']})}
                ]
                for d in data]
        else:
             queries = [
                [
                    {"role":'system',"content": system_prompt},
                    {"role":'user', 'content': prompt_template.format_map({'query': d['question'],'response': d['answer']})}
                ]
                for d in data]

        all_mean_values = []
        all_scores = []  
        for run in range(count):
            print(f"  第 {run + 1}/{count} 次运行...")
            
            cur_scores, mean_value = evaluate_with_judge_model(llm, sampling_params, queries, batch_size=batch_size, tokenizer=tokenizer)
            all_mean_values.append(mean_value)
            all_scores.append(cur_scores.tolist())  
            print(f"  第 {run + 1}/{count} 次运行得分: {mean_value:.4f}")
            
            # 清理显存
            torch.cuda.empty_cache()
        
        # 计算count次运行的平均值
        final_average = sum(all_mean_values) / len(all_mean_values)
        # 利用np计算标准差
        std_dev = np.std(all_mean_values)
        # t_stat, p_value = stats.ttest_1samp(all_mean_values, popmean=final_average)
        # mean = np.mean(all_mean_values)
        # std = np.std(all_mean_values)
        # n = len(all_mean_values)
        # confidence_interval = stats.t.interval(0.95, n-1, loc=mean, scale=stats.sem(all_mean_values))
# 如果置信区间不包含基准值，则显著
        
        print(f"\n  评估{args.principle}, 文件 {p} 的{count}次运行平均分数和标准差: {final_average:.4f} +- {std_dev:.4f}")
        print(f"  每个query的分数已保存")
        save_path = os.path.join(out_path, file_path.split("/")[-1].strip(".json") + f".average_{count}.scores.json")
        scores_list = [{"principle": args.principle, "mean_scores": all_mean_values, "average_score": final_average, "std_dev": std_dev, "per_query_scores": all_scores}]
        with open(save_path, 'w') as f:
            json.dump({'scores': scores_list}, f, ensure_ascii=False, indent=4)
        
        print(f"  结果已保存到: {save_path}")
    
    print(f"\n=== 平均评估模式完成！结果保存至: {out_path} ===")


# 主程序入口
if __name__ == "__main__":
    llm = LLM(args.model, 
              gpu_memory_utilization=0.8, 
              tensor_parallel_size=torch.cuda.device_count(), 
              enforce_eager=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side='left')
    sampling_params = SamplingParams(
        max_tokens=128,
        skip_special_tokens=True,
        temperature=0.8,
        top_p=0.95
    )
    if args.average:
        # 运行平均评估
        evaluate_average(count = 5, sampling_params=sampling_params, tokenizer=tokenizer, llm=llm)
    else:
        evaluate_average(count = 1, sampling_params=sampling_params, tokenizer=tokenizer, llm=llm)
        # # 原有的单轮评估逻辑
        # llm = LLM(args.model, gpu_memory_utilization=0.95, tensor_parallel_size=1)
        # tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side='left')
        
        # # 获取文件列表
        # folder_path = args.input_path
        # paths = sorted(os.listdir(folder_path))
        
        # for p in paths:
        #     file_path = os.path.join(folder_path, p)
        #     if os.path.isdir(file_path): continue
        #     data = get_data(file_path)

        #     scores_list = []
        #     save_path = os.path.join(args.out_path, file_path.split("/")[-1].strip(".json") + ".scores.json")
        #     # if os.path.exists(save_path): continue
        #     prompt_template = user_content
        #     if principle == "helpful":
        #             system_prompt = helpfulness_template
        #     elif principle == "correct":
        #             system_prompt = correctness_template
        #     elif principle == "coherent":
        #             system_prompt = coherence_template
        #             prompt_template = "response:{response}\nScore:\n"
        #     elif principle == "informative":
        #             system_prompt = informativeness_template
        #     elif principle == "expert":
        #             system_prompt = expertise_template
        #     elif principle == "style":
        #         system_prompt = friendly_prompt
        #     elif principle == "fluency":
        #         system_prompt = fluency_prompt
        #         prompt_template = "response:{response}\nScore:\n"
        #     else:
        #             raise ValueError("Invalid Prefence")

        #     if principle == 'fluency' or principle == 'coherent':
        #         queries = [
        #             [
        #                 {"role":'system',"content": system_prompt},
        #                 {"role":'user', 'content': prompt_template.format_map({'response': d['answer']})}
        #             ]
        #             for d in data]
        #     else:
        #          queries = [
        #             [
        #                 {"role":'system',"content": system_prompt},
        #                 {"role":'user', 'content': prompt_template.format_map({'query': d['question'],'response': d['answer']})}
        #             ]
        #             for d in data]
        #     cur_scores, mean_value = evaluate_with_judge_model(llm, sampling_params, queries, batch_size=4, tokenizer=tokenizer)
        #     scores_list.append({"principle": principle, "mean_score": mean_value})
        #     print(p)
        #     print(scores_list)
        #     #if principle == 'fluency' and mean_value >= 4.3530:
        #     #    print("符合要求")
        #     #else:
        #     #    print("不符合要求")
        #     #print("\n")
        #     with open(save_path, 'w') as f:
        #         json.dump({'scores': scores_list}, f, ensure_ascii=False, indent=4)
            
    torch.cuda.empty_cache()



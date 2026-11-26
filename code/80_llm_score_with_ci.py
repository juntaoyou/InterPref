import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from lib_instructions import coherence_template, helpfulness_template, expertise_template, informativeness_template, correctness_template, user_content, friendly_prompt, fluency_prompt, friendly_prompt2
from utils.utils import evaluate_with_judge_model
from transformers import AutoTokenizer
import json
import torch
import argparse
import datetime
import numpy as np
from scipy import stats
import logging

date_string = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="<MODELS_DIR>/Qwen2.5-72B-Instruct-AWQ")
parser.add_argument("--input_path", type=str, default="./res")
parser.add_argument("--out_path", type=str, default=f"../LLM_results/res")
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument("--principle", type=str, default='expert')
parser.add_argument("--num_runs", type=int, default=3,
                    help="每个文件运行的评估次数（默认3次，用于计算3σ误差和置信区间）")
parser.add_argument("--confidence_level", type=float, default=0.95,
                    help="置信水平（默认0.95，即95%置信区间）")
parser.add_argument("--sigma_multiplier", type=float, default=3.0,
                    help="Sigma倍数，用于计算误差范围（默认3.0，即3σ误差）")
args = parser.parse_args()

out_path = args.out_path + f"_{date_string}"
os.makedirs(out_path, exist_ok=True)
os.makedirs("./logs", exist_ok=True)

with open(os.path.join(out_path, "readme.txt"), 'w') as f:
    f.write(args.input_path)
print(out_path)
print("\n")

logging.basicConfig(
    filename=f"./logs/log_ci_{date_string}.txt",
    filemode="w",
    format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
    datefmt="%d-%M-%Y %H:%M:%S",
    level=logging.DEBUG
)
logging.info(f"File: {os.path.basename(__file__)}")
logging.info(f'Model: {args.model}')
logging.info(f'Input Path: {args.input_path}')
logging.info(f'Output Path: {out_path}')
logging.info(f'Principle: {args.principle}')
logging.info(f'Number of runs: {args.num_runs}')
logging.info(f'Confidence level: {args.confidence_level}')
logging.info(f'Sigma multiplier: {args.sigma_multiplier}')


def get_data(filepath):
    datas = []
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():
                data = json.loads(line)
                datas.append(data)
    return datas


def build_queries(data, principle, prompt_template, system_prompt):
    """根据principle构建查询列表"""
    if principle == 'fluency' or principle == 'coherent':
        queries = [
            [
                {"role": 'system', "content": system_prompt},
                {"role": 'user', 'content': prompt_template.format_map({'response': d['answer']})}
            ]
            for d in data
        ]
    else:
        queries = [
            [
                {"role": 'system', "content": system_prompt},
                {"role": 'user', 'content': prompt_template.format_map({'query': d['question'], 'response': d['answer']})}
            ]
            for d in data
        ]
    return queries


def get_system_prompt_and_template(principle):
    """根据principle获取system prompt和prompt template"""
    prompt_template = user_content
    if principle == "helpful":
        system_prompt = helpfulness_template
    elif principle == "correct":
        system_prompt = correctness_template
    elif principle == "coherent":
        system_prompt = coherence_template
        prompt_template = "response:{response}\nScore:\n"
    elif principle == "informative":
        system_prompt = informativeness_template
    elif principle == "expert":
        system_prompt = expertise_template
    elif principle == "style":
        system_prompt = friendly_prompt
    elif principle == "fluency":
        system_prompt = fluency_prompt
        prompt_template = "response:{response}\nScore:\n"
    else:
        raise ValueError("Invalid Principle")
    return system_prompt, prompt_template


def compute_ci_and_sigma(mean_values, confidence_level=0.95, sigma_multiplier=3.0):
    """
    对多次运行的mean_values进行计算：
    - 均值 (mean)
    - 标准差 (std)
    - 3σ误差范围 (sigma_multiplier * std)
    - 置信区间 (confidence_level 置信水平)
    - 标准误差 (SEM)

    返回包含所有统计量的字典。
    """
    mean_values = np.array(mean_values)
    n = len(mean_values)
    mean = np.mean(mean_values)
    std = np.std(mean_values, ddof=1)  # 样本标准差 (无偏估计)
    sem = stats.sem(mean_values)       # 标准误差 = std / sqrt(n)

    # 置信区间 (使用t分布)
    ci = stats.t.interval(confidence_level, df=n - 1, loc=mean, scale=sem)

    # 3σ 误差范围
    sigma_error = sigma_multiplier * std

    return {
        "num_runs": n,
        "mean": float(mean),
        "std": float(std),
        "sem": float(sem),
        "sigma_multiplier": sigma_multiplier,
        "sigma_error": float(sigma_error),
        "sigma_range": [float(mean - sigma_error), float(mean + sigma_error)],
        "confidence_level": confidence_level,
        "confidence_interval": [float(ci[0]), float(ci[1])],
        "ci_width": float(ci[1] - ci[0]),
        "all_mean_values": [float(v) for v in mean_values],
    }


def evaluate_with_ci(count=3, confidence_level=0.95, sigma_multiplier=3.0,
                     sampling_params=None, tokenizer=None, llm=None):
    """
    对每个输入文件运行 count 次评估，计算：
    - 平均分
    - 标准差
    - 3σ 误差范围 (mean ± 3*std)
    - 置信区间 (默认95%)

    结果包含：
    - 每次运行的 mean_score
    - 每次运行的 per_query_scores
    - 最终的统计量
    """
    print(f"\n{'='*60}")
    print(f"=== 开始带置信区间的评估模式 (每个文件运行{count}次) ===")
    print(f"=== 置信水平: {confidence_level:.0%}, Sigma倍数: {sigma_multiplier} ===")
    print(f"{'='*60}\n")

    folder_path = args.input_path
    paths = sorted(os.listdir(folder_path))

    for p in paths:
        file_path = os.path.join(folder_path, p)
        if os.path.isdir(file_path):
            continue

        print(f"\n处理文件: {p}")
        data = get_data(file_path)
        logging.info(f"文件 {p}: 共 {len(data)} 条数据")

        system_prompt, prompt_template = get_system_prompt_and_template(args.principle)
        queries = build_queries(data, args.principle, prompt_template, system_prompt)
        print(f"  当前评估维度: {args.principle} | 样本数: {len(data)} | 运行次数: {count}")

        all_mean_values = []
        all_scores = []  # 存储每次运行中每个query的分数

        for run in range(count):
            print(f"  第 {run + 1}/{count} 次运行...")

            cur_scores, mean_value = evaluate_with_judge_model(
                llm, sampling_params, queries,
                batch_size=args.batch_size, tokenizer=tokenizer
            )
            all_mean_values.append(mean_value)
            all_scores.append(cur_scores.tolist())
            print(f"  第 {run + 1}/{count} 次运行得分: {mean_value:.4f}")

            torch.cuda.empty_cache()

        # ========== 统计计算 ==========
        stats_result = compute_ci_and_sigma(
            all_mean_values,
            confidence_level=confidence_level,
            sigma_multiplier=sigma_multiplier
        )

        # ========== 打印结果 ==========
        print(f"\n  ┌─────────────────────────────────────────────")
        print(f"  │ 评估维度: {args.principle}")
        print(f"  │ 文件: {p}")
        print(f"  │ 运行次数: {count}")
        print(f"  ├─────────────────────────────────────────────")
        print(f"  │ 各次均值得分:")
        for i, v in enumerate(stats_result["all_mean_values"]):
            print(f"  │   Run {i + 1}: {v:.4f}")
        print(f"  ├─────────────────────────────────────────────")
        print(f"  │ 平均分 (Mean):         {stats_result['mean']:.4f}")
        print(f"  │ 标准差 (Std):          {stats_result['std']:.4f}")
        print(f"  │ 标准误差 (SEM):        {stats_result['sem']:.4f}")
        print(f"  ├─────────────────────────────────────────────")
        print(f"  │ {sigma_multiplier}σ 误差范围:            {stats_result['sigma_error']:.4f}")
        print(f"  │ {sigma_multiplier}σ 置信区间:            [{stats_result['sigma_range'][0]:.4f}, {stats_result['sigma_range'][1]:.4f}]")
        print(f"  ├─────────────────────────────────────────────")
        print(f"  │ {confidence_level:.0%} 置信区间 (t分布):   [{stats_result['confidence_interval'][0]:.4f}, {stats_result['confidence_interval'][1]:.4f}]")
        print(f"  │ {confidence_level:.0%} 置信区间宽度:       {stats_result['ci_width']:.4f}")
        print(f"  └─────────────────────────────────────────────")

        # ========== 保存结果 ==========
        save_path = os.path.join(
            out_path,
            file_path.split("/")[-1].strip(".json") + f".ci_{count}runs.json"
        )
        output_data = {
            "metadata": {
                "model": args.model,
                "principle": args.principle,
                "input_file": p,
                "num_runs": count,
                "batch_size": args.batch_size,
                "confidence_level": confidence_level,
                "sigma_multiplier": sigma_multiplier,
            },
            "statistics": stats_result,
            "per_query_scores": all_scores,
        }
        with open(save_path, 'w') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)

        print(f"  ✓ 结果已保存到: {save_path}")

    print(f"\n{'='*60}")
    print(f"=== 评估完成！所有结果保存至: {out_path} ===")
    print(f"{'='*60}\n")


# ========== 主程序入口 ==========
if __name__ == "__main__":
    from vllm import LLM, SamplingParams

    llm = LLM(args.model,
              gpu_memory_utilization=0.9,
              max_model_len=8192,   
              tensor_parallel_size=torch.cuda.device_count(),
              enforce_eager=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side='left')
    sampling_params = SamplingParams(
        max_tokens=128,
        skip_special_tokens=True,
        temperature=0.8,
        top_p=0.95
    )

    evaluate_with_ci(
        count=args.num_runs,
        confidence_level=args.confidence_level,
        sigma_multiplier=args.sigma_multiplier,
        sampling_params=sampling_params,
        tokenizer=tokenizer,
        llm=llm
    )

    torch.cuda.empty_cache()

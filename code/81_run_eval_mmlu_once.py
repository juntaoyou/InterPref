#!/usr/bin/env python3
"""
MMLU Baseline 全维度评估
=========================
加载 Judge 模型，对 Baseline (Base/None.json) 评估所有维度：
  - fluency
  - expert
  - informative
  - style
结果使用缓存，重复运行直接读取。
=========================
"""

import os
import sys
import json
import datetime
import logging
import argparse
import re

import numpy as np
import torch
from scipy import stats
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from lib_instructions import (
    expertise_template, informativeness_template, friendly_prompt,
    fluency_prompt, user_content,
)
from utils.utils import evaluate_with_judge_model

# ==================== argparse ====================
parser = argparse.ArgumentParser(description="MMLU Baseline 全维度评估脚本")
parser.add_argument("--input_path", type=str, required=True,
                    help="原始输入目录，内含 Expertise_high/ 等子目录")
parser.add_argument("--save_dir", type=str, required=True,
                    help="评估结果保存根目录")
parser.add_argument("--baseline_dir", type=str, default=None,
                    help="包含 Base/None.json 的目录 (模型粒度)")

# Judge 模型参数
parser.add_argument("--judge_model", type=str,
                    default="<MODELS_DIR>/Qwen2.5-72B-Instruct-AWQ")
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--num_runs", type=int, default=3)
parser.add_argument("--confidence_level", type=float, default=0.95)
parser.add_argument("--sigma_multiplier", type=float, default=3.0)
parser.add_argument("--fluency_threshold", type=float, default=0.9)

# Generation 参数（当前已注释，baseline 评估无需使用）
# parser.add_argument("--generation_model", type=str, default=None)
# parser.add_argument("--neuron_dir", type=str, default=None)
# parser.add_argument("--neuron_multi_dir", type=str, default=None)
# parser.add_argument("--dataset", type=str, default=None)
# parser.add_argument("--gen_output_base", type=str, default=None)
# parser.add_argument("--gen_mode", type=str, default="union")
# parser.add_argument("--gen_percent", type=float, default=32.0)
# parser.add_argument("--max_adjust_rounds", type=int, default=3)
# parser.add_argument("--gen_batch_size", type=int, default=32)
# parser.add_argument("--gen_gpu", type=str, default=None,
#                     help="生成时使用的 GPU (如 '1')，不指定则使用全部")

args = parser.parse_args()

NUM_RUNS = args.num_runs
CONFIDENCE_LEVEL = args.confidence_level
SIGMA_MULTIPLIER = args.sigma_multiplier
BATCH_SIZE = args.batch_size
JUDGE_MODEL = args.judge_model
FLUENCY_THRESHOLD_RATIO = args.fluency_threshold  # baseline 评估暂未使用，保留以备扩展
# GEN_MODEL = args.generation_model or args.judge_model
# MAX_ROUNDS = args.max_adjust_rounds

logging.basicConfig(
    filename=f"./logs/eval_auto_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.txt",
    filemode="w",
    format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
    datefmt="%d-%M-%Y %H:%M:%S",
    level=logging.DEBUG,
    force=True,
)


# ==================== 工具函数 ====================

ALL_PRINCIPLES = ["fluency", "expert", "informative", "style"]

def get_principles(dir_name):
    principles = ["fluency"]
    if dir_name.startswith("Expertise"):
        principles.append("expert")
    if dir_name.startswith("Informativeness"):
        principles.append("informative")
    if dir_name.startswith("Style"):
        principles.append("style")
    return principles


def get_system_prompt_and_template(principle):
    prompt_template = user_content
    if principle == "expert":
        system_prompt = expertise_template
    elif principle == "informative":
        system_prompt = informativeness_template
    elif principle == "style":
        system_prompt = friendly_prompt
    elif principle == "fluency":
        system_prompt = fluency_prompt
        prompt_template = "response:{response}\nScore:\n"
    else:
        raise ValueError(f"未知 principle: {principle}")
    return system_prompt, prompt_template


def load_data(filepath):
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def build_queries(data, principle, prompt_template, system_prompt):
    if principle == "fluency":
        queries = [
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_template.format_map({"response": d["answer"]})},
            ]
            for d in data
        ]
    else:
        queries = [
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_template.format_map({
                    "query": d["question"], "response": d["answer"]
                })},
            ]
            for d in data
        ]
    return queries


def compute_ci_and_sigma(mean_values, confidence_level=0.95, sigma_multiplier=3.0):
    mean_values = np.array(mean_values)
    n = len(mean_values)
    mean = np.mean(mean_values)
    std = np.std(mean_values, ddof=1)
    sem = stats.sem(mean_values)
    ci = stats.t.interval(confidence_level, df=n - 1, loc=mean, scale=sem)
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


def parse_filename(filename):
    """Expertise_high_2.6.json -> (Expertise_high, 2.6)"""
    stem = filename.rstrip(".json")
    match = re.match(r'^(.+)_(\d+\.?\d*)$', stem)
    if match:
        return match.group(1), float(match.group(2))
    return None, None


# ==================== Baseline (全维度) ====================

def evaluate_baseline_all_dimensions(llm, tokenizer, sampling_params, baseline_dir, out_base):
    """
    对 baseline_dir/Base/None.json 评估所有维度（fluency, expert, informative, style）。
    每个维度独立缓存，结果输出到 out_base/baseline/。
    """
    if not baseline_dir:
        print("  [info] 未指定 --baseline_dir，跳过 baseline 评估")
        return {}

    none_file = os.path.join(baseline_dir, "Base", "None.json")
    if not os.path.exists(none_file):
        print(f"  [warning] {none_file} 不存在，跳过 baseline 评估")
        return {}

    base_out = os.path.join(out_base, "baseline")
    os.makedirs(base_out, exist_ok=True)

    data = load_data(none_file)
    results = {}

    for principle in ALL_PRINCIPLES:
        cache_file = os.path.join(base_out, f"None_{principle}_cache.json")

        # 命中缓存则跳过
        if os.path.exists(cache_file):
            with open(cache_file) as f:
                cached = json.load(f)
            print(f"  [OK] 读取缓存 baseline/{principle}: {cached['mean']:.4f}")
            results[principle] = cached
            continue

        print(f"\n  [info] 评估 baseline/{principle} ({len(data)} 条样本, {NUM_RUNS} runs) ...")
        sys_prompt, prompt_template = get_system_prompt_and_template(principle)
        queries = build_queries(data, principle, prompt_template, sys_prompt)

        all_mean = []
        all_per_query_scores = []  # 保存每条样本的分数
        for run in range(NUM_RUNS):
            cur_scores, mean_value = evaluate_with_judge_model(
                llm, sampling_params, queries, batch_size=BATCH_SIZE, tokenizer=tokenizer,
            )
            all_mean.append(mean_value)
            all_per_query_scores.append(cur_scores.tolist())
            print(f"         run {run + 1}/{NUM_RUNS}: {mean_value:.4f}")
            torch.cuda.empty_cache()

        sr = compute_ci_and_sigma(all_mean)

        out_data = {
            "mean": sr["mean"], "std": sr["std"], "num_runs": NUM_RUNS,
            "all_mean_values": sr["all_mean_values"],
            "sem": sr["sem"], "sigma_error": sr["sigma_error"],
            "confidence_interval": sr["confidence_interval"],
            "per_query_scores": all_per_query_scores,  # [run_idx][query_idx]
            "num_samples": len(data),
            "timestamp": datetime.datetime.now().isoformat(),
        }
        with open(cache_file, "w") as f:
            json.dump(out_data, f, indent=2)

        print(f"  [OK] baseline/{principle} = {sr['mean']:.4f} ± {sr['std']:.4f} (σ)")
        results[principle] = out_data

    return results


# ==================== 评估函数 ====================

def evaluate_single_file(llm, tokenizer, sampling_params, file_path, principle, verbose=True):
    """对单个 JSONL 文件执行评估，返回 stats_result。不保存（由调用方保存）。"""
    file_name = os.path.basename(file_path)
    data = load_data(file_path)
    logging.info(f"{file_name}: {len(data)}条 principle={principle}")

    sys_prompt, prompt_template = get_system_prompt_and_template(principle)
    queries = build_queries(data, principle, prompt_template, sys_prompt)

    if verbose:
        print(f"      样本: {len(data)} | runs: {NUM_RUNS}")

    all_mean = []
    all_scores = []
    for run in range(NUM_RUNS):
        if verbose:
            print(f"      run {run + 1}/{NUM_RUNS}...", end=" ")
        cur_scores, mean_value = evaluate_with_judge_model(
            llm, sampling_params, queries, batch_size=BATCH_SIZE, tokenizer=tokenizer,
        )
        all_mean.append(mean_value)
        all_scores.append(cur_scores.tolist())
        if verbose:
            print(f"{mean_value:.4f}")
        torch.cuda.empty_cache()

    sr = compute_ci_and_sigma(all_mean)

    if verbose:
        print(f"      => mean={sr['mean']:.4f} std={sr['std']:.4f}")

    return sr, all_scores


def save_eval_result(out_dir, file_name, principle, sr, all_scores):
    """保存评估结果到文件。"""
    os.makedirs(out_dir, exist_ok=True)
    save_name = file_name.rstrip(".json") + f".ci_{NUM_RUNS}runs.json"
    output = {
        "metadata": {"model": JUDGE_MODEL, "principle": principle, "input_file": file_name,
                     "num_runs": NUM_RUNS, "batch_size": BATCH_SIZE},
        "statistics": sr,
        "per_query_scores": all_scores,
    }
    with open(os.path.join(out_dir, save_name), "w") as f:
        json.dump(output, f, indent=4)


# ==================== 以下为自动调优相关函数（当前已注释） ====================

# def evaluate_fluency_for_dir(llm, tokenizer, sampling_params, dir_path, out_dir):
#     ...
#
# def evaluate_other_dimensions(llm, tokenizer, sampling_params, file_path, out_base):
#     ...
#
# def unload_judge_model(llm, tokenizer, sampling_params):
#     ...
#
# def reload_judge_model():
#     ...
#
# def run_generation(config_json_str, output_dir, round_label):
#     ...


# ==================== 主流程 ====================

def main():
    input_base = args.input_path
    exp_name = os.path.basename(input_base.rstrip("/"))
    date_string = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    out_base = os.path.join(args.save_dir, f"{exp_name}_{date_string}")
    os.makedirs(out_base, exist_ok=True)

    print("=" * 70)
    print("  MMLU Baseline 全维度评估")
    print("=" * 70)
    print(f"  Judge:      {JUDGE_MODEL}")
    print(f"  Input:      {input_base}")
    print(f"  Output:     {out_base}")
    print(f"  Baseline:   {args.baseline_dir or '(none)'}")
    print(f"  NUM_RUNS:   {NUM_RUNS}")
    print("-" * 70)

    # ---------- Load Judge Model ----------
    print(f"\n  Loading judge model ...")
    llm = LLM(JUDGE_MODEL, gpu_memory_utilization=0.9, max_model_len=8192,
              tensor_parallel_size=torch.cuda.device_count(), enforce_eager=True)
    tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL, padding_side="left")
    sampling_params = SamplingParams(
        max_tokens=128, skip_special_tokens=True, temperature=0.8, top_p=0.95)
    print(f"  [OK] Model loaded\n")

    # ---------- Baseline 全维度评估 ----------
    results = evaluate_baseline_all_dimensions(llm, tokenizer, sampling_params,
                                                args.baseline_dir, out_base)

    if not results:
        print("\n  [warning] Baseline 未产生任何结果，请检查 --baseline_dir 参数")
        return

    # ---------- 打印汇总 ----------
    print("\n" + "=" * 70)
    print("  Baseline 评估汇总")
    print("=" * 70)
    for principle, data in results.items():
        ci = data.get("confidence_interval", [0, 0])
        print(f"  {principle:12s}: mean={data['mean']:.4f}  std={data['std']:.4f}  "
              f"ci=[{ci[0]:.4f}, {ci[1]:.4f}]")
    print("-" * 70)
    print(f"  结果保存至: {out_base}/baseline/")
    print("=" * 70)


if __name__ == "__main__":
    main()

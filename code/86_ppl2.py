import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn import CrossEntropyLoss
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
from typing import List, Optional
import json
import numpy as np
from tqdm import tqdm

def load_model_and_tokenizer(model_name: str = "Qwen/Qwen3-8B-Base", device="cuda"):
    """
    加载模型和分词器，只加载一次以提高效率
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,  # 或 torch.float16，根据显存调整
        trust_remote_code=True,
        device_map="auto"  # 自动分配到可用 GPU
    )
    model.eval()
    return tokenizer, model

def calculate_ppl(text: str, tokenizer, model):
    """
    计算单个文本的PPL
    """
    # Tokenize 文本
    inputs = tokenizer(text, return_tensors="pt", truncation=True).to(model.device)
    input_ids = inputs.input_ids

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss  # 因为 labels=input_ids，loss 是平均负对数似然（token-level）

    ppl = torch.exp(loss).item()
    return ppl

def get_data(filepath):
    datas = []
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():  # 确保不处理空行
                data = json.loads(line)
                datas.append(data)
    return datas

def calculate_file_ppl(file_path: str, tokenizer, model, remove_outliers: bool = False):
    """
    计算单个文件的平均PPL，将长文件分块处理
    
    Args:
        file_path: 文件路径
        tokenizer: 分词器
        model: 模型
        max_chunk_length: 最大块长度
        remove_outliers: 是否移除异常值
    
    Returns:
        平均PPL值
    """
    try:
        # with open(file_path, 'r', encoding='utf-8') as file:
        #     datas = json.load(file)
        datas = get_data(file_path)
        answers = [d['answer'] for d in datas]
        ppl_list = []
        
        for answer in tqdm(answers):
            chunk_ppl = calculate_ppl(answer, tokenizer, model)
            if chunk_ppl != float('inf'):
                ppl_list.append(chunk_ppl)
        
        if not ppl_list:
            return float('inf')
        
        # 移除异常值（IQR方法）
        if remove_outliers and len(ppl_list) >= 4:
            # 计算四分位数
            q1 = np.percentile(ppl_list, 25)
            q3 = np.percentile(ppl_list, 75)
            iqr = q3 - q1
            
            # 定义异常值阈值
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # 过滤掉异常值
            filtered_ppl = [ppl for ppl in ppl_list if lower_bound <= ppl <= upper_bound]
            
            # 如果过滤后还有足够的数据，使用过滤后的数据计算平均值
            if filtered_ppl:
                print(f"文件 {file_path} 移除了 {len(ppl_list) - len(filtered_ppl)} 个异常值")
                print(f"原始数据点数: {len(ppl_list)}, 过滤后数据点数: {len(filtered_ppl)}")
                return sum(filtered_ppl) / len(filtered_ppl)
        
        # # 如果没有移除异常值或过滤后没有数据，使用所有有效数据计算平均值
        # print(f"文件 {file_path} 有效数据点数: {len(ppl_list)}")
        # ppl = sum(ppl_list) / len(ppl_list)
        # print(f"文件 {file_path} 平均PPL: {ppl:.4f}")
        
        # 异常值处理
        ppl_list = np.array(ppl_list)
        ppl_list = ppl_list[~np.isnan(ppl_list)]
        return sum(ppl_list) / len(ppl_list)
        
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        import traceback
        traceback.print_exc()
        return float('inf')

def batch_calculate_ppl(file_paths: List[str], model_name: str = "Qwen/Qwen3-8B-Base", device="cuda", remove_outliers: bool = False):
    """
    批量计算多个文件的平均PPL
    
    Args:
        file_paths: 文件路径列表
        model_name: 模型名称
        device: 设备
        remove_outliers: 是否移除异常值
    
    Returns:
        结果列表 (file_path, avg_ppl)
    """
    # 加载模型和分词器（只加载一次）
    tokenizer, model = load_model_and_tokenizer(model_name, device)
    
    results = []
    
    for file_path in file_paths:
        file_path = os.path.join(args.input_path, file_path)
        if os.path.exists(file_path):
            avg_ppl = calculate_file_ppl(file_path, tokenizer, model, remove_outliers=remove_outliers)
            results.append((file_path, avg_ppl))
            print(f"文件: {file_path} 平均PPL: {avg_ppl:.4f}")
        else:
            print(f"文件不存在: {file_path}")
            results.append((file_path, float('inf')))
    
    return results

# def get_files_from_directory(directory: str, extension: Optional[str] = None) -> List[str]:
#     """
#     获取目录下所有符合条件的文件
#     """
#     file_paths = []
    
#     for root, _, files in os.walk(directory):
#         for file in files:
#             if extension is None or file.endswith(extension):
#                 file_paths.append(os.path.join(root, file))
    
#     return file_paths

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量计算文件的平均PPL")
    parser.add_argument("--input_path", type=str, required=True, help="输入文件路径或目录")
    parser.add_argument("--model", type=str, default="<MODELS_DIR>/Qwen3-8B-Base", help="使用的模型名称")
    parser.add_argument("--device", type=str, default="cuda", help="使用的设备（cuda或cpu）")
    parser.add_argument("--outliers", action="store_true", dest="remove_outliers", help="apply the optional IQR outlier filter before averaging (off by default)")
    
    args = parser.parse_args()
    
    # 获取所有要处理的文件
    if os.path.isfile(args.input_path):
        file_paths = [args.input_path]
    elif os.path.isdir(args.input_path):
        file_paths = os.listdir(args.input_path)
        print(f"找到 {len(file_paths)} 个文件")
    else:
        print(f"输入路径不存在: {args.input_path}")
        exit(1)
    
    # 批量计算PPL
    results = batch_calculate_ppl(file_paths, args.model, args.device, remove_outliers=args.remove_outliers)
    # print()
    # 如果指定了输出文件，保存结果
    # if args.output:
    #     with open(args.output, 'w', encoding='utf-8') as f:
    #         f.write("文件路径,平均PPL\n")
    #         for file_path, avg_ppl in results:
    #             f.write(f"{file_path},{avg_ppl:.4f}\n")
    #     print(f"结果已保存到: {args.output}")
    
    # print("\n批量计算完成！")

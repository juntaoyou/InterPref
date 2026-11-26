import torch
import os
import json
import argparse
import math
parser = argparse.ArgumentParser()
# 定义目录路径
parser = argparse.ArgumentParser()
parser.add_argument("--neuron_dir", type=str, required=True, help="Directory for saving neuron results")
parser.add_argument("--out_dir", type=str, required=True, help="Directory for saving neuron results")
parser.add_argument("--top_percent", type=float, default=10.0, help="选择前N%的神经元，默认为5%")
parser.add_argument("--save_percent", action='store_true', default=False)
parser.add_argument("--delta",  type=float, default=0.1)
args = parser.parse_args()
out_dir = args.out_dir
directory = args.neuron_dir
delta = args.delta
os.makedirs(out_dir, exist_ok=True)
def read_and_convert_to_dict(file_path, num_lines, min_required=2000):
    result_dict = {}
    t = 0
    reached_minimum = False
    # delta = 0.1
    with open(file_path, 'r') as f:
        for idx in range(num_lines):
            line = f.readline().strip()
            if not line:
                break

            layer = eval(line,{"inf": math.inf, "Inf": math.inf, "INF": math.inf})[0]
            difference = eval(line,{"inf": math.inf, "Inf": math.inf, "INF": math.inf})[2]

            if idx < min_required or (idx >= min_required and difference >=delta):
                if layer not in result_dict:
                    result_dict[layer] = []
                result_dict[layer].append(eval(line,{"inf": math.inf, "Inf": math.inf, "INF": math.inf}))
                t += 1

            if t >= min_required and difference < delta:
                reached_minimum = True
                break

    if not reached_minimum:
        print(f"Warning: Not enough data with difference >= 0.1, total added: {t}")

    print(t, file_path)
    return result_dict

def save_dict_to_json(dict_obj, file_path):
    with open(file_path, 'w') as f:
        json.dump(dict_obj, f, indent=4)

# 获取前N%的神经元函数
def get_top_percent_neurons(tuples_list, percent=5):
    """
    从神经元列表中选择前N%的神经元
    
    参数:
    tuples_list: 包含神经元信息的元组列表，每个元组格式为(layer, position, difference, value, ...)
    percent: 要选择的前N%神经元，默认为5%
    
    返回:
    top_neurons: 包含前N%神经元信息的列表
    """
    # 确保列表不为空
    if not tuples_list:
        return []
    
    # 计算要选择的神经元数量
    num_total = len(tuples_list)
    num_to_select = max(1, int(num_total * percent / 100))
    
    print(f"从 {num_total} 个神经元中选择前 {percent}% ({num_to_select} 个) 神经元")
    
    # 元组已经按照差异值降序排列，直接取前N%即可
    top_neurons = tuples_list[:num_to_select]
    
    return top_neurons

# 从文件中读取数据并获取前N%的神经元
def process_top_percent_neurons(file_path, output_path, percent=5):
    """
    从文件中读取神经元数据，选择前N%的神经元并保存（同时保存为文本和JSON格式）
    
    参数:
    file_path: 包含排序后神经元数据的文件路径
    output_path: 保存前N%神经元的文件路径
    percent: 要选择的前N%神经元，默认为5%
    
    返回:
    top_neurons: 包含前N%神经元信息的列表
    """
    # 读取排序后的元组数据
    tuples = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = eval(line,{"inf": math.inf, "Inf": math.inf, "INF": math.inf})
            i = int(parts[0])
            j = int(parts[1])
            k = float(parts[2])
            l = float(parts[3])
            f = float(parts[4])
            tuple_data = (i, j, k, l,f) 
            tuples.append(tuple_data)
    
    # 获取前N%的神经元
    top_neurons = get_top_percent_neurons(tuples, percent)
    
    # 保存文本格式结果
    with open(output_path, 'w') as file:
        for tup in top_neurons:
            file.write(f"{tup}\n")
    
    # 将结果转换为字典格式（按层组织）
    top_neurons_dict = {}
    for neuron in top_neurons:
        layer = neuron[0]  # 第一个元素是层号
        if layer not in top_neurons_dict:
            top_neurons_dict[layer] = []
        # 将元组转换为列表以便JSON序列化
        top_neurons_dict[layer].append(list(neuron))
    
    # 构建JSON输出路径
    json_output_path = output_path.replace('.txt', '.json')
    
    # 保存为JSON文件
    save_dict_to_json(top_neurons_dict, json_output_path)
    
    print(f"前 {percent}% 的神经元已保存到 {output_path}")
    print(f"前 {percent}% 的神经元（字典格式）已保存到 {json_output_path}")
    
    return top_neurons
# 处理和保存差异
def process_and_save_differences(tensor1, tensor2, num_cols, save_path):
    differences = tensor1.view(-1) - tensor2.view(-1)
    sorted_differences, sorted_order = torch.sort(differences, descending=True)
    sorted_indices = sorted_order
    sorted_layer_positions = sorted_indices // num_cols
    sorted_col_positions = sorted_indices % num_cols
    corresponding_values = tensor1.view(-1)[sorted_indices]

    sorted_tuples = list(zip(
        sorted_layer_positions.tolist(),
        sorted_col_positions.tolist(),
        sorted_differences.tolist(),
        corresponding_values.tolist()
    ))

    # 将结果保存到文件，每行一个元组
    with open(save_path, 'w') as f:
        for item in sorted_tuples:
            f.write(f"{item}\n")

# 计算直方图的分位数
def calculate_quantiles(histogram, bins, quantiles=[0.95]):
    cumsum = torch.cumsum(histogram, dim=-1).float()
    total = cumsum[-1]
    results = []

    for q in quantiles:
        threshold = q * total
        index = torch.searchsorted(cumsum, threshold)
        results.append(bins[index.item()].item())

    return results

# 定义所有处理类型
preferences = [
    "Expertise_low.Informativeness_low", "Expertise_high.Informativeness_high",
    "Expertise_low.Informativeness_high", "Expertise_high.Informativeness_low",
    "Expertise_high.Style_high",
    "Expertise_high.Style_low",
    "Expertise_low.Style_low",
    "Expertise_low.Style_high",
    "Informativeness_low.Style_low",
    'Informativeness_high.Style_low',
    "Informativeness_low.Style_high",
    'Informativeness_high.Style_high',
#"Style_low.Expertise_low", "Style_low.Expertise_high", "Style_high.Expertise_low", "Style_high.Expertise_high",
#"Informativeness_low.Expertise_low", "Informativeness_high.Expertise_high","Informativeness_low.Expertise_high", "Informativeness_high.Expertise_low",
#"Style_low.Informativeness_low", "Style_low.Informativeness_high","Style_high.Informativeness_high", "Style_high.Informativeness_low"
]

# 处理每个类型的文件
for p in preferences:
    # 加载over_zero数据
    file = f'{directory}/{p}.pt'
    file_reversed = f'{directory}/None.pt'
    data = torch.load(file)
    data_reversed = torch.load(file_reversed)
    
    over_zero_prob = data['over_zero'] / data['token_num']
    over_zero_reversed_prob = data_reversed['over_zero'] / data_reversed['token_num']
    _, num_cols = over_zero_prob.shape

    # 处理并保存差值
    save_path = f'{out_dir}/{p}_sorted_differences_{delta}.txt'
    process_and_save_differences(over_zero_prob, over_zero_reversed_prob, num_cols, save_path)
    
    #save_path_reversed = f'{directory}/{p[1]}_sorted_differences.txt'
    #process_and_save_differences(over_zero_reversed_prob, over_zero_prob, num_cols, save_path_reversed)

    # print(f"results saved to {save_path} and {save_path_reversed}.")
    #for t in p:
    histograms_path = f'{directory}/{p}.pt'
    histograms_dict = torch.load(histograms_path)
    histograms = histograms_dict['histograms']

    # 定义直方图的bins
    bins = torch.cat([torch.tensor([-float('inf')]), torch.arange(0, 3.01, 0.01)])
    bins[-1] = float('inf')

    # 加载排序后的差异元组
    tuples_path = f'{out_dir}/{p}_sorted_differences_{delta}.txt'
    with open(tuples_path, 'r') as file:
        tuples = []
        for line in file:
            parts = line.strip().strip('()').split(', ')
            i = int(parts[0])
            j = int(parts[1])
            k = float(parts[2])
            l = float(parts[3])
            tuples.append((i, j, k, l))

    # 处理每个元组，计算分位数
    updated_tuples = []
    for i, j, k, l in tuples:
        histogram = histograms[i][j].cpu() 
        quantiles = calculate_quantiles(histogram, bins)
        updated_tuple = (i, j, k, l) + tuple(quantiles)
        updated_tuples.append(updated_tuple)

    # 保存更新后的元组到文件
    with open(tuples_path, 'w') as file:
        for tup in updated_tuples:
            file.write(f"{tup}\n")

    num_lines = 300000
    save_path = f'{out_dir}/{p}_sorted_differences_{delta}.txt'
    save_path_dict = f'{out_dir}/{p}_dict_{delta}.json'
    _dict = read_and_convert_to_dict(save_path, num_lines)
    save_dict_to_json(_dict, save_path_dict)
    if args.save_percent:
        # 处理并保存前N%的神经元
        print(f"\n处理 {p} 的前 {args.top_percent}% 神经元")
        top_neurons_output_path = f'{out_dir}/{p}_top_{args.top_percent}percent.txt'
        process_top_percent_neurons(save_path, top_neurons_output_path, args.top_percent)


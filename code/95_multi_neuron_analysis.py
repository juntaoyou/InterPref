import json
import torch
import argparse

# 解析命令行参数
parser = argparse.ArgumentParser(description='计算union_neuron和both_neuron的交并比')
# parser.add_argument('--neuron_dir', type=str, required=True, help='神经元数据目录')
parser.add_argument('--neuron_dir_multi', type=str, help='多神经元数据目录', default='<REPO_ROOT>/qwen/alpaca_for_dataset_compare')
parser.add_argument('--delta', type=float, default=4.0, help='delta值')

args = parser.parse_args()

# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义神经元数据目录
# neuron_dir = args.neuron_dir
neuron_dir_multi = args.neuron_dir_multi

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

preference_list = [
"Expertise_low.Informativeness_low", 
"Expertise_high.Informativeness_high",
"Expertise_low.Informativeness_high", 
"Expertise_high.Informativeness_low",

"Expertise_high.Style_high", 
"Expertise_high.Style_low", 
"Expertise_low.Style_low",
"Expertise_low.Style_high",    

"Informativeness_low.Style_low",
'Informativeness_high.Style_low',    
"Informativeness_low.Style_high",
'Informativeness_high.Style_high'
]
num_layers = 36
for preference in preference_list:
    preferences = preference.split(".")
    # if mode == 'two':
    # both_neuron_path=f'{neuron_dir_multi}/{preference}_top_{args.delta}percent.json'
    both_neuron_path=f'{neuron_dir_multi}/{preference}_top_{args.delta}percent.json'
    both_neuron=load_neuron_to_change(both_neuron_path)
    indices_both={}
    for layer in range(num_layers):
        if str(layer) in both_neuron:
            indices_both[str(layer)]=both_neuron[str(layer)][:, 1].long()
        # else: 
        #     indices_both[str(layer)]=torch.empty(0).long()
    # elif mode == 'union':
    union_neuron_path1=f'{neuron_dir_multi}/{preferences[0]}_top_{args.delta}percent.json'
    union_neuron1=load_neuron_to_change(union_neuron_path1)
    
    union_neuron_path2=f'{neuron_dir_multi}/{preferences[1]}_top_{args.delta}percent.json'
    union_neuron2=load_neuron_to_change(union_neuron_path2)
    indices_union={}
    for layer in range(num_layers):
        if str(layer) in union_neuron1 and str(layer) in union_neuron2:
            indices_union[str(layer)]=merge(union_neuron1[str(layer)][:, 1].long(), union_neuron2[str(layer)][:, 1].long())
        # else:
        #     indices_union[str(layer)]=torch.empty(0).long()
    
    # 计算交并比 (IoU)
    iou_results = {}
    for layer in range(num_layers):
        # 获取当前层的神经元集合
        both_set = set(indices_both[str(layer)].tolist()) if str(layer) in indices_both else set()
        union_set = set(indices_union[str(layer)].tolist()) if str(layer) in indices_union else set()
        
        # 计算交集和并集
        intersection = both_set.intersection(union_set)
        union_total = both_set.union(union_set)
        
        # 计算IoU
        if len(union_total) == 0:
            iou = 0.0
        else:
            iou = len(intersection) / len(union_total)
        
        iou_results[str(layer)] = iou
    
    # 打印结果
    print(f"Preference: {preference}")
    # for layer in range(num_layers):
    #     print(f"Layer {layer}: IoU = {iou_results[str(layer)]:.4f}")
    
    # 计算两个总的神经元集合的IoU
    all_both_neurons = set()
    all_union_neurons = set()
    
    # 将所有层的神经元合并到总的集合中
    for layer in range(num_layers):
        # 使用(layer, neuron_id)元组唯一标识神经元
        if str(layer) in indices_both:
            both_layer_neurons = {(layer, neuron_id) for neuron_id in indices_both[str(layer)].tolist()}
        else:
            both_layer_neurons = set()
        
        if str(layer) in indices_union:
            union_layer_neurons = {(layer, neuron_id) for neuron_id in indices_union[str(layer)].tolist()}
        else:
            union_layer_neurons = set()
        
        all_both_neurons.update(both_layer_neurons)
        all_union_neurons.update(union_layer_neurons)
    
    # 计算总的IoU
    intersection = all_both_neurons.intersection(all_union_neurons)
    union_total = all_both_neurons.union(all_union_neurons)
    
    if len(union_total) == 0:
        total_iou = 0.0
    else:
        total_iou = len(intersection) / len(union_total)
    
    print(f"Total IoU: {total_iou:.4f}")
    print("=" * 50)
    
    # elif mode == 'first':
    #     neuron_to_deactivate_path1=f'{neuron_dir}/{preferences[0]}_dict.json'
    #     neuron_to_deactivate1=load_neuron_to_change(neuron_to_deactivate_path1)

    # elif mode == 'second':
    #     neuron_to_deactivate_path2=f'{neuron_dir}/{preferences[1]}_dict.json'
    #     neuron_to_deactivate2=load_neuron_to_change(neuron_to_deactivate_path2)
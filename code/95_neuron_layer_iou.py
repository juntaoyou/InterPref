import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
import torch
# 设置英文字体
# font = FontProperties(family='Arial', size=12)

# 文件路径
max_value = 0
preference_list = ["Informativeness_low", "Informativeness_high", "Expertise_high", "Expertise_low", "Style_high", "Style_low"]
labels = ["Info-low", "Info-high", "Expertise-high", "Expertise-low", "Style-high", "Style-low"]
linestyles = ['-', '--', '-', '--', '-', '--']
colors = ['#d86c50', '#d86c50', '#638bd6', '#638bd6', '#46bc99', '#46bc99']
percents = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
# preference_list = ["Expertise_high.Informativeness_high", "Expertise_high.Informativeness_low", "Expertise_low.Informativeness_high", "Expertise_low.Informativeness_low", 
#                    'Expertise_high.Style_high', "Expertise_high.Style_low","Expertise_low.Style_high", "Expertise_low.Style_low",
#                    'Informativeness_high.Style_high', 'Informativeness_high.Style_low', 'Informativeness_low.Style_high', 'Informativeness_low.Style_low']
# preference_list = ["Expertise_high.Informativeness_high", "Expertise_high.Informativeness_low", "Expertise_low.Informativeness_high", "Expertise_low.Informativeness_low"]
# preference_list = ['Expertise_high.Style_high', "Expertise_high.Style_low","Expertise_low.Style_high", "Expertise_low.Style_low"]
# preference_list = ['Informativeness_high.Style_high', 'Informativeness_high.Style_low', 'Informativeness_low.Style_high', 'Informativeness_low.Style_low']
ious = [[] for _ in range(len(preference_list))]
fig, ax = plt.subplots(figsize=(12, 6))

for i, p in enumerate(preference_list):
    for percent in percents:
        # p1, p2 = preferences.split('.')
        file1 = f'<REPO_ROOT>/qwen/psoups_for_dataset_compare/{p}_top_{percent}percent.json'
        file2 = f'<REPO_ROOT>/qwen/mmlu_for_dataset_compare/{p}_top_{percent}percent.json'

        # 读取JSON文件
        def load_neuron_to_change(neuron_to_change_path):
            with open(neuron_to_change_path, 'r', encoding='utf-8') as file:
                neuron_to_change = json.loads(file.read())
                for t in neuron_to_change:
                    neuron_to_change[t]=torch.tensor(neuron_to_change[t])
            return neuron_to_change


        # 计算两个集合的IOU
        def calculate_iou(set1, set2):
            intersection = set1.intersection(set2)
            union = set1.union(set2)
            if len(union) == 0:
                return 0.0
            return len(intersection) / len(union)

        # 加载数据
        high_data = load_neuron_to_change(file1)
        low_data = load_neuron_to_change(file2)

        # 提取所有层
        layers = list(set(high_data.keys()) | set(low_data.keys()))
        max_layer = max([int(x) for x in layers]) + 1
        all_layers = range(max_layer)
        # 计算每层的IOU
        iou_values = []
        all_high_neurons, all_low_neurons = set(), set()
        for layer in all_layers:
            if str(layer) in high_data and str(layer) in low_data:
        #         # 提取神经元索引（子列表的第二个元素）
                high_neurons = set(high_data[str(layer)][:, 1].long().tolist())
                all_high_neurons.update((layer, n) for n in high_neurons)
        #         # print(high_neurons)
                low_neurons = set(low_data[str(layer)][:, 1].long().tolist())
                all_low_neurons.update((layer, n) for n in low_neurons)
        #         iou = calculate_iou(high_neurons, low_neurons)
        #         iou_values.append(iou)
        #     else:
        #         # 如果某一层在其中一个文件中不存在，则IOU为0
        #         iou_values.append(0.0)
                
        total_iou = calculate_iou(all_high_neurons, all_low_neurons)

        # 绘制折线图
        
        # ax.plot(range(len(all_layers)), iou_values, marker='o', linestyle='-', linewidth=3, markersize=6, label=p)
        # max_value = max(max_value, max(iou_values))
        ious[i].append(total_iou)
    # 设置图表标题和轴标签
    # plt.title('Layer-wise IOU between Expertise_high and Expertise_low Neuron Sets', fontproperties=font, fontsize=16)
ax.set_xlabel('neuron fraction (%)', fontsize=20)
ax.set_ylabel('IOU across datasets', fontsize=20)

    # 设置x轴刻度和标签
for i in range(len(preference_list)):
    ax.plot(percents, ious[i], marker='o', linestyle=linestyles[i], color=colors[i], linewidth=3, markersize=6, label=preference_list[i])
ax.set_xticks(percents)
ax.set_xticklabels(map(str, percents), fontsize=16)
ax.tick_params(axis='both', labelsize=16)
ax.set_ylim(0, 1.0)
fig_legend, ax_legend = plt.subplots(1, 1, figsize=(6, 2))
ax_legend.axis('off')
handles, _ = ax.get_legend_handles_labels()

legend = ax_legend.legend(handles, labels, loc='upper center', ncol=3, fontsize=20)
bbox = legend.get_window_extent(fig_legend.canvas.get_renderer())
bbox = bbox.transformed(fig_legend.dpi_scale_trans.inverted())# --- 4. 保存单独的图例图片 ---
# dpi=300 保证清晰度
# bbox_inches=bbox 确保只裁剪图例区域
plt.savefig(f'isolated_legend_cross_dataset_stability.png', dpi=300, bbox_inches=bbox, transparent=False) 
plt.savefig(f'isolated_legend_cross_dataset_stability.pdf', dpi=300, bbox_inches=bbox, transparent=False)
plt.close(fig_legend)
# ax.legend(loc='best', fontsize=20)
    # 添加网格线
# plt.grid(True, linestyle='--', alpha=0.7)

    # 调整布局
plt.tight_layout()

    # 保存图表
# output_file =   
plt.savefig(f'cross_dataset_stability.png', dpi=300, bbox_inches='tight', format='png')
plt.savefig(f'cross_dataset_stability.pdf', dpi=300, bbox_inches='tight', format='pdf')
import json
with open(f'cross_dataset_stability.json', 'w') as f:
    json.dump({preference_list[i]: sum(ious[i]) / len(ious[i]) for i in range(len(preference_list))}, f, indent=4)
    # print(f"Layer-wise IOU plot saved to {output_file}")

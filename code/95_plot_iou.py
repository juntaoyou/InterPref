import json
import matplotlib.pyplot as plt
import os
import numpy as np

# 准备数据集大小列表
dataset_sizes = list(range(1000, 10000, 1000))

# 准备颜色列表
colors = plt.cm.viridis(np.linspace(0, 1, len(dataset_sizes)))

# 定义所有偏好类型
preferences = ['Expertise_high', 'Expertise_low', 'Informativeness_high', 'Informativeness_low', "Style_high", "Style_low"]

# 创建一个大图，包含2行3列的子图
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(24, 16))
# fig.suptitle('Stacked IOU Charts for All Preferences', fontsize=24, y=0.95)

# 展平axes数组以便循环访问
axes = axes.flatten()

# 遍历所有偏好类型
for i, preference_type in enumerate(preferences):
    all_data = {}
    all_percent_list = None
    
    # 获取当前子图
    ax = axes[i]
    
    # 加载所有数据集的数据
    for dataset_size in dataset_sizes:
        # 构建JSON文件路径
        json_file_path = f'<REPO_ROOT>/qwen/alpaca_figures_{dataset_size}/iou.json'
        
        # 检查文件是否存在
        if not os.path.exists(json_file_path):
            print(f"Warning: {json_file_path} does not exist, skipping...")
            continue
        
        # 读取JSON文件
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 提取数据
        percent_list = []
        overall_iou_list = []
        
        for item in data[preference_type]:
            percent_list.append(item["percent"])
            overall_iou_list.append(item["overall_iou"])
        
        all_data[dataset_size] = {
            'percent': percent_list,
            'iou': overall_iou_list
        }
        
        # 确保所有数据集的percent列表相同
        if all_percent_list is None:
            all_percent_list = percent_list
        else:
            if all_percent_list != percent_list:
                print(f"Warning: Percent lists don't match for dataset {dataset_size} in {preference_type}")
    
    # 检查是否有数据
    if all_data and all_percent_list:
        # 设置堆叠柱状图参数
        bar_width = 0.5  # 柱子宽度
        offsets = np.arange(len(all_percent_list))  # 位置偏移
        
        # 初始化累积高度
        cumulative_height = np.zeros(len(all_percent_list))
        
        # 绘制堆叠柱状图
        for j, (dataset_size, data) in enumerate(all_data.items()):
            # 绘制当前数据集的柱子
            ax.bar(offsets, data['iou'] - cumulative_height, width=bar_width, 
                  bottom=cumulative_height, color=colors[j], label=f'{dataset_size}')
            
            # 更新累积高度
            cumulative_height = np.array(data['iou'])
        
        # 设置子图标题和坐标轴标签
        ax.set_title(preference_type, fontsize=16)
        ax.set_xlabel('Percent', fontsize=14)
        ax.set_ylabel('Overall IOU', fontsize=14)
        
        # 设置x轴刻度和标签
        ax.set_xticks(offsets)
        ax.set_xticklabels([f'{p}%' for p in all_percent_list])
        
        # 设置y轴范围
        ax.set_ylim(0.8, 1.0)
        
        # 添加网格线
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 绘制所有数据集的图例（只在第一个子图创建，其他子图复用）
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
    else:
        ax.text(0.5, 0.5, f'No data available for\n{preference_type}', ha='center', va='center', fontsize=14)
        ax.set_title(preference_type, fontsize=16)

# 调整布局
plt.tight_layout(rect=[0, 0.05, 1, 0.92])

# 添加全局图例
fig.legend(handles, labels, loc='lower center', ncol=len(dataset_sizes), fontsize=12, title='Dataset Size')

# 保存包含所有堆叠柱状图的大图
output_file_all_stacked = '<REPO_ROOT>/qwen/figures/all_preferences_stacked_iou_chart.png'
plt.savefig(output_file_all_stacked, dpi=300, bbox_inches='tight')

print(f"All stacked charts saved to: {output_file_all_stacked}")
plt.legend(loc='lower center', bbox_to_anchor=(0.5, 0), fontsize=12, title='Dataset Size')
# 显示图表
plt.show()
print("All charts generated successfully!")
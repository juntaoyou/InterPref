import json
import numpy as np
from scipy import stats
import argparse
import datetime
import os

def load_scores_from_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def calculate_t_test(dist1, dist2):
    t_stat, p_value = stats.ttest_rel(dist1, dist2)
    return t_stat, p_value

def main():
    parser = argparse.ArgumentParser(description='计算两个分布的t-test和p值')
    parser.add_argument('--file1', type=str, required=True, help='第一个JSON文件路径')
    parser.add_argument('--file2', type=str, default=None, help='第二个JSON文件路径（可选，如不提供则比较同一文件的前两次运行）')
    parser.add_argument('--output-dir', type=str, default='t-test-res-files', help='输出目录路径')
    args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    
    data1 = load_scores_from_json(args.file1)
    
    if args.file2:
        data2 = load_scores_from_json(args.file2)
        dist1 = np.array(data1['scores'][0]['per_query_scores'][0])
        dist2 = np.array(data2['scores'][0]['per_query_scores'][0])
        print(f"比较 {args.file1} 和 {args.file2} 的第一次运行分数")
    else:
        per_query_scores = data1['scores'][0]['per_query_scores']
        if len(per_query_scores) < 2:
            print("错误：同一文件至少需要两次运行数据才能进行比较")
            return
        dist1 = np.array(per_query_scores[0])
        dist2 = np.array(per_query_scores[1])
        print(f"比较 {args.file1} 的第一次和第二次运行分数")

    print(f"\n分布1（第一次运行）: {dist1}")
    print(f"分布2（第二次运行）: {dist2}")
    print(f"\n分布1均值: {np.mean(dist1):.4f}, 标准差: {np.std(dist1):.4f}")
    print(f"分布2均值: {np.mean(dist2):.4f}, 标准差: {np.std(dist2):.4f}")

    t_stat, p_value = calculate_t_test(dist1, dist2)
    
    print(f"\n=== t检验结果 ===")
    print(f"t值: {t_stat:.4f}")
    print(f"p值: {p_value:.4f}")
    
    is_significant = bool(p_value < 0.05)
    conclusion = "两个分布存在显著差异（p < 0.05）" if is_significant else "两个分布无显著差异（p >= 0.05）"
    print(f"结论：{conclusion}")
    
    result = {
        "timestamp": timestamp,
        "file1": args.file1,
        "file2": args.file2,
        "dist1": dist1.tolist(),
        "dist2": dist2.tolist(),
        "mean1": float(np.mean(dist1)),
        "std1": float(np.std(dist1)),
        "mean2": float(np.mean(dist2)),
        "std2": float(np.std(dist2)),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "is_significant": is_significant,
        "conclusion": conclusion
    }
    
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    output_filename = f"t_test_result_{timestamp}.json"
    output_path = os.path.join(output_dir, output_filename)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    
    print(f"\n结果已保存至: {output_path}")

if __name__ == "__main__":
    main()

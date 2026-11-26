"""
比较 Delta Cosine Similarity 与 occurence.py 的 Multi-Single 一致性
===============================================================

occurence.py 的数据:
  - conflict / non-conflict 偏好对
  - 每个偏好有 Single Prompt 和 Multiple Prompt 的 LLM Score
  - "interaction" = Multi - Single

我们的分析:
  - delta = prob(preference) - prob(None)  [12288 维向量]
  - delta cosine similarity = 两个偏好 delta 向量的余弦相似度

比较问题:
  - delta cosine similarity 高的偏好对 (变化方向一致) → interaction 小?
  - delta cosine similarity 低的偏好对 (变化方向相反) → interaction 大?
"""

import torch
import numpy as np
import sys
from pathlib import Path

# ── 路径 ──
DATA_DIR = Path("<REPO_ROOT>/res/qwen/mmlu_neuron_results")
OCCUR_PATH = Path("<REPO_ROOT>/cFigures/bishe/occurence.py")

PREFERENCES = [
    "None",
    "Expertise_high", "Expertise_low",
    "Informativeness_high", "Informativeness_low",
    "Style_high", "Style_low",
]

# 偏好名称映射: 我们的命名 → occurence.py 的命名
NAME_MAP = {
    "Expertise_high": "Expert-high",
    "Expertise_low": "Expert-low",
    "Informativeness_high": "Info-high",
    "Informativeness_low": "Info-low",
    "Style_high": "Style-high",
    "Style_low": "Style-low",
}

# occurence.py 中的偏好对数据 (从文件中提取)
# Conflict pairs + Non-conflict pairs
# 格式: (cat1_name, cat1_single, cat2_single, cat1_multi, cat2_multi)
PAIR_DATA = [
    # --- conflict pairs ---
    ("Expert-high", 4.7217, "Info-low", 3.0149, 4.0248, 3.6932),
    ("Expert-low", 2.5292, "Info-high", 4.5366, 2.8770, 3.9776),
    ("Info-high", 4.6326, "Style-low", 1.5478, 4.0236, 2.4956),
    ("Expert-high", 4.7217, "Style-high", 4.6783, 3.7056, 4.7702),
    ("Expert-high", 4.7217, "Style-low", 1.5478, 3.9963, 1.4807),
    ("Expert-low", 2.5292, "Style-low", 1.5478, 2.2472, 2.6546),
    ("Info-low", 3.0612, "Style-low", 1.5478, 2.3751, 2.1205),
    ("Info-low", 3.0612, "Style-high", 4.6783, 3.9130, 4.6087),
    # --- non-conflict pairs ---
    ("Expert-high", 4.7217, "Info-high", 4.5366, 4.7192, 4.8484),
    ("Expert-low", 2.5292, "Info-low", 3.0149, 2.6087, 2.9950),
    ("Expert-low", 2.5292, "Style-high", 4.6783, 2.1851, 4.9354),
    ("Info-high", 4.6326, "Style-high", 4.6783, 4.7093, 4.8944),
]

# 反向映射: occurence.py 命名 → 我们的命名
REV_MAP = {v: k for k, v in NAME_MAP.items()}


def load_delta_cosine_sim(data_dir: Path, layer: int):
    """加载指定层的 delta 余弦相似度."""
    probs = {}
    for name in PREFERENCES:
        path = data_dir / f"{name}.pt"
        data = torch.load(path, map_location="cpu", weights_only=True)
        prob = data["over_zero"].float() / data["token_num"]
        probs[name] = prob

    base = probs["None"][layer].numpy()
    delta_names = [n for n in PREFERENCES if n != "None"]
    deltas = {}
    for n in delta_names:
        deltas[n] = probs[n][layer].numpy() - base

    # 计算所有两两余弦相似度
    n = len(delta_names)
    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            vi, vj = deltas[delta_names[i]], deltas[delta_names[j]]
            sim[i, j] = sim[j, i] = np.dot(vi, vj) / (np.linalg.norm(vi) * np.linalg.norm(vj) + 1e-12)

    return sim, delta_names


def compute_delta_cosine_for_pair(n1, n2, sim, names):
    """从相似度矩阵中取指定偏好对的 delta cosine similarity."""
    idx1 = names.index(REV_MAP[n1])
    idx2 = names.index(REV_MAP[n2])
    return sim[idx1, idx2]


# (compute_interaction 已内联到 pair_info 列表中)


def main():
    # ── 1. 获取 delta cosine similarity ──────────────────────────
    print("=" * 80)
    print("  Delta Cosine Similarity vs Multi-Single Interaction")
    print("=" * 80)

    # 分析所有层，按层比较
    layers_to_check = [0, 4, 8, 12, 15, 18, 22, 26, 30, 35]

    # 每个 pair 的信息
    pair_info = []
    for i, pair in enumerate(PAIR_DATA):
        cat1, s1, cat2, s2, m1, m2 = pair
        ptype = "conflict" if i < 8 else "non-conflict"
        pair_info.append({
            "pair_id": i,
            "cat1": cat1, "cat2": cat2,
            "type": ptype,
            "single1": s1, "single2": s2,
            "multi1": m1, "multi2": m2,
            "diff1": m1 - s1,
            "diff2": m2 - s2,
            "abs_diff_sum": abs(m1 - s1) + abs(m2 - s2),
            "interaction_mag": abs((m1 - s1) - (m2 - s2)),
        })

    # 打印 occurence.py 的数据
    print(f"\n{'─'*80}")
    print("  occurence.py 原始数据")
    print(f"{'─'*80}")
    print(f"{'ID':>3s}  {'Type':<12s}  {'Cat1':<16s}  {'Cat2':<16s}  "
          f"{'S1':>7s}  {'S2':>7s}  {'M1':>7s}  {'M2':>7s}  "
          f"{'D1':>7s}  {'D2':>7s}  {'|Δ|sum':>7s}")
    print("  " + "-" * 110)
    for p in pair_info:
        print(f"{p['pair_id']:>3d}  {p['type']:<12s}  {p['cat1']:<16s}  {p['cat2']:<16s}  "
              f"{p['single1']:>7.4f}  {p['single2']:>7.4f}  "
              f"{p['multi1']:>7.4f}  {p['multi2']:>7.4f}  "
              f"{p['diff1']:>+7.4f}  {p['diff2']:>+7.4f}  "
              f"{p['abs_diff_sum']:>7.4f}")

    # ── 2. 按层比较 ──────────────────────────────────────────────
    print(f"\n{'='*110}")
    print("  Delta Cosine Similarity vs Multi-Single Interaction (跨层)")
    print(f"{'='*110}")

    for layer in layers_to_check:
        sim, delta_names = load_delta_cosine_sim(DATA_DIR, layer)

        print(f"\n  ── Layer {layer:2d} ──────────────────────────────")
        print(f"{'ID':>3s}  {'Pair':<36s}  {'Type':<12s}  {'Delta Cos Sim':>14s}  "
              f"{'|Δ|sum':>8s}  {'int_mag':>8s}")
        print("  " + "-" * 85)

        for p in pair_info:
            n1, n2 = p["cat1"], p["cat2"]
            try:
                dcos = compute_delta_cosine_for_pair(n1, n2, sim, delta_names)
            except (ValueError, KeyError):
                dcos = float("nan")

            pair_label = f"{n1} vs {n2}"
            print(f"{p['pair_id']:>3d}  {pair_label:<36s}  {p['type']:<12s}  "
                  f"{dcos:>14.6f}  {p['abs_diff_sum']:>8.4f}  {p['interaction_mag']:>8.4f}")

    # ── 3. Spearman 相关性 ───────────────────────────────────────
    print(f"\n{'='*80}")
    print("  Spearman 相关性: Delta Cosine Similarity vs Interaction")
    print(f"{'='*80}")

    try:
        from scipy.stats import spearmanr
    except ImportError:
        print("  scipy 未安装, 跳过相关性计算")
        sys.exit(0)

    # 按层和偏好对做相关性
    for layer in layers_to_check:
        sim, delta_names = load_delta_cosine_sim(DATA_DIR, layer)

        dcos_values = []
        abs_diff_values = []
        int_mag_values = []

        for p in pair_info:
            n1, n2 = p["cat1"], p["cat2"]
            try:
                dcos = compute_delta_cosine_for_pair(n1, n2, sim, delta_names)
            except (ValueError, KeyError):
                continue
            dcos_values.append(dcos)
            abs_diff_values.append(p["abs_diff_sum"])
            int_mag_values.append(p["interaction_mag"])

        r1, p1 = spearmanr(dcos_values, abs_diff_values)
        r2, p2 = spearmanr(dcos_values, int_mag_values)
        print(f"  Layer {layer:2d}: "
              f"ρ(delta_cos, |Δ|sum) = {r1:>+7.4f} (p={p1:.4f})  "
              f"ρ(delta_cos, int_mag) = {r2:>+7.4f} (p={p2:.4f})")

    # ── 4. 总结性对比: Layer 15 ──────────────────────────────────
    print(f"\n{'='*80}")
    print("  Layer 15 详细对比 (按 |Δ|sum 排序)")
    print(f"{'='*80}")

    layer = 15
    sim, delta_names = load_delta_cosine_sim(DATA_DIR, layer)

    rows = []
    for p in pair_info:
        n1, n2 = p["cat1"], p["cat2"]
        try:
            dcos = compute_delta_cosine_for_pair(n1, n2, sim, delta_names)
        except (ValueError, KeyError):
            continue
        rows.append((dcos, p["abs_diff_sum"], p["interaction_mag"], p["type"], n1, n2))

    # 按 |Δ|sum 排序
    rows.sort(key=lambda r: r[1], reverse=True)

    print(f"{'Pair':<36s}  {'Type':<12s}  {'Delta Cos Sim':>14s}  "
          f"{'|Δ|sum':>8s}  {'int_mag':>8s}")
    print("  " + "-" * 82)
    for dcos, ads, imag, ptype, n1, n2 in rows:
        print(f"{n1} vs {n2:<28s}  {ptype:<12s}  {dcos:>14.6f}  {ads:>8.4f}  {imag:>8.4f}")


if __name__ == "__main__":
    main()

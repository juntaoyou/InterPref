"""
比较 Raw Cosine, Delta Cosine 与 occurence.py 的 Multi-Single Interaction
========================================================================

三种向量:
  1. raw prob = over_zero / token_num                    (原始激活概率)
  2. delta    = prob(pref) - prob(None)                 (减去None)
  3. zscore   = (prob - mean(prob)) / std(prob)         (标准化)

比较它们与 occurence.py 的交互效应 (|Δ|sum = |Multi - Single|₁ + |Multi - Single|₂) 的一致性。
"""

import torch
import numpy as np
from pathlib import Path

# ── 路径 ──
DATA_DIR = Path("<REPO_ROOT>/res/qwen/mmlu_neuron_results")

PREFERENCES = [
    "None",
    "Expertise_high", "Expertise_low",
    "Informativeness_high", "Informativeness_low",
    "Style_high", "Style_low",
]

NAME_MAP = {
    "Expertise_high": "Expert-high",
    "Expertise_low": "Expert-low",
    "Informativeness_high": "Info-high",
    "Informativeness_low": "Info-low",
    "Style_high": "Style-high",
    "Style_low": "Style-low",
}
REV_MAP = {v: k for k, v in NAME_MAP.items()}

# occurence.py 的偏好对数据 (12对)
PAIR_DATA = [
    # (cat1_name, cat1_single, cat2_name, cat2_single, cat1_multi, cat2_multi)
    # --- conflict pairs (8) ---
    ("Expert-high", 4.7217, "Info-low",   3.0149, 4.0248, 3.6932),
    ("Expert-low",  2.5292, "Info-high",  4.5366, 2.8770, 3.9776),
    ("Info-high",   4.6326, "Style-low",  1.5478, 4.0236, 2.4956),
    ("Expert-high", 4.7217, "Style-high", 4.6783, 3.7056, 4.7702),
    ("Expert-high", 4.7217, "Style-low",  1.5478, 3.9963, 1.4807),
    ("Expert-low",  2.5292, "Style-low",  1.5478, 2.2472, 2.6546),
    ("Info-low",    3.0612, "Style-low",  1.5478, 2.3751, 2.1205),
    ("Info-low",    3.0612, "Style-high", 4.6783, 3.9130, 4.6087),
    # --- non-conflict pairs (4) ---
    ("Expert-high", 4.7217, "Info-high",  4.5366, 4.7192, 4.8484),
    ("Expert-low",  2.5292, "Info-low",   3.0149, 2.6087, 2.9950),
    ("Expert-low",  2.5292, "Style-high", 4.6783, 2.1851, 4.9354),
    ("Info-high",   4.6326, "Style-high", 4.6783, 4.7093, 4.8944),
]

NUM_LAYERS = 36


def load_probs(data_dir):
    """加载所有偏好的激活概率."""
    probs = {}
    for name in PREFERENCES:
        d = torch.load(data_dir / f"{name}.pt", map_location="cpu", weights_only=True)
        probs[name] = d["over_zero"].float() / d["token_num"]
    return probs


def cos_sim_vec(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)


def compute_sim_matrix(probs, names, layer, mode):
    """
    mode:
      'raw'    — prob 直接算 cos
      'delta'  — prob - prob(None) 算 cos
      'zscore' — (prob - μ) / σ 算 cos  (逐神经元标准化)
    """
    if mode == "raw":
        vectors = [probs[n][layer].numpy() for n in names]
    elif mode == "delta":
        base = probs["None"][layer].numpy()
        vectors = [(probs[n][layer].numpy() - base) for n in names]
    elif mode == "zscore":
        vectors = []
        for n in names:
            v = probs[n][layer].numpy()
            vectors.append((v - v.mean()) / (v.std() + 1e-12))
    else:
        raise ValueError(f"unknown mode: {mode}")

    k = len(names)
    sim = np.eye(k)
    for i in range(k):
        for j in range(i + 1, k):
            s = cos_sim_vec(vectors[i], vectors[j])
            sim[i, j] = sim[j, i] = s
    return sim, names


def spearman_r(x, y):
    n = len(x)
    rx = np.argsort(np.argsort(np.array(x)))
    ry = np.argsort(np.argsort(np.array(y)))
    d = rx - ry
    return 1 - 6 * np.sum(d ** 2) / (n * (n ** 2 - 1))


def main():
    probs = load_probs(DATA_DIR)

    # 6个非None偏好
    pref_names = [n for n in PREFERENCES if n != "None"]

    # 构造 pair_info 列表
    pair_info = []
    for i, pair in enumerate(PAIR_DATA):
        c1, s1, c2, s2, m1, m2 = pair
        pair_info.append({
            "id": i,
            "n1": REV_MAP[c1], "n2": REV_MAP[c2],
            "label": f"{c1} vs {c2}",
            "type": "conflict" if i < 8 else "non-conflict",
            "abs_diff_sum": abs(m1 - s1) + abs(m2 - s2),
        })

    layers_show = [0, 4, 8, 12, 15, 18, 22, 26, 30, 35]

    print("=" * 120)
    print("  Raw Cosine / Delta Cosine / Zscore Cosine  vs  Occurence Interaction")
    print("=" * 120)

    # ── 每个模式的跨层 Spearman ────────────────────────────────
    for mode_name in ["raw", "delta", "zscore"]:
        print(f"\n{'─'*80}")
        print(f"  模式: {mode_name}")
        print(f"{'─'*80}")

        print(f"\n  Spearman ρ(cos_sim, |Δ|sum) 按层:")
        print(f"  {'Layer':>6s}  {'ρ':>8s}")
        print("  " + "-" * 18)

        all_rhos = []
        for layer in range(NUM_LAYERS):
            sim, _ = compute_sim_matrix(probs, pref_names, layer, mode_name)
            cos_vals = []
            abs_vals = []
            for p in pair_info:
                i = pref_names.index(p["n1"])
                j = pref_names.index(p["n2"])
                cos_vals.append(sim[i, j])
                abs_vals.append(p["abs_diff_sum"])
            r = spearman_r(cos_vals, abs_vals)
            all_rhos.append(r)
            if layer in layers_show:
                print(f"  {layer:>6d}  {r:>+8.4f}")

        avg_r = np.mean(all_rhos)
        std_r = np.std(all_rhos)
        max_r_layer = np.argmax(np.abs(all_rhos))
        print(f"  {'平均':>6s}  {avg_r:>+8.4f} ± {std_r:.4f}")
        print(f"  {'最佳层':>6s}  Layer {max_r_layer} (ρ={all_rhos[max_r_layer]:+.4f})")

    # ── 展示一个代表性层 (Layer 15) 的三种对比 ─────────────────
    print(f"\n\n{'='*120}")
    print("  Layer 15 详细对比: Raw vs Delta vs Zscore vs |Δ|sum")
    print(f"{'='*120}")

    raw_sim, _ = compute_sim_matrix(probs, pref_names, 15, "raw")
    delta_sim, _ = compute_sim_matrix(probs, pref_names, 15, "delta")
    zscore_sim, _ = compute_sim_matrix(probs, pref_names, 15, "zscore")

    print(f"\n{'Pair':<42s}  {'Type':<12s}  {'Raw Cos':>10s}  {'Delta Cos':>10s}  {'Zscore Cos':>11s}  {'|Δ|sum':>8s}")
    print("  " + "-" * 97)
    for p in pair_info:
        i = pref_names.index(p["n1"])
        j = pref_names.index(p["n2"])
        rc = raw_sim[i, j]
        dc = delta_sim[i, j]
        zc = zscore_sim[i, j]
        print(f"  {p['label']:<40s}  {p['type']:<12s}  {rc:>+10.6f}  {dc:>+10.6f}  {zc:>+10.6f}  {p['abs_diff_sum']:>8.4f}")

    # ── 相关性对比表 ───────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  三种模式 Spearman ρ 对比 (各层)")
    print(f"{'='*80}")
    print(f"{'Layer':>6s}  {'Raw ρ':>10s}  {'Delta ρ':>10s}  {'Zscore ρ':>11s}")
    print("  " + "-" * 42)

    for layer in layers_show:
        raw_sim, _ = compute_sim_matrix(probs, pref_names, layer, "raw")
        delta_sim, _ = compute_sim_matrix(probs, pref_names, layer, "delta")
        zscore_sim, _ = compute_sim_matrix(probs, pref_names, layer, "zscore")

        raw_vals, delta_vals, zscore_vals, abs_vals = [], [], [], []
        for p in pair_info:
            i, j = pref_names.index(p["n1"]), pref_names.index(p["n2"])
            raw_vals.append(raw_sim[i, j])
            delta_vals.append(delta_sim[i, j])
            zscore_vals.append(zscore_sim[i, j])
            abs_vals.append(p["abs_diff_sum"])

        print(f"  {layer:>6d}  {spearman_r(raw_vals, abs_vals):>+10.4f}  "
              f"{spearman_r(delta_vals, abs_vals):>+10.4f}  "
              f"{spearman_r(zscore_vals, abs_vals):>+11.4f}")

    avg_raw = np.mean([spearman_r(
        [compute_sim_matrix(probs, pref_names, l, "raw")[0][pref_names.index(p["n1"]), pref_names.index(p["n2"])] for p in pair_info],
        [p["abs_diff_sum"] for p in pair_info]) for l in range(NUM_LAYERS)])
    avg_delta = np.mean([spearman_r(
        [compute_sim_matrix(probs, pref_names, l, "delta")[0][pref_names.index(p["n1"]), pref_names.index(p["n2"])] for p in pair_info],
        [p["abs_diff_sum"] for p in pair_info]) for l in range(NUM_LAYERS)])
    avg_zscore = np.mean([spearman_r(
        [compute_sim_matrix(probs, pref_names, l, "zscore")[0][pref_names.index(p["n1"]), pref_names.index(p["n2"])] for p in pair_info],
        [p["abs_diff_sum"] for p in pair_info]) for l in range(NUM_LAYERS)])

    print(f"  {'平均':>6s}  {avg_raw:>+10.4f}  {avg_delta:>+10.3f}  {avg_zscore:>+11.4f}")


if __name__ == "__main__":
    main()

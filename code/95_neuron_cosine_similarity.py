"""
计算偏好激活概率(减去None后)的 Cosine Similarity
===============================================

激活概率 = over_zero / token_num
Delta = 偏好激活概率 - None激活概率
对6个偏好的delta向量计算层内的余弦相似度

用法: python 95_neuron_cosine_similarity.py [--layer LAYER]
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import numpy as np

# ── 配置 ──────────────────────────────────────────────────────────
DATA_DIR = Path("<REPO_ROOT>/res/qwen/mmlu_neuron_results")

PREFERENCES = [
    "None",
    "Expertise_high", "Expertise_low",
    "Informativeness_high", "Informativeness_low",
    "Style_high", "Style_low",
]

NUM_LAYERS = 36
NUM_NEURONS = 12288


def load_activation_prob(data_dir: Path) -> dict[str, torch.Tensor]:
    """加载所有偏好文件的激活概率矩阵."""
    probs = {}
    for name in PREFERENCES:
        path = data_dir / f"{name}.pt"
        if not path.exists():
            print(f"  [警告] 文件不存在: {path}", file=sys.stderr)
            continue
        data = torch.load(path, map_location="cpu", weights_only=True)
        token_num = data["token_num"]
        over_zero = data["over_zero"].float()
        prob = over_zero / token_num
        probs[name] = prob
        print(f"  ✓ {name:>20s}: token_num={token_num:.0f}, "
              f"prob mean={prob.mean():.6f}")
    return probs


def cosine_similarity_matrix(vectors: list[np.ndarray]) -> np.ndarray:
    """计算向量列表两两之间的余弦相似度矩阵."""
    n = len(vectors)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            vi = vectors[i]
            vj = vectors[j]
            nrm = np.linalg.norm(vi) * np.linalg.norm(vj)
            matrix[i, j] = matrix[j, i] = np.dot(vi, vj) / (nrm + 1e-12)
    return matrix


def format_matrix(names: list[str], matrix: np.ndarray) -> str:
    """将相似度矩阵格式化为表格字符串."""
    col_w = 22
    lines = []
    header = f"{'':>{col_w}s}"
    for n in names:
        header += f"  {n:>{col_w-2}s}"
    lines.append(header)
    lines.append("  " + "-" * (col_w + col_w * len(names)))
    for i, ni in enumerate(names):
        row = f"{ni:>{col_w}s}"
        for j in range(len(names)):
            row += f"  {matrix[i, j]:>{col_w-2}.6f}"
        lines.append(row)
    return "\n".join(lines)


def layer_delta_analysis(probs: dict[str, torch.Tensor], layer: int, out_dir: Path):
    """
    分析指定层: 计算6个偏好delta向量的余弦相似度.
    """
    base = probs["None"][layer].numpy()
    delta_names = [n for n in PREFERENCES if n != "None"]
    delta_vectors = [(probs[n][layer].numpy() - base) for n in delta_names]

    # 余弦相似度
    sim = cosine_similarity_matrix(delta_vectors)

    # ── 打印表格 ──────────────────────────────────────────────
    print(f"\n{'='*90}")
    print(f"  Layer {layer:2d}  |  6个偏好 delta向量 (激活概率 - None) 的 Cosine Similarity")
    print(f"{'='*90}")
    print(format_matrix(delta_names, sim))

    # ── 变化幅度统计 ───────────────────────────────────────────
    print(f"\n  ── 各偏好 vs None 的神经元变化幅度 ──")
    for n in delta_names:
        delta = probs[n][layer].numpy() - base
        nonzero = delta[delta != 0]
        print(f"  {n:>20s}: nonzero={len(nonzero):>6d}/{NUM_NEURONS}, "
              f"  max_up={delta.max():+.6f}, max_down={delta.min():+.6f},  "
              f"|delta|_mean={np.abs(nonzero).mean():.6f}")

    # ── 保存 ──────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"delta_cosine_sim_layer_{layer:02d}.npy", sim)
    with open(out_dir / f"delta_cosine_sim_layer_{layer:02d}.txt", "w") as f:
        f.write(f"Layer {layer}  |  Delta (偏好-None) Cosine Similarity\n\n")
        f.write(format_matrix(delta_names, sim))
    print(f"\n  📁 已保存: {out_dir}/delta_cosine_sim_layer_{layer:02d}.txt")

    # ── Heatmap ───────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        short = [n.replace("Informativeness", "Info").replace("Expertise", "Exp")
                    .replace("_", "\n") for n in delta_names]
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(sim, vmin=-1, vmax=1, cmap="RdYlBu_r")
        ax.set_xticks(range(len(delta_names)))
        ax.set_yticks(range(len(delta_names)))
        ax.set_xticklabels(short, fontsize=8)
        ax.set_yticklabels(short, fontsize=8)
        ax.set_title(f"Layer {layer} — Delta Cosine Similarity\n(activation prob - None)", fontsize=12)

        for i in range(len(delta_names)):
            for j in range(len(delta_names)):
                color = "white" if abs(sim[i, j]) > 0.6 else "black"
                ax.text(j, i, f"{sim[i, j]:.3f}", ha="center", va="center", fontsize=7, color=color)

        fig.colorbar(im, ax=ax, shrink=0.85)
        fig.tight_layout()
        fig.savefig(str(out_dir / f"delta_heatmap_layer_{layer:02d}.png"), dpi=200)
        plt.close(fig)
        print(f"  📊 Heatmap: {out_dir}/delta_heatmap_layer_{layer:02d}.png")
    except ImportError:
        pass

    return sim, delta_names


def all_layers_delta_summary(probs: dict[str, torch.Tensor], out_dir: Path):
    """
    计算所有层: 6个偏好delta向量余弦相似度的跨层趋势.
    """
    delta_names = [n for n in PREFERENCES if n != "None"]
    k = len(delta_names)
    base_name = "None"

    # delta_all[layer, i, j]
    delta_all = np.zeros((NUM_LAYERS, k, k))
    for layer in range(NUM_LAYERS):
        base = probs[base_name][layer].numpy()
        vectors = [(probs[n][layer].numpy() - base) for n in delta_names]
        delta_all[layer] = cosine_similarity_matrix(vectors)

    # ── 打印层 × 关键对比 ─────────────────────────────────────
    key_pairs = [
        ("Expertise_high", "Expertise_low", "Expertise H vs L"),
        ("Informativeness_high", "Informativeness_low", "Informativeness H vs L"),
        ("Style_high", "Style_low", "Style H vs L"),
        ("Expertise_high", "Informativeness_high", "Exp H vs Info H"),
        ("Expertise_high", "Style_high", "Exp H vs Style H"),
        ("Expertise_low", "Informativeness_low", "Exp L vs Info L"),
        ("Expertise_low", "Style_low", "Exp L vs Style L"),
        ("Informativeness_high", "Style_high", "Info H vs Style H"),
        ("Informativeness_low", "Style_low", "Info L vs Style L"),
    ]

    print(f"\n{'='*120}")
    print("  跨层趋势: 各偏好对 delta 余弦相似度 (选层展示)")
    print(f"{'='*120}")
    header = f"{'Layer':>6s}"
    for _, _, label in key_pairs:
        header += f"  {label:>24s}"
    print(header)
    print("  " + "-" * (6 + 26 * len(key_pairs)))
    for layer in [0, 4, 8, 12, 15, 18, 22, 26, 30, 35]:
        row = f"  {layer:>4d}"
        for n1, n2, _ in key_pairs:
            i, j = delta_names.index(n1), delta_names.index(n2)
            row += f"  {delta_all[layer, i, j]:>24.6f}"
        print(row)

    # ── 保存 ──────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "delta_cosine_sim_all_layers.npy", delta_all)

    # ── 绘图 ──────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

        # (a) 同维度高-低对比
        ax = axes[0]
        for n1, n2, label in key_pairs[:3]:
            i, j = delta_names.index(n1), delta_names.index(n2)
            ax.plot(range(NUM_LAYERS), delta_all[:, i, j], label=label, lw=1.8)
        ax.axhline(y=0, color="gray", ls="--", alpha=0.4)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Cosine Similarity")
        ax.set_title("Same dimension: High vs Low")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

        # (b) 跨维度 high
        ax = axes[1]
        for n1, n2, label in [(k[0], k[1], k[2]) for k in key_pairs if " H " in k[2] and "vs" in k[2]]:
            i, j = delta_names.index(n1), delta_names.index(n2)
            ax.plot(range(NUM_LAYERS), delta_all[:, i, j], label=label, lw=1.8)
        ax.axhline(y=0, color="gray", ls="--", alpha=0.4)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Cosine Similarity")
        ax.set_title("Cross-dimension: High")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

        # (c) 跨维度 low
        ax = axes[2]
        for n1, n2, label in [(k[0], k[1], k[2]) for k in key_pairs if " L " in k[2] and "vs" in k[2]]:
            i, j = delta_names.index(n1), delta_names.index(n2)
            ax.plot(range(NUM_LAYERS), delta_all[:, i, j], label=label, lw=1.8)
        ax.axhline(y=0, color="gray", ls="--", alpha=0.4)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Cosine Similarity")
        ax.set_title("Cross-dimension: Low")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

        fig.suptitle("Delta Cosine Similarity Across Layers\n(activation prob - None, for 6 preferences)", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(str(out_dir / "delta_cosine_sim_across_layers.png"), dpi=200)
        plt.close(fig)
        print(f"\n  📊 跨层趋势图: {out_dir}/delta_cosine_sim_across_layers.png")
    except ImportError:
        print("\n  (matplotlib 未安装，跳过绘图)")

    return delta_all, delta_names


def main():
    parser = argparse.ArgumentParser(description="计算6个偏好delta向量 (激活概率-None) 的 Cosine Similarity")
    parser.add_argument("--layer", type=int, default=None, help="指定层 (0-35), 不指定则全量分析")
    args = parser.parse_args()

    out_dir = Path("<REPO_ROOT>/res/neuron_cosine_sim")

    print("=" * 80)
    print("  Delta Cosine Similarity 分析")
    print("  (偏好激活概率 - None激活概率) × 6偏好")
    print("  数据:", DATA_DIR)
    print("=" * 80)

    print("\n📥 加载数据...")
    probs = load_activation_prob(DATA_DIR)

    if args.layer is not None:
        layer_delta_analysis(probs, args.layer, out_dir)
    else:
        print(f"\n📊 计算全部 {NUM_LAYERS} 层...")
        all_layers_delta_summary(probs, out_dir)
        # 额外展示两个代表性层
        for demo in [15, 30]:
            layer_delta_analysis(probs, demo, out_dir)


if __name__ == "__main__":
    main()

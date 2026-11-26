#!/usr/bin/env python3
"""Compute Pearson r between primary judge (Qwen2.5-72B) and alternative judge (Llama-3.1-72B).

Usage:
    python 85_compute_judge_robustness.py \
        --primary  <path>/eval_results/mmlu_test \
        --alternative <path>/eval_results/mmlu_test_robustness \
        --out_path <path>/robustness_results.json
"""

import argparse
import json
import os
import re
from scipy.stats import pearsonr

PRINCIPLES = ["expert", "fluency", "informative", "style"]

def load_scores(eval_dir):
    """Load all .ci_3runs.json files from eval_dir, return {filename: {principle: mean}}."""
    scores = {}
    for principle in PRINCIPLES:
        principle_dir = os.path.join(eval_dir, principle)
        if not os.path.isdir(principle_dir):
            continue
        for fname in os.listdir(principle_dir):
            if not fname.endswith(".ci_3runs.json"):
                continue
            fpath = os.path.join(principle_dir, fname)
            with open(fpath) as f:
                data = json.load(f)
            mean = data["statistics"]["mean"]
            # Use filename without principle-specific suffix as key
            # e.g. "qwen_Expertise_high.ci_3runs.json" -> key
            key = fname.replace(".ci_3runs.json", "")
            if key not in scores:
                scores[key] = {}
            scores[key][principle] = mean
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", required=True, help="Primary judge results dir")
    parser.add_argument("--alternative", required=True, help="Alternative judge results dir")
    parser.add_argument("--out_path", default="robustness_results.json")
    args = parser.parse_args()

    primary_scores = load_scores(args.primary)
    alt_scores = load_scores(args.alternative)

    common_keys = sorted(set(primary_scores) & set(alt_scores))
    print(f"Primary judge:   {len(primary_scores)} files")
    print(f"Alternative judge: {len(alt_scores)} files")
    print(f"Common files:    {len(common_keys)}")

    results = {}
    for principle in PRINCIPLES:
        pairs = []
        for key in common_keys:
            if principle in primary_scores[key] and principle in alt_scores[key]:
                pairs.append((primary_scores[key][principle], alt_scores[key][principle]))

        if len(pairs) < 3:
            print(f"  {principle}: insufficient pairs ({len(pairs)})")
            continue

        p_scores, a_scores = zip(*pairs)
        r, p_val = pearsonr(p_scores, a_scores)
        results[principle] = {
            "pearson_r": round(r, 4),
            "p_value": p_val,
            "n": len(pairs),
        }
        print(f"  {principle}: r={r:.4f}, p={p_val:.6f}, n={len(pairs)}")

    # Self-consistency: within primary judge, check agreement across runs
    # Use the std/sem from ci_3runs files as a proxy for repeated-call consistency
    print("\n--- Self-consistency (primary judge run-level agreement) ---")
    within_r_values = []
    for principle in PRINCIPLES:
        principle_dir = os.path.join(args.primary, principle)
        if not os.path.isdir(principle_dir):
            continue
        for fname in os.listdir(principle_dir):
            if not fname.endswith(".ci_3runs.json"):
                continue
            fpath = os.path.join(principle_dir, fname)
            with open(fpath) as f:
                data = json.load(f)
            run_vals = data["statistics"].get("all_mean_values", [])
            if len(run_vals) >= 2:
                # Use mean / std as consistency signal
                cv = data["statistics"]["std"] / max(data["statistics"]["mean"], 1e-8)
                within_r_values.append(1.0 - min(cv * 10, 1.0))  # heuristic

    if within_r_values:
        avg_within = sum(within_r_values) / len(within_r_values)
        results["self_consistency"] = {
            "pearson_r": round(avg_within, 4),
            "n_files": len(within_r_values),
        }
        print(f"  Self-consistency (run-level agreement, heuristic): r ≈ {avg_within:.4f}")
    else:
        print("  No run-level data found for self-consistency")

    with open(args.out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.out_path}")


if __name__ == "__main__":
    main()

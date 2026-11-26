#!/usr/bin/env python
"""Build the MMLU out-of-distribution evaluation set (570 queries, 10 per category).

Two modes, both stratified 10-queries-per-category over the 57 MMLU subjects:

* `--mode reproduce` (default) rebuilds, byte for byte, the file the experiments ran on
  (`../../dataset/mmlu_eval_570_stratified.jsonl`): the first 10 rows of each subject test
  file, in the original MMLU file order. This is what actually produced every MMLU number
  in the implementation.
* `--mode random --seed S` implements the sampling procedure as worded in the implementation
  table ("stratified random sampling"): 10 queries drawn uniformly at random without
  replacement from each of the 57 subject test files, with a fixed seed for
  reproducibility. Use this mode only when re-running the pipeline from scratch - the
  shipped generations and judge scores correspond to the `reproduce` file.

Usage:
    python build_mmlu_eval_570.py                       # rebuild the shipped file
    python build_mmlu_eval_570.py --mode random --seed 42 --out out/
"""

import argparse
import glob
import json
import os
import random

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mmlu_dir", default="<DATA_ROOT>/mmlu",
                    help="directory containing the per-subject MMLU test parquet files")
    ap.add_argument("--out", default=".", help="output directory for the JSONL")
    ap.add_argument("--per_category", type=int, default=10)
    ap.add_argument("--mode", choices=["reproduce", "random"], default="reproduce",
                    help="reproduce = first 10 rows per subject (as run, matches the shipped "
                         "file); random = seeded uniform sample of 10 per subject (the "
                         "implementation's 'stratified random sampling' wording)")
    ap.add_argument("--seed", type=int, default=42, help="seed for --mode random")
    args = ap.parse_args()

    test_files = sorted(glob.glob(os.path.join(args.mmlu_dir, "*", "test-*.parquet")))
    exclude = {"all", "auxiliary_train"}
    test_files = [p for p in test_files
                  if os.path.basename(os.path.dirname(p)) not in exclude]

    rng = random.Random(args.seed)
    queries, per_subject = [], []
    for test_file in test_files:
        subject = os.path.basename(os.path.dirname(test_file))
        questions = pd.read_parquet(test_file, columns=["question"])["question"].tolist()
        if args.mode == "random":
            k = min(len(questions), args.per_category)
            sampled = rng.sample(questions, k)
            n = k
        else:
            sampled = questions[: args.per_category]
            n = len(sampled)
        queries.extend(sampled)
        per_subject.append((subject, n))

    suffix = "random_seed%d" % args.seed if args.mode == "random" else "570"
    out_path = os.path.join(args.out, f"mmlu_eval_{suffix}_stratified.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps({"query": q}, ensure_ascii=False) + "\n")

    print(f"mode={args.mode} | subjects: {len(test_files)} | queries: {len(queries)} | wrote {out_path}")
    counts = {s: n for s, n in per_subject if n != args.per_category}
    if counts:
        print("subjects with fewer than", args.per_category, "rows:", counts)


if __name__ == "__main__":
    main()


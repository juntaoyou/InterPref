#!/usr/bin/env python
"""Verify the four shipped datasets.

Checks performed
----------------
1. psoups_localization_2525   2,525 rows, all unique.
2. psoups_eval_koala_49       49 rows, all unique, disjoint from the localization pool.
3. mmlu_localization_2525     2,525 rows; every query occurs in the MMLU auxiliary-train set
                              (the MMLU "training set"); dev/validation/test untouched.
4. mmlu_eval_570_stratified   570 rows; reconstructible as exactly the first 10 rows of each
                              of the 57 MMLU subject test files, i.e. 10 queries per category.

Checks 3 and 4 need the upstream MMLU parquet files; pass their directory with --mmlu_dir
(default: <DATA_ROOT>/mmlu). Check 1 optionally cross-references the raw Personalized
Soups training parquet when --psoups_dir is given (default: <DATA_ROOT>/pSoups).

Usage:
    python verify_datasets.py [--here DIR] [--mmlu_dir DIR] [--psoups_dir DIR]
"""

import argparse
import csv
import glob
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))


def read_jsonl(name):
    path = os.path.join(HERE, name)
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line)["query"])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mmlu_dir", default="<DATA_ROOT>/mmlu")
    ap.add_argument("--psoups_dir", default="<DATA_ROOT>/pSoups")
    args = ap.parse_args()

    failures = []

    def check(label, ok, detail=""):
        print(f"[{'PASS' if ok else 'FAIL'}] {label}{(' - ' + detail) if detail else ''}")
        if not ok:
            failures.append(label)

    # ---- 1. Psoups localization -------------------------------------------
    loc = read_jsonl("psoups_localization_2525.jsonl")
    check("Psoups localization has 2,525 rows", len(loc) == 2525, f"n={len(loc)}")
    check("Psoups localization queries are unique", len(set(loc)) == len(loc),
          f"unique={len(set(loc))}")

    # ---- 2. Psoups (Koala) evaluation -------------------------------------
    ev = read_jsonl("psoups_eval_koala_49.jsonl")
    check("Psoups eval has 49 rows", len(ev) == 49, f"n={len(ev)}")
    check("Psoups eval queries are unique", len(set(ev)) == len(ev))
    check("Psoups eval disjoint from localization pool", not (set(ev) & set(loc)))

    # optional provenance check against the raw Personalized Soups train set
    psoups_parquet = os.path.join(args.psoups_dir, "data", "train-00000-of-00001.parquet")
    if os.path.exists(psoups_parquet):
        import pyarrow.parquet as pq
        raw = pq.read_table(psoups_parquet, columns=["user_input"]).to_pandas()["user_input"]
        in_raw = sum(q in set(raw.str[: len(q)]) for q in loc[:200])
        check("Psoups localization queries appear in the raw Psoups train set (spot check 200)",
              in_raw == 200, f"{in_raw}/200")

    # ---- 3. MMLU localization ---------------------------------------------
    try:
        import pyarrow.parquet as pq
        have_mmlu = True
    except Exception:
        have_mmlu = False
        print("[SKIP] pyarrow not available - MMLU provenance checks skipped")

    mmlu_loc = read_jsonl("mmlu_localization_2525.jsonl")
    check("MMLU localization has 2,525 rows", len(mmlu_loc) == 2525, f"n={len(mmlu_loc)}")

    if have_mmlu and os.path.isdir(args.mmlu_dir):
        aux_path = os.path.join(args.mmlu_dir, "all", "auxiliary_train-00000-of-00001.parquet")
        if os.path.exists(aux_path):
            aux = set(pq.read_table(aux_path, columns=["question"]).to_pandas()["question"])
            matched = sum(q in aux for q in mmlu_loc)
            check("MMLU localization drawn from the MMLU training set (auxiliary_train)",
                  matched == len(mmlu_loc), f"{matched}/{len(mmlu_loc)} matched")
        else:
            print(f"[SKIP] auxiliary_train parquet not found under {args.mmlu_dir}")
        for split in ("dev", "validation", "test"):
            p = os.path.join(args.mmlu_dir, "all", f"{split}-00000-of-00001.parquet")
            if os.path.exists(p):
                q = set(pq.read_table(p, columns=["question"]).to_pandas()["question"])
                check(f"MMLU localization does not touch the {split} split",
                      not (set(mmlu_loc) & q))

    # ---- 4. MMLU evaluation (stratified) ----------------------------------
    mmlu_ev = read_jsonl("mmlu_eval_570_stratified.jsonl")
    check("MMLU eval has 570 rows", len(mmlu_ev) == 570, f"n={len(mmlu_ev)}")

    subject_files = sorted(glob.glob(os.path.join(args.mmlu_dir, "*", "test-*.parquet"))) \
        if have_mmlu and os.path.isdir(args.mmlu_dir) else []
    subject_files = [p for p in subject_files
                     if os.path.basename(os.path.dirname(p)) not in ("all", "auxiliary_train")]
    if len(subject_files) == 57:
        recon = Counter()
        for p in subject_files:
            qs = pq.read_table(p, columns=["question"]).to_pandas()["question"].head(10)
            recon.update(qs.tolist())
        check("MMLU eval = first 10 rows of each of the 57 subject test files "
              "(10 queries per category, 57 categories)",
              recon == Counter(mmlu_ev),
              f"{len(subject_files)} subject files; identical multiset={recon == Counter(mmlu_ev)}")
    else:
        print(f"[SKIP] expected 57 subject test parquet files under {args.mmlu_dir}, "
              f"found {len(subject_files)} - stratification check skipped")

    print()
    if failures:
        print(f"RESULT: {len(failures)} check(s) FAILED: {failures}")
        return 1
    print("RESULT: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
Evaluate multi-dimensional steering results in with_preferences_partial1/2
using 80_llm_score_with_ci.py.

For each JSONL file, runs the LLM judge on all applicable principles:
  - Expertise.Informativeness/  → expert, informative
  - Expertise.Style/            → expert, style
  - Informativeness.Style/      → informative, style
Plus fluency for all.

Usage:
  python 80_eval_with_preferences.py [--model MODEL_PATH] [--batch_size 4]
                                  [--num_runs 3] [--gpu 0]

The script creates isolated temp directories (one file per run) so that
80_llm_score_with_ci.py's "process all files in dir" behavior works correctly.
"""

import os
import sys
import json
import subprocess
import argparse
import tempfile
import shutil
import datetime
from pathlib import Path

# ──────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────
CODE_DIR = "<REPO_ROOT>/code"
EVAL_SCRIPT = os.path.join(CODE_DIR, "80_llm_score_with_ci.py")
BASE_INPUT_DIRS = [
    "<REPO_ROOT>/qwen/with_preferences_partial1",
    "<REPO_ROOT>/qwen/with_preferences_partial2",
]

# For each pair subdirectory, which principles to evaluate
PAIR_PRINCIPLES = {
    "Expertise.Informativeness": ["expert", "informative"],
    "Expertise.Style":           ["expert", "style"],
    "Informativeness.Style":     ["informative", "style"],
}

# Extra principles evaluated for ALL files
EXTRA_PRINCIPLES = ["fluency"]

# Default output root
DEFAULT_OUT_ROOT = "<REPO_ROOT>/code/LLM_results/with_preferences"

# ──────────────────────────────────────────────────────────
def collect_files():
    """Return list of (pair_dir, file_path, basename) for all JSONL files."""
    items = []
    for base_dir in BASE_INPUT_DIRS:
        part_name = os.path.basename(base_dir)  # e.g. with_preferences_partial1
        for pair_name in sorted(os.listdir(base_dir)):
            pair_path = os.path.join(base_dir, pair_name)
            if not os.path.isdir(pair_path):
                continue
            for fname in sorted(os.listdir(pair_path)):
                if not fname.endswith(".json"):
                    continue
                fpath = os.path.join(pair_path, fname)
                items.append((pair_name, fpath, fname, part_name))
    return items


def get_principles(pair_name):
    """Return list of all principles to evaluate for a given pair directory."""
    principles = list(PAIR_PRINCIPLES.get(pair_name, []))
    # Always add fluency
    for ep in EXTRA_PRINCIPLES:
        if ep not in principles:
            principles.append(ep)
    return principles


def run_evaluation(file_path, principle, out_dir, model_path, batch_size, num_runs, gpu_id):
    """
    Run 80_llm_score_with_ci.py for a single file + principle.
    Creates a temporary directory containing only this file.
    """
    # Create output dir for this principle
    principle_out = os.path.join(out_dir, principle)
    os.makedirs(principle_out, exist_ok=True)

    # Create a temp directory with a symlink/hardlink to the file
    # (so the script doesn't see other files in the same folder)
    with tempfile.TemporaryDirectory(prefix=f"eval_{principle}_") as tmpdir:
        tmp_file = os.path.join(tmpdir, os.path.basename(file_path))
        # Hardlink if possible, else copy
        try:
            os.link(file_path, tmp_file)
        except OSError:
            shutil.copy2(file_path, tmp_file)

        env = os.environ.copy()
        if gpu_id is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        cmd = [
            sys.executable, EVAL_SCRIPT,
            "--model", model_path,
            "--input_path", tmpdir,
            "--out_path", principle_out,
            "--principle", principle,
            "--batch_size", str(batch_size),
            "--num_runs", str(num_runs),
        ]

        print(f"\n{'='*70}")
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] "
              f"Evaluating: {os.path.basename(file_path)}")
        print(f"  Principle: {principle}")
        print(f"  Command: {' '.join(cmd)}")
        print(f"{'='*70}")

        result = subprocess.run(cmd, env=env, cwd=CODE_DIR,
                                capture_output=False, text=True)

        if result.returncode != 0:
            print(f"  ⚠ ERROR (code {result.returncode}) for {file_path} / {principle}")
            if result.stderr:
                print(f"  stderr: {result.stderr[:500]}")
        else:
            print(f"  ✓ Done: {os.path.basename(file_path)} / {principle}")

        return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate multi-dimensional steering results with LLM judge."
    )
    parser.add_argument("--model", type=str,
                        default="<MODELS_DIR>/Qwen2.5-72B-Instruct-AWQ",
                        help="Judge model path")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_runs", type=int, default=3,
                        help="Number of evaluation runs per file (for CI)")
    parser.add_argument("--gpu", type=int, default=None,
                        help="GPU device ID (default: let CUDA_VISIBLE_DEVICES decide)")
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT_ROOT,
                        help="Root output directory")
    parser.add_argument("--dry_run", action="store_true",
                        help="Only list what would be run, don't execute")
    args = parser.parse_args()

    files = collect_files()
    print(f"Found {len(files)} JSONL files to evaluate across "
          f"{len(BASE_INPUT_DIRS)} input directories.\n")

    # Group by file to show the plan
    print("Evaluation plan:")
    print(f"{'File':50s} {'Principles':30s} {'Source':25s}")
    print("-" * 105)
    for pair_name, fpath, fname, part_name in files:
        principles = get_principles(pair_name)
        print(f"{fname:50s} {', '.join(principles):30s} {pair_name:25s}")
    print()

    if args.dry_run:
        print("Dry run mode. Not executing.")
        return

    total = len(files)
    for idx, (pair_name, fpath, fname, part_name) in enumerate(files, 1):
        principles = get_principles(pair_name)
        fpart = part_name.replace("with_preferences_", "")  # partial1 / partial2
        # Output: out_dir/partial1/Expertise.Informativeness/...
        out_subdir = os.path.join(args.out_dir, fpart, pair_name)
        os.makedirs(out_subdir, exist_ok=True)

        print(f"\n{'#'*70}")
        print(f"# [{idx}/{total}] {fname}")
        print(f"# Source: {pair_name} ({part_name})")
        print(f"# Principles: {principles}")
        print(f"# Output: {out_subdir}")
        print(f"{'#'*70}")

        for principle in principles:
            run_evaluation(
                fpath, principle, out_subdir,
                args.model, args.batch_size, args.num_runs, args.gpu,
            )

    print(f"\n{'='*70}")
    print(f"All evaluations complete! Results in: {args.out_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

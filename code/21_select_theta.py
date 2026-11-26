#!/usr/bin/env python
"""Select the retained neuron fraction theta from a dose-response sweep.

The sweep CSV (`Condition, Metric, fraction %, Value`) tracks, for every
preference condition and every deactivated fraction rho of the preference-neuron set,
both the preference score (LLM Score) and the fluency score. The rho = 0 row of the
Fluency metric is the un-intervened baseline. This script:

  1. reads the sweep,
  2. for each preference condition finds the smallest rho > 0 whose fluency score is
     within `--tolerance` (default 5 %) of the rho = 0 fluency baseline, and
  3. prints the selected theta per condition, together with the fluency at that point
     and the corresponding preference-score deviation from the neutral reference.

Usage:
    python 21_select_theta.py --sweep /path/to/sweep.csv [--tolerance 0.05]
"""

import argparse
import csv
import os
import sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True,
                    help="dose-response CSV (Condition, Metric, fraction %, Value)")
    ap.add_argument("--tolerance", type=float, default=0.05,
                    help="maximum relative fluency drop vs the un-intervened baseline "
                         "(default 0.05)")
    args = ap.parse_args()

    if not os.path.exists(args.sweep):
        print(f"sweep file not found: {args.sweep}", file=sys.stderr)
        return 1

    with open(args.sweep, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    def fraction(row):
        key = [k for k in row if "fraction" in k][0]
        return float(row[key])

    sweep = {}
    last_cond = None   # each condition's LLM-Score block is followed by its own (blank-labelled) fluency block
    for row in rows:
        cond = (row["Preference condition"] or "").strip()
        metric = row["Metric"].strip()
        rho = fraction(row)
        if cond and cond.lower() != "nan":
            last_cond = cond
            sweep.setdefault(cond, {}).setdefault(metric, {})[rho] = float(row["Value"])
        elif last_cond is not None:
            sweep.setdefault(last_cond, {}).setdefault(metric, {})[rho] = float(row["Value"])

    def fluency_curve(cond):
        return sweep.get(cond, {}).get("Fluency", {})

    print(f"{'condition':<22} {'theta (%)':>9} {'fluency':>8} {'baseline':>9} "
          f"{'rel. change':>11} {'|pref. dev.|':>12}")
    selected = {}
    for cond in sorted(sweep):
        flu = fluency_curve(cond)
        sco = sweep[cond].get("LLM Score", {})
        if not flu or 0.0 not in flu:
            print(f"{cond:<22}   (no fluency baseline in sweep - skipped)")
            continue
        base = flu[0.0]
        limit = base * (1.0 - args.tolerance)
        theta, fluency = None, None
        for rho in sorted(r for r in flu if r > 0):
            if flu[rho] >= limit:
                theta, fluency = rho, flu[rho]
                break
        if theta is None:
            print(f"{cond:<22}   (no rho within {args.tolerance:.0%} tolerance - skipped)")
            continue
        dev = abs(sco.get(theta, float("nan")) - sco.get(0.0, float("nan"))) \
            if 0.0 in sco else float("nan")
        selected[cond] = theta
        rel = (fluency - base) / base
        print(f"{cond:<22} {theta:>9.1f} {fluency:>8.3f} {base:>9.3f} "
              f"{rel:>+11.1%} {dev:>12.3f}")

    print(f"\ncriterion: smallest swept fraction whose fluency is within "
          f"{args.tolerance:.0%} of the un-intervened (rho = 0) baseline")
    print("selected:", ", ".join(f"{c}={t:g}%" for c, t in selected.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())

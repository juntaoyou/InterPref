#!/bin/bash
# Orchestrator: wait for generation (60_with_preferences_generation_multi.py) and gemma eval (80_llm_score_with_ci.py)
# to finish, then run 81_qwen_eval_0818.sh and copy results to "<REPO_ROOT>"/results/0818
LOG="<REPO_ROOT>"/code/logs/qwen_eval_0818_orchestrator.log
echo "=== orchestrator started $(date) ===" > "$LOG"

echo "[$(date +%H:%M:%S)] waiting for generation job (60_with_preferences_generation_multi.py)..." >> "$LOG"
while pgrep -f "with_preferences_generation_multi[.]py" >/dev/null; do sleep 60; done
echo "[$(date +%H:%M:%S)] generation finished" >> "$LOG"

echo "[$(date +%H:%M:%S)] waiting for gemma eval job (80_llm_score_with_ci.py)..." >> "$LOG"
while pgrep -f "llm_score_with_ci[.]py" >/dev/null; do sleep 60; done
echo "[$(date +%H:%M:%S)] gemma eval finished" >> "$LOG"

# snapshot existing qwen result dirs
ls "<REPO_ROOT>"/code/t-test_results_qwen_mmlu/ 2>/dev/null | grep '^res_' | sort > /tmp/qwen_eval_before.txt || true

echo "[$(date +%H:%M:%S)] starting qwen eval..." >> "$LOG"
bash "<REPO_ROOT>"/code/81_qwen_eval_0818.sh >> "$LOG" 2>&1
echo "[$(date +%H:%M:%S)] qwen eval script exit: $?" >> "$LOG"

# copy new qwen result dirs to results/0818
ls "<REPO_ROOT>"/code/t-test_results_qwen_mmlu/ 2>/dev/null | grep '^res_' | sort > /tmp/qwen_eval_after.txt || true
comm -13 /tmp/qwen_eval_before.txt /tmp/qwen_eval_after.txt 2>/dev/null | while read d; do
  cp -r "<REPO_ROOT>/code/t-test_results_qwen_mmlu/$d" "<REPO_ROOT>"/results/0818/
  echo "copied: $d" >> "$LOG"
done
echo "=== results/0818 content ===" >> "$LOG"
ls -la "<REPO_ROOT>"/results/0818/ >> "$LOG"
echo "=== orchestrator done $(date) ===" >> "$LOG"

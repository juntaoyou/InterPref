#!/bin/bash
# 0818 multi-model orchestrator: llama gen -> gemma gen -> llama eval -> gemma eval -> copy results
LOG="<REPO_ROOT>"/code/logs/multimodel_0818_orchestrator.log
echo "=== orchestrator started $(date) ===" > "$LOG"

# snapshot existing result dirs for both models
ls "<REPO_ROOT>"/code/t-test_results_llama_mmlu/ 2>/dev/null | grep '^res_' | sort > /tmp/llama_eval_before.txt || true
ls "<REPO_ROOT>"/code/t-test_results_gemma_mmlu/ 2>/dev/null | grep '^res_' | sort > /tmp/gemma_eval_before.txt || true

echo "[$(date +%H:%M:%S)] === llama generation ===" >> "$LOG"
bash "<REPO_ROOT>"/code/90_gen_llm_0818.sh llama "<MODELS_DIR>"/Llama-3.1-8B-Instruct "<REPO_ROOT>"/eval/llama/configs >> "$LOG" 2>&1
echo "[$(date +%H:%M:%S)] llama gen exit: $?" >> "$LOG"

echo "[$(date +%H:%M:%S)] === gemma generation ===" >> "$LOG"
bash "<REPO_ROOT>"/code/90_gen_llm_0818.sh gemma "<MODELS_DIR>"/gemma-2-9b-it "<REPO_ROOT>"/eval/gemma/configs >> "$LOG" 2>&1
echo "[$(date +%H:%M:%S)] gemma gen exit: $?" >> "$LOG"

echo "[$(date +%H:%M:%S)] === llama evaluation ===" >> "$LOG"
bash "<REPO_ROOT>"/code/90_eval_llm_0818.sh llama >> "$LOG" 2>&1
echo "[$(date +%H:%M:%S)] llama eval exit: $?" >> "$LOG"

echo "[$(date +%H:%M:%S)] === gemma evaluation ===" >> "$LOG"
bash "<REPO_ROOT>"/code/90_eval_llm_0818.sh gemma >> "$LOG" 2>&1
echo "[$(date +%H:%M:%S)] gemma eval exit: $?" >> "$LOG"

# copy new result dirs to results/0818
ls "<REPO_ROOT>"/code/t-test_results_llama_mmlu/ 2>/dev/null | grep '^res_' | sort > /tmp/llama_eval_after.txt || true
comm -13 /tmp/llama_eval_before.txt /tmp/llama_eval_after.txt 2>/dev/null | while read d; do
  cp -r "<REPO_ROOT>/code/t-test_results_llama_mmlu/$d" "<REPO_ROOT>"/results/0818/
  echo "copied llama: $d" >> "$LOG"
done
ls "<REPO_ROOT>"/code/t-test_results_gemma_mmlu/ 2>/dev/null | grep '^res_' | sort > /tmp/gemma_eval_after.txt || true
comm -13 /tmp/gemma_eval_before.txt /tmp/gemma_eval_after.txt 2>/dev/null | while read d; do
  cp -r "<REPO_ROOT>/code/t-test_results_gemma_mmlu/$d" "<REPO_ROOT>"/results/0818/
  echo "copied gemma: $d" >> "$LOG"
done
echo "=== results/0818 content ===" >> "$LOG"
ls -la "<REPO_ROOT>"/results/0818/ >> "$LOG"
echo "=== orchestrator done $(date) ===" >> "$LOG"

#!/bin/bash
# ============================================================
# 依赖安装脚本 — 80_llm_score_with_ci.py + bitsandbytes
# 环境: Python 3.12, CUDA 11.8, 清华镜像源, uv
# ============================================================
set -euo pipefail

UV="uv"
PYTHON="python3.12"
PIP_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"
TORCH_INDEX="https://download.pytorch.org/whl/cu118"

# 确认 uv 和 Python 版本
$UV --version
$PYTHON --version

echo ""
echo "=== 1. 安装 PyTorch (CUDA 11.8) ==="
$UV pip install torch==2.4.0 torchvision==0.19.0 \
    --index-url $TORCH_INDEX

echo ""
echo "=== 2. 安装 vllm ==="
$UV pip install vllm==0.6.3 \
    --index-url $PIP_INDEX

echo ""
echo "=== 3. 安装 bitsandbytes ==="
$UV pip install "bitsandbytes>=0.45.0" \
    --index-url $PIP_INDEX

echo ""
echo "=== 4. 安装 HuggingFace 生态 ==="
$UV pip install "transformers>=4.44.0" "accelerate>=0.33.0" \
    --index-url $PIP_INDEX

echo ""
echo "=== 5. 安装其他依赖 ==="
$UV pip install numpy scipy "datasets>=3.0.0" \
    --index-url $PIP_INDEX

echo ""
echo "=== 6. 验证安装 ==="
$UV pip list | grep -iE "torch|vllm|transformers|numpy|scipy|bitsandbytes|datasets|accelerate"

echo ""
echo "=== 完成! ==="

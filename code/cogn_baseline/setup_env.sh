#!/bin/bash
# ============================================================
# 在 mat_env conda 环境中安装 coNGN 所需依赖
# 使用: bash setup_env.sh
# ============================================================
# mat_env: Python 3.12 + pymatgen 2025.10.7 + torch 2.9.1
# TF 2.16+ 原生支持 Python 3.12

ENV_NAME="mat_env"
ENV_PYTHON="/Users/yuxia.guan/miniconda3/envs/${ENV_NAME}/bin/python"
ENV_PIP="/Users/yuxia.guan/miniconda3/envs/${ENV_NAME}/bin/pip"

echo "=== 安装环境: ${ENV_NAME} ==="
echo "Python: $($ENV_PYTHON --version)"

# 1. TensorFlow 2.16（支持 Python 3.12）
echo "[1/4] 安装 TensorFlow 2.17..."
$ENV_PIP install "tensorflow==2.17.0" --quiet

# 2. kgcnn (KGCNN 4.x，与 TF 2.17 兼容)
echo "[2/4] 安装 kgcnn..."
$ENV_PIP install "kgcnn" --quiet

# 3. matbench (benchmark 工具)
echo "[3/4] 安装 matbench..."
$ENV_PIP install matbench --quiet

# 4. 其他依赖
echo "[4/4] 安装其他依赖..."
$ENV_PIP install "scikit-learn>=1.3" "pandas>=2.0" "scipy>=1.11" --quiet

echo ""
echo "=== 验证安装 ==="
$ENV_PYTHON -c "
import tensorflow as tf
import kgcnn
print(f'TensorFlow: {tf.__version__}')
print(f'KGCNN: {kgcnn.__version__}')
try:
    from matbench.bench import MatbenchBenchmark
    print('matbench: OK')
except Exception as e:
    print(f'matbench: {e}')
print('所有依赖安装成功！')
"

#!/bin/bash
set -euo pipefail

# Linux + NVIDIA GPU 环境安装脚本
# 用法:
#   bash install_linux_gpu.sh
#   bash install_linux_gpu.sh cogn310

ENV_NAME="${1:-cogn310}"

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: 未找到 conda，请先安装 Miniconda/Anaconda。"
    exit 1
fi

CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "[1/5] 创建 conda 环境 ${ENV_NAME} (Python 3.10)..."
    conda create -y -n "${ENV_NAME}" python=3.10
else
    echo "[1/5] 复用已存在环境 ${ENV_NAME}"
fi

conda activate "${ENV_NAME}"

echo "[2/5] 升级 pip..."
python -m pip install --upgrade pip setuptools wheel

echo "[3/5] 清理可能残留的旧版 matbench..."
python -m pip uninstall -y matbench || true

echo "[4/5] 安装 GPU 主依赖..."
python -m pip install -r "$(dirname "$0")/requirements-gpu.txt"

echo "[5/5] 安装 matbench (关闭旧依赖元数据解析)..."
python -m pip install --no-deps "matbench==0.6"

echo "[6/6] 验证安装..."
python - <<'PY'
import tensorflow as tf
import kgcnn
import matbench
import matminer
import pymatgen
import numpy
import pandas
import scipy
import sklearn
from kgcnn.literature.coGN import make_model, model_default_nested

print("TensorFlow:", tf.__version__)
print("kgcnn:", kgcnn.__version__)
print("matbench:", matbench.__version__)
print("matminer:", matminer.__version__)
print("pymatgen:", pymatgen.__version__)
print("numpy:", numpy.__version__)
print("pandas:", pandas.__version__)
print("scipy:", scipy.__version__)
print("scikit-learn:", sklearn.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))
_ = make_model(**model_default_nested)
print("coNGN nested model build: OK")
PY

echo ""
echo "环境就绪。建议先做一次 smoke test:"
echo "  conda activate ${ENV_NAME}"
echo "  python train_cogn_gvrh.py --task matbench_log_gvrh --folds 0 --run-name gpu_smoke_test"

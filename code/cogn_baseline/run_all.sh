#!/bin/bash
# ============================================================
# coNGN 完整复现流程（一键运行）
# ============================================================
# 预计耗时：2-6 小时（取决于是否有 GPU）
# 使用 mp310 环境（Python 3.10）

set -e
PYTHON="/Users/yuxia.guan/miniconda3/envs/mat_env/bin/python"
DIR="$(cd "$(dirname "$0")"; pwd)"

echo "======================================"
echo " coNGN 复现流程"
echo " 工作目录: $DIR"
echo "======================================"
cd "$DIR"

# Step 1: 环境安装（首次运行需要）
if [ "$1" == "--setup" ]; then
    echo "[0/3] 安装依赖..."
    bash setup_env.sh
fi

# Step 2: 训练
echo ""
echo "[1/3] 训练 coNGN on log_gvrh (Matbench 5-fold)..."
$PYTHON train_cogn_gvrh.py 2>&1 | tee train_cogn.log
echo "训练完成，结果：cogn_cv_results.json"

# Step 3: HEA 推理
echo ""
echo "[2/3] 推理 95 个 HEA 样本..."
$PYTHON infer_cogn_hea.py 2>&1 | tee infer_hea.log
echo "推理完成，结果：cogn_hea_results/"

# Step 4: 可视化
echo ""
echo "[3/3] 生成对比图..."
$PYTHON plot_cogn_vs_agmoe.py 2>&1 | tee plot.log
echo "图片保存至 Article/"

echo ""
echo "======================================"
echo " 全部完成！"
echo " 训练结果: cogn_cv_results.json"
echo " HEA 指标: cogn_hea_results/hea_metrics.json"
echo " 对比图:   Article/fig_cogn_vs_agmoe.pdf"
echo "======================================"

"""
plot_cogn_vs_agmoe.py
---------------------
生成 coNGN vs AG-MoE 在 95-sample HEA 上的对比图（NMI 风格）：
  左图：散点图（真值 vs 预测）
  右图：各排名指标柱状对比

运行命令：
    /Users/yuxia.guan/miniconda3/envs/mp310/bin/python plot_cogn_vs_agmoe.py
    （或 mat_env，只需 matplotlib + numpy）
"""

from __future__ import annotations
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr

RESULTS_DIR = Path("cogn_hea_results")
AGMOE_PRED_JSON = Path("../hea_dataset0330_standard/preds.json")  # AG-MoE 预测文件
OUTPUT_PDF = Path("../Article/fig_cogn_vs_agmoe.pdf")
OUTPUT_PNG = Path("../Article/fig_cogn_vs_agmoe.png")

# AG-MoE 汇报指标（直接填入）
AGMOE_METRICS = {
    "SpearmanR": 0.918,
    "KendallTau": 0.756,
    "Precision@9": 0.707,
    "PairwiseAcc": 0.756,
}

# --------------------------------------------------------
# NMI 风格设置
# --------------------------------------------------------
FONT = "Arial"
plt.rcParams.update({
    "font.family": FONT,
    "font.size": 7,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

COLORS = {
    "cogn":  "#D55E00",   # 橙红
    "agmoe": "#009E73",   # 绿色
}


def load_predictions():
    pred_file = RESULTS_DIR / "hea_predictions.json"
    if not pred_file.exists():
        raise FileNotFoundError(
            f"找不到 {pred_file}，请先运行 infer_cogn_hea.py"
        )
    with open(pred_file) as f:
        d = json.load(f)
    return np.array(d["y_true"]), np.array(d["y_pred_ensemble"])


def load_agmoe_predictions():
    if not AGMOE_PRED_JSON.exists():
        return None
    with open(AGMOE_PRED_JSON) as f:
        d = json.load(f)
    # 格式可能是 {"log_gvrh_head": [pred, ...], "y_true": [...]}
    if "log_gvrh_head" in d and "y_true" in d:
        return np.array(d["y_true"]), np.array(d["log_gvrh_head"])
    return None


def main():
    print("生成 coNGN vs AG-MoE 对比图...")

    y_true, y_pred_cogn = load_predictions()
    agmoe_data = load_agmoe_predictions()

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8))

    # --------------------------------------------------------
    # 左图：散点图 coNGN 预测 vs 真值
    # --------------------------------------------------------
    ax = axes[0]
    ax.scatter(y_true, y_pred_cogn, s=18, alpha=0.7,
               color=COLORS["cogn"], linewidths=0, label="coNGN")
    if agmoe_data is not None:
        y_true_agmoe, y_pred_agmoe = agmoe_data
        ax.scatter(y_true_agmoe, y_pred_agmoe, s=18, alpha=0.7,
                   color=COLORS["agmoe"], linewidths=0, label="AG-MoE")

    # 对角线
    lims = [min(y_true) - 0.05, max(y_true) + 0.05]
    ax.plot(lims, lims, "k--", linewidth=0.8, alpha=0.5)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("True log$_{10}$(E / GPa)", fontsize=7)
    ax.set_ylabel("Predicted log$_{10}$(E / GPa)", fontsize=7)
    ax.set_title("a   HEA Young's Modulus (Zero-Shot)", fontsize=7,
                 loc="left", fontweight="bold")

    sr_cogn = spearmanr(y_true, y_pred_cogn).statistic
    ax.text(0.05, 0.93, f"coNGN  ρ = {sr_cogn:.3f}",
            transform=ax.transAxes, fontsize=6, color=COLORS["cogn"])
    ax.text(0.05, 0.85, f"AG-MoE ρ = {AGMOE_METRICS['SpearmanR']:.3f}",
            transform=ax.transAxes, fontsize=6, color=COLORS["agmoe"])

    ax.legend(fontsize=6, frameon=False, loc="lower right")

    # 去掉右轴和上轴
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # --------------------------------------------------------
    # 右图：指标柱状对比
    # --------------------------------------------------------
    ax2 = axes[1]
    # 从结果文件中读取 coNGN 指标
    metrics_file = RESULTS_DIR / "hea_metrics.json"
    with open(metrics_file) as f:
        cogn_metrics = json.load(f)

    metric_labels = ["SpearmanR", "KendallTau", "Precision@9", "PairwiseAcc"]
    cogn_vals = []
    cogn_errs_lo = []
    cogn_errs_hi = []
    for m in metric_labels:
        # 匹配 metrics JSON 的键（Precision@9 可能是 Precision@5 或 Precision@10）
        key = m
        if m == "Precision@9":
            key = [k for k in cogn_metrics if k.startswith("Precision@") and "10" in k]
            key = key[0] if key else m
        elif m == "PairwiseAcc":
            key = "PairwiseAcc(delta=0.0)"
        if key in cogn_metrics:
            cogn_vals.append(cogn_metrics[key]["mean"])
            cogn_errs_lo.append(cogn_metrics[key]["mean"] - cogn_metrics[key]["ci_2p5"])
            cogn_errs_hi.append(cogn_metrics[key]["ci_97p5"] - cogn_metrics[key]["mean"])
        else:
            cogn_vals.append(0.0)
            cogn_errs_lo.append(0.0)
            cogn_errs_hi.append(0.0)

    agmoe_vals = [
        AGMOE_METRICS["SpearmanR"],
        AGMOE_METRICS["KendallTau"],
        AGMOE_METRICS["Precision@9"],
        AGMOE_METRICS["PairwiseAcc"],
    ]

    x = np.arange(len(metric_labels))
    width = 0.35

    bars1 = ax2.bar(x - width / 2, cogn_vals, width,
                    color=COLORS["cogn"], alpha=0.85, label="coNGN")
    bars2 = ax2.bar(x + width / 2, agmoe_vals, width,
                    color=COLORS["agmoe"], alpha=0.85, label="AG-MoE")

    # 误差棒（coNGN）
    ax2.errorbar(
        x - width / 2, cogn_vals,
        yerr=[cogn_errs_lo, cogn_errs_hi],
        fmt="none", color="black", linewidth=0.8, capsize=2,
    )

    ax2.axhline(0.5, color="gray", linewidth=0.6, linestyle="--", alpha=0.6,
                label="Random baseline")

    ax2.set_xticks(x)
    ax2.set_xticklabels(
        ["SpearmanR", "KendallTau", "Prec@10", "PairAcc"],
        fontsize=6
    )
    ax2.set_ylabel("Score", fontsize=7)
    ax2.set_ylim(0, 1.05)
    ax2.set_title("b   Ranking Metrics Comparison", fontsize=7,
                  loc="left", fontweight="bold")
    ax2.legend(fontsize=6, frameon=False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    plt.tight_layout(pad=0.8)

    OUTPUT_PDF.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(str(OUTPUT_PDF), dpi=300, bbox_inches="tight")
    fig.savefig(str(OUTPUT_PNG), dpi=300, bbox_inches="tight")
    plt.close()

    print(f"图片保存至：")
    print(f"  {OUTPUT_PDF}")
    print(f"  {OUTPUT_PNG}")


if __name__ == "__main__":
    main()

"""
infer_cogn_hea.py
-----------------
用训练好的 coNGN 模型（log_gvrh）对 95 个 HEA 样本做零样本迁移预测，
并计算排名指标（SpearmanR、KendallTau、Precision@K）与 AG-MoE 做对比。

运行命令：
    /Users/yuxia.guan/miniconda3/envs/mat_env/bin/python infer_cogn_hea.py

前提：
    1. 已运行 train_cogn_gvrh.py，权重保存在 cogn_weights/fold_X/
    2. hea_dataset0330.json 格式：[[structure_dict, log_E], ...]
"""

from __future__ import annotations
import os, json, sys
import numpy as np
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

from pymatgen.core import Structure
from kgcnn.graph.preprocessor import SetRangePeriodic
from kgcnn.data.crystal import CrystalDataset
try:
    from kgcnn.literature.coNGN._make import make_model
except ImportError:
    from kgcnn.literature.coNGN import make_model
from scipy.stats import spearmanr, kendalltau

# ============================================================
# 路径配置
# ============================================================
WEIGHTS_DIR = Path("cogn_weights")       # train_cogn_gvrh.py 输出目录
HEA_JSON = Path("../data/hea_dataset0330.json")
OUTPUT_DIR = Path("cogn_hea_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# 与 train_cogn_gvrh.py 保持一致
K_NEIGHBORS = 24
CUTOFF = 5.0
BATCH_SIZE = 32

# AG-MoE 在 95 个 HEA 上的结果（供对比）
AGMOE_RESULTS = {
    "SpearmanR": 0.918,
    "KendallTau": 0.756,
    "Precision@9": 0.707,
    "PairwiseAcc(delta=0.0)": 0.756,
}

# ============================================================
# coNGN 模型配置（与训练时完全一致）
# ============================================================
MODEL_CONFIG = {
    "name": "coNGN",
    "inputs": [
        {"shape": [None], "name": "node_number", "dtype": "float32", "ragged": True},
        {"shape": [None, 3], "name": "node_coordinates", "dtype": "float32", "ragged": True},
        {"shape": [None, 2], "name": "range_indices", "dtype": "int64", "ragged": True},
        {"shape": [None, 3], "name": "range_image", "dtype": "int64", "ragged": True},
        {"shape": [None], "name": "graph_size", "dtype": "float32", "ragged": False},
    ],
    "input_tensor_type": "ragged",
    "cast_disjoint_kwargs": {"padded_disjoint": False},
    "input_embedding": None,
    "input_node_embedding": {"input_dim": 95, "output_dim": 64},
    "depth": 5,
    "gin_mlp": {"units": [128, 128], "use_bias": True, "activation": ["swish", "linear"]},
    "graph_mlp": {"units": [128, 64, 1], "use_bias": True, "activation": ["swish", "swish", "linear"]},
    "output_embedding": "graph",
    "output_tensor_type": "padded",
    "output_mlp": {"use_bias": [True, True], "units": [128, 1], "activation": ["swish", "linear"]},
}

# ============================================================
# 辅助函数
# ============================================================
GRAPH_PREPROCESSORS = [
    SetRangePeriodic(
        cutoff=CUTOFF,
        max_neighbors=K_NEIGHBORS,
        node_coordinates="node_coordinates",
        range_indices="range_indices",
        range_image="range_image",
        range_attributes="range_attributes",
    )
]


def bootstrap_ci(func, *args, n_boot=2000, seed=42, ci=95):
    """配对 bootstrap CI。"""
    rng = np.random.default_rng(seed)
    n = len(args[0])
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        resampled = [a[idx] for a in args]
        try:
            val = func(*resampled)
            if np.isfinite(val):
                stats.append(val)
        except Exception:
            pass
    lo = np.percentile(stats, (100 - ci) / 2)
    hi = np.percentile(stats, 100 - (100 - ci) / 2)
    return float(np.mean(stats)), lo, hi


def precision_at_k(y_true, y_pred, k, top_frac=0.1):
    """Precision@K：前 K 个预测中，有多少在真值 top-K 里。"""
    top_true = set(np.argsort(y_true)[-k:])
    top_pred = set(np.argsort(y_pred)[-k:])
    return len(top_true & top_pred) / k


def pairwise_acc(y_true, y_pred, delta=0.0):
    """PairwiseAcc(delta)：对具有显著差异的样本对，预测排序是否一致。"""
    n = len(y_true)
    correct = total = 0
    for i in range(n):
        for j in range(i + 1, n):
            if abs(y_true[i] - y_true[j]) > delta:
                total += 1
                if (y_pred[i] - y_pred[j]) * (y_true[i] - y_true[j]) > 0:
                    correct += 1
    return correct / total if total > 0 else 0.0


# ============================================================
# 将 pymatgen Structure 转为 KGCNN 图
# ============================================================
def structure_to_kgcnn_graph(struct: Structure) -> dict:
    """
    将 pymatgen Structure 转换为 KGCNN CrystalDataset 所需的字典格式。
    包含：node_number, node_coordinates, lattice_matrix
    边（range_indices / range_image）在 preprocessor 中生成。
    """
    numbers = np.array([site.specie.Z for site in struct], dtype=np.float32)
    coords = np.array(struct.cart_coords, dtype=np.float32)
    lattice = np.array(struct.lattice.matrix, dtype=np.float32)

    return {
        "node_number": numbers,
        "node_coordinates": coords,
        "graph_lattice": lattice,
        "graph_size": np.array([len(struct)], dtype=np.float32),
    }


def load_hea_as_kgcnn_dataset():
    """加载 HEA JSON 并构建 KGCNN 格式数据集。"""
    print(f"加载 HEA 数据：{HEA_JSON}")
    with open(HEA_JSON) as f:
        raw = json.load(f)

    structures, targets = [], []
    for struct_dict, target in raw:
        structures.append(Structure.from_dict(struct_dict))
        targets.append(float(target))

    print(f"  样本数：{len(structures)}")
    print(f"  目标值范围：{min(targets):.3f} ~ {max(targets):.3f} (log_E)")

    # 构建 CrystalDataset
    graphs = [structure_to_kgcnn_graph(s) for s in structures]

    dataset = CrystalDataset()
    dataset.assign_property(graphs)
    dataset.apply_preprocessor(GRAPH_PREPROCESSORS)

    return dataset, np.array(targets)


def get_model_inputs(dataset):
    """将数据集转为 coNGN 模型输入张量。"""
    return (
        dataset.tensor([{"name": "node_number", "ragged": True}]),
        dataset.tensor([{"name": "node_coordinates", "ragged": True}]),
        dataset.tensor([{"name": "range_indices", "ragged": True}]),
        dataset.tensor([{"name": "range_image", "ragged": True}]),
        dataset.tensor([{"name": "graph_size", "ragged": False}]),
    )


# ============================================================
# 主流程
# ============================================================
def main():
    print("=" * 60)
    print("coNGN → HEA Zero-Shot 推理")
    print("=" * 60)

    # 检查权重目录
    if not WEIGHTS_DIR.exists():
        print(f"ERROR：找不到权重目录 {WEIGHTS_DIR}")
        print("请先运行：python train_cogn_gvrh.py")
        sys.exit(1)

    fold_dirs = sorted(WEIGHTS_DIR.glob("fold_*"))
    if not fold_dirs:
        print("ERROR：未找到任何 fold 权重，请先运行训练脚本。")
        sys.exit(1)

    print(f"找到 {len(fold_dirs)} 个 fold 权重")

    # 加载 HEA 数据
    dataset, y_true = load_hea_as_kgcnn_dataset()
    x = get_model_inputs(dataset)

    # 对所有 fold 做推理，取平均（ensemble）
    all_preds = []
    for fold_dir in fold_dirs:
        # 兼容 .keras（新格式）和 .h5（旧格式）
        weight_path = fold_dir / "weights.keras"
        if not weight_path.exists():
            weight_path = fold_dir / "weights.h5"
        scaler_path = fold_dir / "scaler.json"

        if not weight_path.exists():
            print(f"  跳过 {fold_dir.name}（无权重文件）")
            continue

        print(f"\n加载 {fold_dir.name}...")
        model = make_model(**MODEL_CONFIG)
        model.load_weights(str(weight_path))

        # 加载归一化参数（反变换）
        with open(scaler_path) as f:
            scaler_params = json.load(f)
        mean_  = scaler_params["mean"]
        scale_ = scaler_params["std"]   # 训练时用 std 归一化

        y_pred_scaled = model.predict(x, batch_size=BATCH_SIZE, verbose=0).flatten()
        y_pred = y_pred_scaled * scale_ + mean_  # 反归一化
        all_preds.append(y_pred)

        tf.keras.backend.clear_session()

    if not all_preds:
        print("ERROR：没有有效的预测结果。")
        sys.exit(1)

    # Ensemble 平均
    y_pred_ensemble = np.mean(all_preds, axis=0)

    # ============================================================
    # 计算排名指标
    # ============================================================
    print("\n" + "=" * 60)
    print("计算排名指标")
    print("=" * 60)

    N = len(y_true)
    k5  = max(1, round(N * 0.05))   # 5%  → ~5
    k10 = max(1, round(N * 0.10))   # 10% → ~9/10

    # SpearmanR
    sr_mean, sr_lo, sr_hi = bootstrap_ci(
        lambda a, b: spearmanr(a, b).statistic, y_true, y_pred_ensemble
    )
    # KendallTau
    kt_mean, kt_lo, kt_hi = bootstrap_ci(
        lambda a, b: kendalltau(a, b).statistic, y_true, y_pred_ensemble
    )
    # Precision@K
    p5_mean, p5_lo, p5_hi = bootstrap_ci(
        lambda a, b: precision_at_k(a, b, k=k5), y_true, y_pred_ensemble
    )
    p10_mean, p10_lo, p10_hi = bootstrap_ci(
        lambda a, b: precision_at_k(a, b, k=k10), y_true, y_pred_ensemble
    )
    # PairwiseAcc
    pa_mean, pa_lo, pa_hi = bootstrap_ci(
        lambda a, b: pairwise_acc(a, b, delta=0.0), y_true, y_pred_ensemble
    )

    results = {
        "model": "coNGN (log_gvrh → HEA Young's modulus, zero-shot)",
        "n_folds_ensemble": len(all_preds),
        "n_samples": N,
        "SpearmanR": {"mean": sr_mean, "ci_2p5": sr_lo, "ci_97p5": sr_hi},
        "KendallTau": {"mean": kt_mean, "ci_2p5": kt_lo, "ci_97p5": kt_hi},
        f"Precision@{k5}": {"mean": p5_mean, "ci_2p5": p5_lo, "ci_97p5": p5_hi},
        f"Precision@{k10}": {"mean": p10_mean, "ci_2p5": p10_lo, "ci_97p5": p10_hi},
        "PairwiseAcc(delta=0.0)": {"mean": pa_mean, "ci_2p5": pa_lo, "ci_97p5": pa_hi},
    }

    # 保存预测结果
    pred_out = {
        "y_true": y_true.tolist(),
        "y_pred_ensemble": y_pred_ensemble.tolist(),
        "y_pred_per_fold": [p.tolist() for p in all_preds],
    }
    with open(OUTPUT_DIR / "hea_predictions.json", "w") as f:
        json.dump(pred_out, f, indent=2)

    with open(OUTPUT_DIR / "hea_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # ============================================================
    # 打印对比表
    # ============================================================
    print(f"\n{'指标':<30} {'coNGN':>12} {'AG-MoE':>12}")
    print("-" * 56)
    metrics_map = [
        ("SpearmanR", f"{sr_mean:.3f} [{sr_lo:.3f},{sr_hi:.3f}]",
         f"{AGMOE_RESULTS['SpearmanR']:.3f}"),
        ("KendallTau", f"{kt_mean:.3f} [{kt_lo:.3f},{kt_hi:.3f}]",
         f"{AGMOE_RESULTS['KendallTau']:.3f}"),
        (f"Precision@{k5}", f"{p5_mean:.3f} [{p5_lo:.3f},{p5_hi:.3f}]", "-"),
        (f"Precision@{k10}", f"{p10_mean:.3f} [{p10_lo:.3f},{p10_hi:.3f}]",
         f"{AGMOE_RESULTS['Precision@9']:.3f}"),
        ("PairwiseAcc(delta=0.0)", f"{pa_mean:.3f} [{pa_lo:.3f},{pa_hi:.3f}]",
         f"{AGMOE_RESULTS['PairwiseAcc(delta=0.0)']:.3f}"),
    ]
    for name, cogn_val, agmoe_val in metrics_map:
        print(f"{name:<30} {cogn_val:>20} {agmoe_val:>12}")

    print("\n结果已保存至:", OUTPUT_DIR)
    print("  hea_predictions.json  - 每个样本的预测值")
    print("  hea_metrics.json      - 完整指标")


if __name__ == "__main__":
    main()

"""
train_cogn_gvrh.py
------------------
使用 KGCNN 官方 coNGN 模型在 Matbench log_gvrh 上做 5-fold 交叉验证训练。
每个 fold 保存模型权重，供 HEA 推理使用。

运行环境：mat_env (Python 3.12 + TF 2.17 + kgcnn)
安装依赖：bash setup_env.sh

运行命令：
    /Users/yuxia.guan/miniconda3/envs/mat_env/bin/python train_cogn_gvrh.py

输出：
    cogn_weights/fold_0/   ...  cogn_weights/fold_4/
    cogn_cv_results.json
"""

from __future__ import annotations
import os, json, time, logging
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import tensorflow as tf
print(f"TensorFlow: {tf.__version__}")
print(f"GPU 设备: {tf.config.list_physical_devices('GPU')}")

# ====================================================
# KGCNN imports
# ====================================================
try:
    from kgcnn.data.datasets.MatbenchDataset2020 import MatbenchDataset2020
    from kgcnn.literature.coNGN._make import make_model
    from kgcnn.graph.preprocessor import SetRangePeriodic
except ImportError:
    # 兼容不同 kgcnn 版本
    try:
        from kgcnn.literature.coNGN import make_model
    except ImportError:
        raise ImportError(
            "请先安装 kgcnn: pip install kgcnn\n"
            "或运行: bash setup_env.sh"
        )

# ====================================================
# 配置
# ====================================================
TASK = "matbench_log_gvrh"
EPOCHS = 800
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
K_NEIGHBORS = 24       # coNGN 关键：高连通性
CUTOFF = 5.0           # Å
WEIGHT_DIR = "cogn_weights"
PATIENCE = 100         # EarlyStopping patience

# ====================================================
# coNGN 模型配置（参照论文 Table S1）
# ====================================================
MODEL_KWARGS = {
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
    "input_node_embedding": {"input_dim": 95, "output_dim": 64},
    "depth": 5,
    "gin_mlp": {
        "units": [128, 128],
        "use_bias": True,
        "activation": ["swish", "linear"],
    },
    "output_embedding": "graph",
    "output_tensor_type": "padded",
    "output_mlp": {
        "use_bias": [True, True],
        "units": [128, 1],
        "activation": ["swish", "linear"],
    },
}


def build_dataset_and_folds():
    """加载 Matbench log_gvrh 数据集，生成 5-fold splits。"""
    print(f"\n[数据] 加载 {TASK}...")
    dataset = MatbenchDataset2020(TASK)
    dataset.prepare_data()
    dataset.read_in_memory(label_column_name="target")

    print(f"  预处理：{K_NEIGHBORS}-NN 周期边界...")
    preprocessor = SetRangePeriodic(
        cutoff=CUTOFF,
        max_neighbors=K_NEIGHBORS,
        node_coordinates="node_coordinates",
        range_indices="range_indices",
        range_image="range_image",
        range_attributes="range_attributes",
    )
    dataset.apply_preprocessor([preprocessor])
    print(f"  总样本数：{len(dataset)}")

    # matbench 5-fold
    folds = dataset.kfold_splits(n_splits=5)
    return dataset, folds


def get_tf_inputs(dataset):
    """从数据集生成 coNGN 所需的 TF 张量元组。"""
    x = (
        dataset.tensor([{"name": "node_number", "ragged": True}]),
        dataset.tensor([{"name": "node_coordinates", "ragged": True}]),
        dataset.tensor([{"name": "range_indices", "ragged": True}]),
        dataset.tensor([{"name": "range_image", "ragged": True}]),
        dataset.tensor([{"name": "graph_size", "ragged": False}]),
    )
    return x


def train_one_fold(dataset, train_idx, val_idx, fold_idx: int) -> dict:
    """训练单个 fold，返回验证集 MAE。"""
    fold_dir = os.path.join(WEIGHT_DIR, f"fold_{fold_idx}")
    os.makedirs(fold_dir, exist_ok=True)

    train_ds = dataset[train_idx]
    val_ds   = dataset[val_idx]

    y_train = np.array(train_ds.get("graph_labels"), dtype=np.float32)
    y_val   = np.array(val_ds.get("graph_labels"),   dtype=np.float32)

    # 标签归一化（zero-mean/unit-std）
    y_mean  = float(y_train.mean())
    y_std   = float(y_train.std()) + 1e-8
    y_train_scaled = (y_train - y_mean) / y_std
    y_val_scaled   = (y_val   - y_mean) / y_std

    scaler_info = {"mean": y_mean, "std": y_std}
    with open(os.path.join(fold_dir, "scaler.json"), "w") as f:
        json.dump(scaler_info, f)

    # 模型
    model = make_model(**MODEL_KWARGS)

    # 学习率衰减
    def lr_schedule(epoch):
        warmup = 200
        if epoch < warmup:
            return LEARNING_RATE * (epoch + 1) / warmup
        return LEARNING_RATE * (0.9985 ** (epoch - warmup))

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss="mae", metrics=["mae"])

    x_train = get_tf_inputs(train_ds)
    x_val   = get_tf_inputs(val_ds)

    callbacks = [
        tf.keras.callbacks.LearningRateScheduler(lr_schedule),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(fold_dir, "weights.keras"),
            save_best_only=True,
            monitor="val_mae",
            verbose=0,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_mae",
            patience=PATIENCE,
            restore_best_weights=True,
        ),
    ]

    print(f"\n[Fold {fold_idx}] 训练：{len(y_train)} 样本，验证：{len(y_val)} 样本")
    model.fit(
        x_train, y_train_scaled,
        validation_data=(x_val, y_val_scaled),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    # 计算真实尺度 MAE
    y_pred_scaled = model.predict(x_val, batch_size=BATCH_SIZE, verbose=0).flatten()
    y_pred = y_pred_scaled * y_std + y_mean
    mae = float(np.mean(np.abs(y_pred - y_val)))
    print(f"  Fold {fold_idx} MAE = {mae:.4f}")

    tf.keras.backend.clear_session()
    return {"fold": fold_idx, "val_mae": mae, "n_train": len(y_train), "n_val": len(y_val)}


def main():
    t0 = time.time()
    print("=" * 60)
    print(f"coNGN 训练 | 任务: {TASK}")
    print("=" * 60)

    os.makedirs(WEIGHT_DIR, exist_ok=True)
    dataset, folds = build_dataset_and_folds()

    fold_results = []
    for i, (train_idx, val_idx) in enumerate(folds):
        result = train_one_fold(dataset, train_idx, val_idx, i)
        fold_results.append(result)

    maes = [r["val_mae"] for r in fold_results]
    summary = {
        "task": TASK,
        "model": "coNGN",
        "k_neighbors": K_NEIGHBORS,
        "folds": fold_results,
        "mean_mae": float(np.mean(maes)),
        "std_mae":  float(np.std(maes)),
        "elapsed_min": (time.time() - t0) / 60,
    }
    with open("cogn_cv_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print(f"coNGN on {TASK}（复现）")
    print(f"  5-fold MAE：{summary['mean_mae']:.4f} ± {summary['std_mae']:.4f}")
    print(f"  论文报告  ：0.0670")
    print(f"  总耗时    ：{summary['elapsed_min']:.1f} 分钟")
    print(f"  权重目录  ：{WEIGHT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from pymatgen.core import Structure
from scipy.stats import kendalltau, spearmanr

from kgcnn.data.crystal import CrystalDataset

try:
    from kgcnn.crystal.preprocessor import VoronoiAsymmetricUnitCell
    from kgcnn.literature.coGN import make_model, model_default_nested
except ImportError:
    try:
        from kgcnn.crystal.preprocessor import VoronoiAsymmetricUnitCell
        from kgcnn.literature.coNGN._make import make_model
        model_default_nested = None
    except ImportError:
        from kgcnn.crystal.preprocessor import VoronoiAsymmetricUnitCell
        from kgcnn.literature.coNGN import make_model
        model_default_nested = None


MODEL_CONFIG = dict(model_default_nested) if model_default_nested is not None else {
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

DEFAULT_BATCH_SIZE = 32


def configure_tensorflow_runtime() -> dict:
    gpu_devices = tf.config.list_physical_devices("GPU")
    configured = []
    for gpu in gpu_devices:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
            configured.append({"name": gpu.name, "memory_growth": True})
        except Exception as exc:
            configured.append({"name": gpu.name, "memory_growth": False, "error": str(exc)})
    return {"physical_gpus": [gpu.name for gpu in gpu_devices], "memory_growth": configured}


def bootstrap_ci(metric_fn, y_true, y_score, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        try:
            v = metric_fn(y_true[idx], y_score[idx])
        except Exception:
            continue
        if np.isfinite(v):
            vals.append(v)
    vals = np.asarray(vals, dtype=float)
    return float(np.mean(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def precision_at_k(y_true, y_score, k):
    idx_true = set(np.argsort(-y_true)[:k])
    idx_pred = set(np.argsort(-y_score)[:k])
    return len(idx_true & idx_pred) / k


def pairwise_acc(y_true, y_score, delta=0.0):
    n = len(y_true)
    correct, total = 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            if abs(y_true[i] - y_true[j]) < delta:
                continue
            total += 1
            if (y_true[i] > y_true[j]) == (y_score[i] > y_score[j]):
                correct += 1
    return correct / total if total > 0 else float("nan")


def build_line_graph_edge_indices(edge_indices: np.ndarray) -> np.ndarray:
    edge_indices = np.asarray(edge_indices, dtype=np.int32)
    if edge_indices.ndim != 2 or edge_indices.shape[1] != 2 or len(edge_indices) == 0:
        return np.zeros((0, 2), dtype=np.int32)

    incoming = {}
    outgoing = {}
    for eid, (src, dst) in enumerate(edge_indices):
        outgoing.setdefault(int(src), []).append(eid)
        incoming.setdefault(int(dst), []).append(eid)

    line_edges = []
    for center in sorted(set(incoming) & set(outgoing)):
        for e_in in incoming[center]:
            for e_out in outgoing[center]:
                if e_in != e_out:
                    line_edges.append((e_in, e_out))

    if not line_edges:
        return np.zeros((0, 2), dtype=np.int32)
    return np.asarray(line_edges, dtype=np.int32)


def structure_to_cogn_graph(struct: Structure, preprocessor: VoronoiAsymmetricUnitCell) -> dict:
    graph = preprocessor(struct)
    graph["line_graph_edge_indices"] = build_line_graph_edge_indices(graph["edge_indices"])
    return graph


def load_real_dataset(json_path: Path) -> tuple[CrystalDataset, np.ndarray]:
    raw = json.load(open(json_path))

    if isinstance(raw, list):
        rows = raw
    elif isinstance(raw, dict) and "data" in raw:
        rows = raw["data"]
    else:
        raise ValueError(f"Unsupported JSON format in {json_path}")

    structures, targets = [], []
    for item in rows:
        struct_dict, target = item[0], item[1]
        structures.append(Structure.from_dict(struct_dict))
        targets.append(float(target))

    preprocessor = VoronoiAsymmetricUnitCell()
    preprocessor.output_graph_as_dict = True
    graphs = [structure_to_cogn_graph(s, preprocessor) for s in structures]
    dataset = CrystalDataset()
    dataset.clear()
    dataset.extend(graphs)
    return dataset, np.asarray(targets, dtype=float)


def get_model_inputs(dataset: CrystalDataset):
    return (
        dataset.tensor([{"name": "offset", "ragged": True}]),
        dataset.tensor([{"name": "voronoi_ridge_area", "ragged": True}]),
        dataset.tensor([{"name": "atomic_number", "ragged": True}]),
        dataset.tensor([{"name": "multiplicity", "ragged": True}]),
        dataset.tensor([{"name": "line_graph_edge_indices", "ragged": True}]),
        dataset.tensor([{"name": "edge_indices", "ragged": True}]),
    )


def resolve_fold_dirs(weights_path: Path) -> list[Path]:
    if weights_path.is_file():
        return [weights_path.parent]

    if weights_path.is_dir() and any(
        (weights_path / name).exists()
        for name in ["weights.keras", "weights.h5", "best_model.keras", "weights.weights.h5"]
    ):
        return [weights_path]

    fold_dirs = sorted([p for p in weights_path.glob("fold_*") if p.is_dir()])
    if fold_dirs:
        return fold_dirs

    raise FileNotFoundError(
        f"Could not resolve fold directories from {weights_path}. "
        "Pass either a fold directory or a run directory containing fold_*/"
    )


def run_inference(fold_dirs: list[Path], x, batch_size: int) -> tuple[np.ndarray, list[np.ndarray]]:
    all_preds = []

    for fold_dir in fold_dirs:
        weight_path = None
        for candidate in ["weights.keras", "best_model.keras", "weights.h5", "weights.weights.h5"]:
            cand_path = fold_dir / candidate
            if cand_path.exists():
                weight_path = cand_path
                break
        scaler_path = fold_dir / "scaler.json"

        if weight_path is None:
            print(f"[Warn] skip {fold_dir} (missing weights)")
            continue
        if not scaler_path.exists():
            raise FileNotFoundError(f"Missing scaler.json in {fold_dir}")

        print(f"[Eval] loading {fold_dir}")
        model = make_model(**MODEL_CONFIG)
        model.load_weights(str(weight_path))

        scaler = json.load(open(scaler_path))
        mean_ = scaler["mean"]
        std_ = scaler["std"]

        y_pred_scaled = model.predict(x, batch_size=batch_size, verbose=0).flatten()
        y_pred = y_pred_scaled * std_ + mean_
        all_preds.append(y_pred)

        tf.keras.backend.clear_session()

    if not all_preds:
        raise RuntimeError("No valid predictions were produced.")

    return np.mean(all_preds, axis=0), all_preds


def save_metrics(out_dir: Path, score_name: str, y_true: np.ndarray, y_pred: np.ndarray, ks: list[int], deltas: list[float], n_boot: int, seed: int):
    rows = []
    for k in ks:
        mean, lo, hi = bootstrap_ci(
            lambda a, b: precision_at_k(a, b, k),
            y_true,
            y_pred,
            n_boot=n_boot,
            seed=seed,
        )
        rows.append(
            {
                "score": score_name,
                "metric": f"Precision@{k}",
                "mean": mean,
                "ci_2p5": lo,
                "ci_97p5": hi,
                "random": k / len(y_true),
            }
        )

    for delta in deltas:
        mean, lo, hi = bootstrap_ci(
            lambda a, b: pairwise_acc(a, b, delta),
            y_true,
            y_pred,
            n_boot=n_boot,
            seed=seed,
        )
        rows.append(
            {
                "score": score_name,
                "metric": f"PairwiseAcc(delta={delta})",
                "mean": mean,
                "ci_2p5": lo,
                "ci_97p5": hi,
                "random": 0.5,
            }
        )

    sr_mean, sr_lo, sr_hi = bootstrap_ci(
        lambda a, b: spearmanr(a, b).statistic,
        y_true,
        y_pred,
        n_boot=n_boot,
        seed=seed,
    )
    kt_mean, kt_lo, kt_hi = bootstrap_ci(
        lambda a, b: kendalltau(a, b).statistic,
        y_true,
        y_pred,
        n_boot=n_boot,
        seed=seed,
    )
    rows.extend(
        [
            {"score": score_name, "metric": "SpearmanR", "mean": sr_mean, "ci_2p5": sr_lo, "ci_97p5": sr_hi, "random": None},
            {"score": score_name, "metric": "KendallTau", "mean": kt_mean, "ci_2p5": kt_lo, "ci_97p5": kt_hi, "random": None},
        ]
    )

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(rows, f, indent=2)

    with open(out_dir / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["score", "metric", "mean", "ci_2p5", "ci_97p5", "random"])
        writer.writeheader()
        writer.writerows(rows)

    return rows


def save_predictions(out_dir: Path, y_true: np.ndarray, y_pred: np.ndarray, all_preds: list[np.ndarray]):
    preds = []
    for i in range(len(y_true)):
        preds.append(
            {
                "i": i,
                "y_true": float(y_true[i]),
                "score_pred": float(y_pred[i]),
                "score_g": float(y_pred[i]),
            }
        )
    with open(out_dir / "preds.json", "w") as f:
        json.dump(preds, f, indent=2)

    payload = {
        "y_true": y_true.tolist(),
        "y_pred_ensemble": y_pred.tolist(),
        "y_pred_per_fold": [p.tolist() for p in all_preds],
    }
    with open(out_dir / "predictions_full.json", "w") as f:
        json.dump(payload, f, indent=2)


def save_topk_table(out_dir: Path, y_true: np.ndarray, y_pred: np.ndarray, topk: int):
    import pandas as pd

    df = pd.DataFrame({"y_true": y_true, "score_pred": y_pred})
    df["rank_true"] = df["y_true"].rank(ascending=False, method="min")
    df["rank_pred"] = df["score_pred"].rank(ascending=False, method="min")
    df["hit"] = ((df["rank_true"] <= topk) & (df["rank_pred"] <= topk)).astype(int)
    df.sort_values("rank_pred").to_csv(out_dir / f"top{topk}_hit_table.csv", index=False)


def main():
    ap = argparse.ArgumentParser(description="Evaluate coNGN baseline on real Young's modulus datasets.")
    ap.add_argument(
        "--weights",
        default="fold_0",
        help="A fold directory like cogn_baseline/fold_0 or a run directory containing fold_*/",
    )
    ap.add_argument(
        "--data_json",
        required=True,
        help="Path to real-data JSON, e.g. ../data/hea_dataset0330.json or ../data/matbench_Young's_modulus_merged.json",
    )
    ap.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--Ks", default="6,9")
    ap.add_argument("--deltas", default="0,0.05,0.08")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--score_name", default="cogn_log_gvrh")
    ap.add_argument("--topk_table", type=int, default=9)
    ap.add_argument("--out_dir", default="cogn_real_eval")
    args = ap.parse_args()

    weights_path = Path(args.weights)
    data_json = Path(args.data_json)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ks = [int(x) for x in args.Ks.split(",") if x.strip()]
    deltas = [float(x) for x in args.deltas.split(",") if x.strip()]

    print("=" * 60)
    print("coNGN real-data evaluation")
    print(f"[Eval] weights = {weights_path}")
    print(f"[Eval] data_json = {data_json}")
    print(f"[Eval] out_dir = {out_dir}")
    print(f"[Eval] Ks = {ks}, deltas = {deltas}")
    print(configure_tensorflow_runtime())
    print("=" * 60)

    fold_dirs = resolve_fold_dirs(weights_path)
    print(f"[Eval] using {len(fold_dirs)} fold dir(s)")

    dataset, y_true = load_real_dataset(data_json)
    x = get_model_inputs(dataset)
    y_pred, all_preds = run_inference(fold_dirs, x, args.batch_size)

    save_metrics(out_dir, args.score_name, y_true, y_pred, ks, deltas, args.n_boot, args.seed)
    save_predictions(out_dir, y_true, y_pred, all_preds)
    save_topk_table(out_dir, y_true, y_pred, args.topk_table)

    run_cfg = {
        "weights": str(weights_path),
        "fold_dirs": [str(p) for p in fold_dirs],
        "data_json": str(data_json),
        "n_samples": int(len(y_true)),
        "score_name": args.score_name,
    }
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(run_cfg, f, indent=2)

    print("[Done] Saved:")
    print(f"  {out_dir / 'metrics.csv'}")
    print(f"  {out_dir / 'metrics.json'}")
    print(f"  {out_dir / 'preds.json'}")
    print(f"  {out_dir / f'top{args.topk_table}_hit_table.csv'}")


if __name__ == "__main__":
    main()

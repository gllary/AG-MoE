from __future__ import annotations

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import kendalltau, spearmanr
from torch.utils.data import DataLoader, Dataset

from mat_models.SingleExpertModel_v17 import SingleExpertModelV17
from mat_models.moe.Stage2MoEModel_v17 import Stage2MoEModelV17
from mat_models.moe.Stage2MoEModel_v18 import Stage2MoEModelV18
from mat_models.moe.Stage2MoEModel_v23 import Stage2MoEModelV23


class StructureDataset(Dataset):
    def __init__(self, raw):
        self.samples = []
        for (atom, nbr, idx), target, meta in raw:
            self.samples.append(
                (atom.float(), nbr.float(), idx.long(), target.float().view(1), meta)
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


def pyg_collate(batch):
    xs, eis, eas, bis, ys, metas = [], [], [], [], [], []
    offset = 0

    for g, (atom, nbr, idx, y, meta) in enumerate(batch):
        n_atoms, n_nbrs = idx.shape
        xs.append(atom)

        src = torch.arange(n_atoms, dtype=torch.long).repeat_interleave(n_nbrs) + offset
        dst = idx.reshape(-1).long() + offset
        eis.append(torch.stack([src, dst], dim=0))
        eas.append(nbr.reshape(-1, nbr.size(2)))
        bis.append(torch.full((n_atoms,), g, dtype=torch.long))
        ys.append(y.view(1, 1))
        metas.append(meta)

        offset += n_atoms

    return (
        torch.cat(xs, dim=0),
        torch.cat(eis, dim=1),
        torch.cat(eas, dim=0),
        torch.cat(bis, dim=0),
        torch.cat(ys, dim=0),
        metas,
    )


def precision_at_k(y_true, y_score, k):
    idx_true = np.argsort(-y_true)[:k]
    idx_pred = np.argsort(-y_score)[:k]
    return len(set(idx_true) & set(idx_pred)) / k


def pairwise_accuracy(y_true, y_score, delta=0.0):
    correct, total = 0, 0
    n = len(y_true)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(y_true[i] - y_true[j]) < delta:
                continue
            total += 1
            if (y_true[i] > y_true[j]) == (y_score[i] > y_score[j]):
                correct += 1
    return correct / total if total > 0 else float("nan")


def bootstrap_ci(metric_fn, y_true, y_score, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        vals.append(metric_fn(y_true[idx], y_score[idx]))
    vals = np.asarray(vals, dtype=float)
    return float(np.mean(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def bootstrap_rank_corr(y_true, y_score, kind="spearman", n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        if kind == "spearman":
            v = spearmanr(y_true[idx], y_score[idx]).statistic
        else:
            v = kendalltau(y_true[idx], y_score[idx]).statistic
        if not np.isnan(v):
            vals.append(v)
    vals = np.asarray(vals, dtype=float)
    return float(np.mean(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def fusion_softplus_phys(pred_k, pred_g, eps=1e-8):
    k_val = np.log1p(np.exp(pred_k)) + eps
    g_val = np.log1p(np.exp(pred_g)) + eps
    return 9.0 * k_val * g_val / (3.0 * k_val + g_val + eps)


def infer_model_version(args_model_version: str | None, config_path: str, ckpt_path: str) -> str:
    if args_model_version:
        return args_model_version.lower()

    merged = f"{config_path} {ckpt_path}".lower()
    match = re.search(r"v(17|18|23)", merged)
    if not match:
        raise ValueError(
            "Cannot infer model version from --config/--ckpt. "
            "Please pass --model_version v17|v18|v23."
        )
    return f"v{match.group(1)}"


def resolve_task_name(cfg_tasks: dict, user_task: str | None, candidates: list[str], role: str) -> str:
    if user_task:
        if user_task not in cfg_tasks:
            raise KeyError(f"{role} task '{user_task}' not found in config tasks: {list(cfg_tasks.keys())}")
        return user_task

    for name in candidates:
        if name in cfg_tasks:
            return name

    raise KeyError(f"Cannot auto-resolve {role} task from config tasks: {list(cfg_tasks.keys())}")


def get_model_bundle(version: str):
    if version == "v17":
        from mat_models.moe.Stage2MoEModel_v17 import ExpertSpec

        return Stage2MoEModelV17, ExpertSpec
    if version == "v18":
        from mat_models.moe.Stage2MoEModel_v18 import ExpertSpec

        return Stage2MoEModelV18, ExpertSpec
    if version == "v23":
        from mat_models.moe.Stage2MoEModel_v23 import ExpertSpec

        return Stage2MoEModelV23, ExpertSpec
    raise ValueError(f"Unsupported model version: {version}")


def build_model(cfg: dict, ckpt_path: str, version: str, device: torch.device):
    model_cls, expert_spec_cls = get_model_bundle(version)
    moe_dim = int(cfg.get("moe_dim", 512))

    experts = [
        expert_spec_cls(
            name=name,
            mode=tc["mode"],
            ckpt_path=tc["ckpt_path"],
            stage1_cfg=tc["stage1_cfg"],
            moe_dim=moe_dim,
        )
        for name, tc in cfg["tasks"].items()
    ]

    model = model_cls(
        experts=experts,
        moe_dim=moe_dim,
        top_k=int(cfg.get("top_k", 2)),
        router_hidden=int(cfg.get("router_hidden", 256)),
        router_dropout=float(cfg.get("router_dropout", 0.1)),
        device=device,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def infer_structure_dims_from_raw(raw) -> tuple[int, int]:
    atom, nbr, _idx = raw[0][0]
    return int(atom.shape[1]), int(nbr.shape[2])


def build_v17_stage1_model(cfg: dict, ckpt_path: str, raw, device: torch.device):
    mode = cfg["mode"]
    embed_dim = int(cfg.get("embed_dim", 512))

    if mode != "structure":
        raise ValueError(f"Current real-data script expects structure experts for v17, got mode={mode}")

    atom_dim, edge_dim = infer_structure_dims_from_raw(raw)
    struct_cfg = cfg.get("struct_config", {})

    model = SingleExpertModelV17(
        mode="structure",
        embed_dim=embed_dim,
        atom_dim=atom_dim,
        edge_dim=edge_dim,
        struct_node_dim=int(struct_cfg.get("node_dim", 128)),
        struct_conv_layers=int(struct_cfg.get("conv_layers", cfg.get("depth", 3))),
        struct_graphormer_layers=int(struct_cfg.get("graphormer_layers", 2)),
        struct_num_heads=int(struct_cfg.get("num_heads", 4)),
        struct_ff_hidden=int(struct_cfg.get("ff_hidden", 256)),
        struct_dropout=float(cfg.get("dropout", 0.1)),
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def build_metric_rows(score_name: str, score: np.ndarray, y_true: np.ndarray, ks: list[int], deltas: list[float], n_boot: int, seed: int):
    rows = []
    for k in ks:
        mean, lo, hi = bootstrap_ci(
            lambda yt, ys: precision_at_k(yt, ys, k),
            y_true,
            score,
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
            lambda yt, ys: pairwise_accuracy(yt, ys, delta),
            y_true,
            score,
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

    r_mean, r_lo, r_hi = bootstrap_rank_corr(y_true, score, "spearman", n_boot, seed)
    t_mean, t_lo, t_hi = bootstrap_rank_corr(y_true, score, "kendall", n_boot, seed)
    rows.extend(
        [
            {
                "score": score_name,
                "metric": "SpearmanR",
                "mean": r_mean,
                "ci_2p5": r_lo,
                "ci_97p5": r_hi,
                "random": None,
            },
            {
                "score": score_name,
                "metric": "KendallTau",
                "mean": t_mean,
                "ci_2p5": t_lo,
                "ci_97p5": t_hi,
                "random": None,
            },
        ]
    )
    return rows


def save_topk_table(out_dir: Path, score_name: str, col_name: str, y_true: np.ndarray, score: np.ndarray, formulas: list[str], topk: int):
    df = pd.DataFrame(
        {
            "formula": formulas,
            "y_true": y_true,
            col_name: score,
        }
    )
    df["rank_true"] = df["y_true"].rank(ascending=False, method="min")
    df["rank_pred"] = df[col_name].rank(ascending=False, method="min")
    df["hit"] = ((df["rank_true"] <= topk) & (df["rank_pred"] <= topk)).astype(int)
    df.sort_values("rank_pred").to_csv(out_dir / f"top{topk}_hit_table_{score_name}.csv", index=False)


def save_precision_plot(out_dir: Path, score_name: str, rows: list[dict]):
    pk_rows = [r for r in rows if r["score"] == score_name and r["metric"].startswith("Precision@")]
    pk_rows = sorted(pk_rows, key=lambda r: int(r["metric"].split("@")[1]))
    xs = [int(r["metric"].split("@")[1]) for r in pk_rows]
    ys = [r["mean"] for r in pk_rows]
    ylo = [r["ci_2p5"] for r in pk_rows]
    yhi = [r["ci_97p5"] for r in pk_rows]
    baseline = [r["random"] for r in pk_rows]

    plt.figure()
    plt.errorbar(
        xs,
        ys,
        yerr=[np.array(ys) - np.array(ylo), np.array(yhi) - np.array(ys)],
        fmt="o-",
        capsize=4,
        label=f"AG-MoE ({score_name})",
    )
    plt.plot(xs, baseline, "--", label="Random")
    plt.xlabel("K (Top-K selected)")
    plt.ylabel("Precision@K")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"precision_at_k_{score_name}.pdf")
    plt.savefig(out_dir / f"precision_at_k_{score_name}.png", dpi=300)
    plt.close()


def main():
    ap = argparse.ArgumentParser(
        description="Evaluate v17/v18/v23 AG-MoE on real Young's-modulus-like structure datasets."
    )
    ap.add_argument("--config", default=None, help="Stage2 config for v18/v23.")
    ap.add_argument("--ckpt", default=None, help="Stage2 ckpt for v18/v23.")
    ap.add_argument("--config_k", default=None, help="Stage1 config for v17 log_kvrh expert.")
    ap.add_argument("--ckpt_k", default=None, help="Stage1 ckpt for v17 log_kvrh expert.")
    ap.add_argument("--config_g", default=None, help="Stage1 config for v17 log_gvrh expert.")
    ap.add_argument("--ckpt_g", default=None, help="Stage1 ckpt for v17 log_gvrh expert.")
    ap.add_argument("--data_pt", default="data/hea_dataset0330_standard.pt")
    ap.add_argument("--model_version", choices=["v17", "v18", "v23"], default=None)
    ap.add_argument("--task_k", default=None, help="Defaults to auto-detected log_kvrh task.")
    ap.add_argument("--task_g", default=None, help="Defaults to auto-detected log_gvrh task.")
    ap.add_argument("--Ks", default="6,9")
    ap.add_argument("--deltas", default="0,0.05,0.08")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--topk_table", type=int, default=9)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    data_pt = Path(args.data_pt)
    version = infer_model_version(args.model_version, args.config or "", args.ckpt or "")
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else Path("results") / f"{data_pt.stem}_{version}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    ks = [int(x) for x in args.Ks.split(",") if x.strip()]
    deltas = [float(x) for x in args.deltas.split(",") if x.strip()]
    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    print(f"[Eval] device={device}")
    print(f"[Eval] model_version={version}")
    print(f"[Eval] data_pt={data_pt}")
    print(f"[Eval] out_dir={out_dir}")

    raw = torch.load(data_pt, map_location="cpu")
    y_true = np.array([float(item[1]) for item in raw], dtype=float)
    ds = StructureDataset(raw)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=pyg_collate)

    # 小数据集（n < 30）自动跳过 Precision@K，与 b2_18_fold_results 格式对齐
    if len(y_true) < 30:
        ks_eff: list[int] = []
        print(f"[Eval] n={len(y_true)} < 30 → small-dataset mode: Precision@K skipped")
    else:
        ks_eff = ks
    print(f"[Eval] n_samples={len(y_true)}, Ks={ks_eff}, deltas={deltas}")

    if version == "v17":
        if not all([args.config_k, args.ckpt_k, args.config_g, args.ckpt_g]):
            raise ValueError(
                "For v17 you must provide --config_k --ckpt_k --config_g --ckpt_g "
                "(Stage1 single-expert configs/checkpoints)."
            )
        config_k_path = Path(args.config_k)
        ckpt_k_path = Path(args.ckpt_k)
        config_g_path = Path(args.config_g)
        ckpt_g_path = Path(args.ckpt_g)
        print(f"[Eval] config_k={config_k_path}")
        print(f"[Eval] ckpt_k={ckpt_k_path}")
        print(f"[Eval] config_g={config_g_path}")
        print(f"[Eval] ckpt_g={ckpt_g_path}")

        cfg_k = yaml.safe_load(open(config_k_path, "r"))
        cfg_g = yaml.safe_load(open(config_g_path, "r"))
        task_k = args.task_k or "log_kvrh"
        task_g = args.task_g or "log_gvrh"
        model_k = build_v17_stage1_model(cfg_k, str(ckpt_k_path), raw, device)
        model_g = build_v17_stage1_model(cfg_g, str(ckpt_g_path), raw, device)
    else:
        if not args.config or not args.ckpt:
            raise ValueError("For v18/v23 you must provide --config and --ckpt.")
        config_path = Path(args.config)
        ckpt_path = Path(args.ckpt)
        print(f"[Eval] config={config_path}")
        print(f"[Eval] ckpt={ckpt_path}")

        cfg = yaml.safe_load(open(config_path, "r"))
        task_k = resolve_task_name(
            cfg["tasks"],
            args.task_k,
            ["log_kvrh", "matbench_log_kvrh"],
            "K-head",
        )
        task_g = resolve_task_name(
            cfg["tasks"],
            args.task_g,
            ["log_gvrh", "matbench_log_gvrh"],
            "G-head",
        )
        model = build_model(cfg, str(ckpt_path), version, device)
        model_k = model
        model_g = model

    print(f"[Eval] task_k={task_k}, task_g={task_g}")

    score_k_parts, score_g_parts, metas_all = [], [], []
    with torch.no_grad():
        for x, ei, ea, bi, yb, metas in loader:
            batch = {
                "x": x.to(device),
                "edge_index": ei.to(device),
                "edge_attr": ea.to(device),
                "batch": bi.to(device),
            }
            if version == "v17":
                score_k_parts.append(model_k(batch).detach().cpu().numpy().ravel())
                score_g_parts.append(model_g(batch).detach().cpu().numpy().ravel())
            else:
                score_k_parts.append(model_k(task_k, batch).detach().cpu().numpy().ravel())
                score_g_parts.append(model_g(task_g, batch).detach().cpu().numpy().ravel())
            metas_all.extend(metas)

    score_k = np.concatenate(score_k_parts)
    score_g = np.concatenate(score_g_parts)
    score_fused = fusion_softplus_phys(score_k, score_g)

    formulas = [
        meta.get("reduced_formula", "") if isinstance(meta, dict) else ""
        for meta in metas_all
    ]

    # metrics.json 只保留 fused_KG，与 hea_95_fold_results / b2_18_fold_results 格式对齐
    all_rows = build_metric_rows(
        "fused_KG", score_fused, y_true, ks_eff, deltas, args.n_boot, args.seed
    )
    # 额外记录单头指标（仅用于 preds.json 参考，不写入 metrics.json）
    _head_rows = {
        "log_kvrh_head": build_metric_rows(
            "log_kvrh_head", score_k, y_true, ks_eff, deltas, args.n_boot, args.seed
        ),
        "log_gvrh_head": build_metric_rows(
            "log_gvrh_head", score_g, y_true, ks_eff, deltas, args.n_boot, args.seed
        ),
    }

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(all_rows, f, indent=2)

    with open(out_dir / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["score", "metric", "mean", "ci_2p5", "ci_97p5", "random"],
            extrasaction="ignore",
            restval="",
        )
        writer.writeheader()
        writer.writerows(all_rows)

    preds = []
    for i, meta in enumerate(metas_all):
        preds.append(
            {
                "i": i,
                "formula": meta.get("reduced_formula", "") if isinstance(meta, dict) else "",
                "meta": meta if isinstance(meta, dict) else {},
                "y_true": float(y_true[i]),
                "score_k": float(score_k[i]),
                "score_g": float(score_g[i]),
                "score_fused": float(score_fused[i]),
            }
        )
    with open(out_dir / "preds.json", "w") as f:
        json.dump(preds, f, indent=2)

    save_topk_table(out_dir, "log_kvrh", "score_k", y_true, score_k, formulas, args.topk_table)
    save_topk_table(out_dir, "log_gvrh", "score_g", y_true, score_g, formulas, args.topk_table)
    save_topk_table(out_dir, "fused_KG", "score_fused", y_true, score_fused, formulas, args.topk_table)

    save_precision_plot(out_dir, "fused_KG", all_rows)

    # 单头辅助指标：写入独立文件，不影响主 metrics.json
    all_head_rows = _head_rows["log_kvrh_head"] + _head_rows["log_gvrh_head"]
    with open(out_dir / "metrics_heads.json", "w") as f:
        json.dump(all_head_rows, f, indent=2)
    save_precision_plot(out_dir, "log_kvrh_head", all_head_rows)
    save_precision_plot(out_dir, "log_gvrh_head", all_head_rows)

    summary = {
        "model_version": version,
        "task_k": task_k,
        "task_g": task_g,
        "n_samples": int(len(y_true)),
        "out_dir": str(out_dir),
    }
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("[Done] External real-data Young's modulus evaluation complete.")


if __name__ == "__main__":
    main()

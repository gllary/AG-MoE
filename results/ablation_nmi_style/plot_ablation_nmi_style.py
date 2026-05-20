from __future__ import annotations

import math
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
LOCAL_PY_PKGS = REPO_ROOT / "py_pkgs"
if LOCAL_PY_PKGS.exists():
    sys.path.insert(0, str(LOCAL_PY_PKGS))

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT_DIR = Path(__file__).resolve().parent

METHODS = ["No-sharing", "Fully joint", "Pure MoE", "Per-task gate", "AG-MoE"]

# Color-blind friendly palette with AG-MoE highlighted in a warmer accent.
METHOD_COLORS = {
    "No-sharing": "#6E6E6E",
    "Fully joint": "#4C78A8",
    "Pure MoE": "#54A24B",
    "Per-task gate": "#B279A2",
    "AG-MoE": "#D64F3A",
}

# Soft curve palette inspired by the reference ROC figure.
CURVE_COLORS = {
    "No-sharing": "#3E7CB1",
    "Fully joint": "#3DB27D",
    "Pure MoE": "#D64F4F",
    "Per-task gate": "#7A68B3",
    "AG-MoE": "#38B7C9",
}

TASK_LABELS = {
    "steels": "Steel Yield Strength",
    "glass": "Glass Forming Ability",
    "expt_gap": "Exp. Band Gap",
    "expt_is_metal": "Exp. Metallicity",
    "jdft2d": "JDFT-2D E_form",
    "dielectric": "Dielectric Const.",
    "log_kvrh": "Bulk Modulus (log K)",
    "log_gvrh": "Shear Modulus (log G)",
    "perovskites": "Perovskite E_form",
    "phonons": "Phonon Peak Freq.",
    "mp_gap": "MP Band Gap",
    "mp_is_metal": "MP Metallicity",
    "mp_e_form": "MP Formation E",
}


def build_data() -> pd.DataFrame:
    rows = []

    def add(task: str, group: str, metric: str, unit: str, higher_is_better: bool, values):
        for method, mean, std in values:
            rows.append(
                {
                    "task": task,
                    "task_label": TASK_LABELS[task],
                    "group": group,
                    "metric": metric,
                    "unit": unit,
                    "higher_is_better": higher_is_better,
                    "method": method,
                    "mean": mean,
                    "std": std,
                }
            )

    add(
        "steels",
        "Composition-based tasks",
        "MAE",
        "MPa",
        False,
        [
            ("No-sharing", 73.8809, 15.4175),
            ("Fully joint", 69.8233, 25.9167),
            ("Pure MoE", 69.8252, 23.5607),
            ("Per-task gate", 69.7682, 12.2045),
            ("AG-MoE", 69.7549, 11.9604),
        ],
    )
    add(
        "expt_gap",
        "Composition-based tasks",
        "MAE",
        "eV",
        False,
        [
            ("No-sharing", 0.4245, 0.0111),
            ("Fully joint", 0.2482, 0.0186),
            ("Pure MoE", 0.2483, 0.0169),
            ("Per-task gate", 0.2475, 0.0088),
            ("AG-MoE", 0.2480, 0.0086),
        ],
    )
    add(
        "glass",
        "Composition-based tasks",
        "AUC",
        "",
        True,
        [
            ("No-sharing", 0.9296, 0.0037),
            ("Fully joint", 0.9722, 0.0063),
            ("Pure MoE", 0.9768, 0.0057),
            ("Per-task gate", 0.9768, 0.0030),
            ("AG-MoE", 0.9768, 0.0029),
        ],
    )
    add(
        "expt_is_metal",
        "Composition-based tasks",
        "AUC",
        "",
        True,
        [
            ("No-sharing", 0.9463, 0.0039),
            ("Fully joint", 0.9673, 0.0065),
            ("Pure MoE", 0.9741, 0.0059),
            ("Per-task gate", 0.9741, 0.0031),
            ("AG-MoE", 0.9741, 0.0030),
        ],
    )
    add(
        "phonons",
        "Structure-based tasks",
        "MAE",
        "cm$^{-1}$",
        False,
        [
            ("No-sharing", 30.6806, 3.4554),
            ("Fully joint", 28.8963, 5.8085),
            ("Pure MoE", 28.5117, 5.2805),
            ("Per-task gate", 29.4037, 2.7353),
            ("AG-MoE", 26.3261, 2.6806),
        ],
    )
    add(
        "mp_gap",
        "Structure-based tasks",
        "MAE",
        "eV",
        False,
        [
            ("No-sharing", 0.2386, 0.0032),
            ("Fully joint", 0.1977, 0.0054),
            ("Pure MoE", 0.1221, 0.0049),
            ("Per-task gate", 0.2586, 0.0026),
            ("AG-MoE", 0.1244, 0.0025),
        ],
    )
    add(
        "mp_e_form",
        "Structure-based tasks",
        "MAE",
        "eV/atom",
        False,
        [
            ("No-sharing", 0.0298, 0.0004),
            ("Fully joint", 0.0782, 0.0007),
            ("Pure MoE", 0.2199, 0.0006),
            ("Per-task gate", 0.01638, 0.0003),
            ("AG-MoE", 0.0160, 0.0003),
        ],
    )
    add(
        "log_kvrh",
        "Structure-based tasks",
        "MAE",
        "log$_{10}$ GPa",
        False,
        [
            ("No-sharing", 0.0572, 0.0030),
            ("Fully joint", 0.0673, 0.0050),
            ("Pure MoE", 0.1223, 0.0045),
            ("Per-task gate", 0.0395, 0.0023),
            ("AG-MoE", 0.0391, 0.0023),
        ],
    )
    add(
        "log_gvrh",
        "Structure-based tasks",
        "MAE",
        "log$_{10}$ GPa",
        False,
        [
            ("No-sharing", 0.0790, 0.0009),
            ("Fully joint", 0.0926, 0.0015),
            ("Pure MoE", 0.1267, 0.0014),
            ("Per-task gate", 0.0568, 0.0007),
            ("AG-MoE", 0.0569, 0.0007),
        ],
    )
    add(
        "perovskites",
        "Structure-based tasks",
        "MAE",
        "eV/atom",
        False,
        [
            ("No-sharing", 0.0383, 0.0012),
            ("Fully joint", 0.0293, 0.0020),
            ("Pure MoE", 0.0255, 0.0018),
            ("Per-task gate", 0.0245, 0.0009),
            ("AG-MoE", 0.0242, 0.0009),
        ],
    )
    add(
        "jdft2d",
        "Structure-based tasks",
        "MAE",
        "meV/atom",
        False,
        [
            ("No-sharing", 33.3379, 11.2723),
            ("Fully joint", 33.1241, 18.9487),
            ("Pure MoE", 30.6433, 17.2261),
            ("Per-task gate", 33.4267, 8.9232),
            ("AG-MoE", 30.2535, 8.7447),
        ],
    )
    add(
        "dielectric",
        "Structure-based tasks",
        "MAE",
        "",
        False,
        [
            ("No-sharing", 0.2912, 0.0982),
            ("Fully joint", 0.2633, 0.1651),
            ("Pure MoE", 0.2636, 0.1501),
            ("Per-task gate", 0.2629, 0.0778),
            ("AG-MoE", 0.2627, 0.0762),
        ],
    )
    add(
        "mp_is_metal",
        "Structure-based tasks",
        "AUC",
        "",
        True,
        [
            ("No-sharing", 0.9619, 0.0021),
            ("Fully joint", 0.9722, 0.0035),
            ("Pure MoE", 0.9752, 0.0032),
            ("Per-task gate", 0.9759, 0.0016),
            # The prompt is truncated after "0.9759 ±"; keep the mean and omit the
            # error bar rather than inventing an uncertainty.
            ("AG-MoE", 0.9759, np.nan),
        ],
    )

    df = pd.DataFrame(rows)
    df["method"] = pd.Categorical(df["method"], METHODS, ordered=True)
    return df


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9.5,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 160,
            "savefig.dpi": 450,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_all(fig: mpl.figure.Figure, stem: str) -> None:
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT_DIR / f"{stem}.{ext}")
    plt.close(fig)


def add_panel_label(ax: mpl.axes.Axes, label: str) -> None:
    ax.text(
        -0.08,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="bottom",
        ha="right",
    )


def score_within_task(df: pd.DataFrame) -> pd.DataFrame:
    scored = []
    for _, task_df in df.groupby("task", sort=False):
        vals = task_df["mean"].to_numpy(dtype=float)
        high = bool(task_df["higher_is_better"].iloc[0])
        best = np.nanmax(vals) if high else np.nanmin(vals)
        worst = np.nanmin(vals) if high else np.nanmax(vals)
        denom = best - worst
        if math.isclose(denom, 0.0):
            score = np.ones_like(vals)
        elif high:
            score = (vals - worst) / denom
        else:
            score = (worst - vals) / (worst - best)
        tmp = task_df.copy()
        tmp["normalized_score"] = score
        scored.append(tmp)
    return pd.concat(scored, ignore_index=True)


def plot_normalized_heatmap(scored: pd.DataFrame) -> None:
    task_order = scored.drop_duplicates("task")["task"].tolist()
    matrix = (
        scored.pivot(index="task", columns="method", values="normalized_score")
        .loc[task_order, METHODS]
        .to_numpy()
    )

    fig, ax = plt.subplots(figsize=(5.9, 5.0))
    im = ax.imshow(matrix, cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(METHODS)), METHODS, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(task_order)), [TASK_LABELS[t] for t in task_order])
    ax.set_title("Task-wise normalized performance")

    for i, task in enumerate(task_order):
        task_df = scored[scored["task"] == task].set_index("method")
        high = bool(task_df["higher_is_better"].iloc[0])
        vals = task_df.loc[METHODS, "mean"].to_numpy()
        best = np.nanmax(vals) if high else np.nanmin(vals)
        for j, method in enumerate(METHODS):
            mean = task_df.loc[method, "mean"]
            txt = "*" if math.isclose(mean, best, rel_tol=1e-9, abs_tol=1e-12) else ""
            color = "white" if matrix[i, j] > 0.58 else "#1f1f1f"
            ax.text(j, i, txt, ha="center", va="center", color=color, fontweight="bold")

    ax.axhline(3.5, color="white", lw=2.0)
    ax.text(
        len(METHODS) - 0.5,
        1.5,
        "Composition",
        color="#444444",
        ha="left",
        va="center",
        rotation=-90,
        fontsize=8,
    )
    ax.text(
        len(METHODS) - 0.5,
        8.0,
        "Structure",
        color="#444444",
        ha="left",
        va="center",
        rotation=-90,
        fontsize=8,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.038, pad=0.03)
    cbar.set_label("Normalized score (best = 1)")
    add_panel_label(ax, "a")
    save_all(fig, "fig1_normalized_performance_heatmap")


def plot_vs_no_sharing(df: pd.DataFrame) -> None:
    rows = []
    for task, task_df in df.groupby("task", sort=False):
        base = task_df.loc[task_df["method"] == "No-sharing"].iloc[0]
        for _, row in task_df.iterrows():
            if row["method"] == "No-sharing":
                continue
            if row["higher_is_better"]:
                value = (row["mean"] - base["mean"]) * 100.0
                label = "AUC gain vs No-sharing (percentage points)"
            else:
                value = (base["mean"] - row["mean"]) / base["mean"] * 100.0
                label = "MAE reduction vs No-sharing (%)"
            rows.append({**row.to_dict(), "improvement": value, "improvement_label": label})
    rel = pd.DataFrame(rows)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.4, 4.7),
        sharex=False,
        gridspec_kw={"width_ratios": [0.78, 1.45]},
    )
    for ax, group, label in zip(
        axes,
        ["Composition-based tasks", "Structure-based tasks"],
        ["b", "c"],
    ):
        sub = rel[rel["group"] == group]
        task_order = sub.drop_duplicates("task")["task"].tolist()
        y = np.arange(len(task_order))
        offsets = np.linspace(-0.24, 0.24, len(METHODS) - 1)
        for offset, method in zip(offsets, METHODS[1:]):
            m = sub[sub["method"] == method].set_index("task").loc[task_order]
            ax.scatter(
                m["improvement"],
                y + offset,
                s=28,
                color=METHOD_COLORS[method],
                edgecolor="white",
                linewidth=0.45,
                label=method,
                zorder=3,
            )
        ax.axvline(0, color="#333333", lw=0.8)
        ax.set_yticks(y, [TASK_LABELS[t] for t in task_order])
        ax.invert_yaxis()
        ax.grid(axis="x", color="#E8E8E8", lw=0.7)
        ax.set_title(group.replace("-based tasks", ""))
        ax.set_xlabel("Improvement vs No-sharing\n(MAE: % reduction; AUC: percentage-point gain)")
        add_panel_label(ax, label)

    axes[1].legend(
        ncol=2,
        loc="lower right",
        frameon=False,
        handletextpad=0.4,
        columnspacing=1.0,
    )
    save_all(fig, "fig2_improvement_vs_no_sharing")


def plot_auc_only(df: pd.DataFrame) -> None:
    auc = df[df["metric"] == "AUC"].copy()
    task_order = auc.drop_duplicates("task")["task"].tolist()

    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    x = np.arange(len(task_order))
    width = 0.14
    offsets = (np.arange(len(METHODS)) - (len(METHODS) - 1) / 2) * width

    for offset, method in zip(offsets, METHODS):
        m = auc[auc["method"] == method].set_index("task").loc[task_order]
        yerr = m["std"].to_numpy(dtype=float)
        yerr = np.nan_to_num(yerr, nan=0.0)
        ax.bar(
            x + offset,
            m["mean"],
            width=width,
            color=METHOD_COLORS[method],
            edgecolor="white",
            linewidth=0.5,
            yerr=yerr,
            error_kw={"elinewidth": 0.75, "ecolor": "#333333", "capsize": 2.0},
            label=method,
            zorder=3,
        )

    ax.set_xticks(x, [TASK_LABELS[t] for t in task_order])
    ax.set_ylabel("AUC")
    ax.set_title("Classification tasks")
    ax.set_ylim(0.91, 0.985)
    ax.grid(axis="y", color="#ECECEC", lw=0.7, zorder=0)
    ax.legend(
        ncol=3,
        loc="lower right",
        frameon=False,
        handlelength=1.0,
        handletextpad=0.4,
        columnspacing=0.9,
    )
    add_panel_label(ax, "f")
    save_all(fig, "fig5_auc_tasks_only")


def plot_auc_only_curves(df: pd.DataFrame) -> None:
    auc = df[df["metric"] == "AUC"].copy()
    task_order = auc.drop_duplicates("task")["task"].tolist()

    fig, axes = plt.subplots(1, len(task_order), figsize=(8.4, 2.9), sharex=True)
    axes = np.asarray(axes).ravel()
    x = np.arange(len(task_order))
    method_x = np.arange(len(METHODS))
    marker_colors = [CURVE_COLORS[m] for m in METHODS]

    for ax, task in zip(axes, task_order):
        task_df = auc[auc["task"] == task].set_index("method").loc[METHODS]
        y = task_df["mean"].to_numpy(dtype=float)
        std = np.nan_to_num(task_df["std"].to_numpy(dtype=float), nan=0.0)
        ax.fill_between(
            method_x,
            y - std,
            y + std,
            color="#3E7CB1",
            alpha=0.10,
            linewidth=0,
            zorder=1,
        )
        ax.plot(
            method_x,
            y,
            linestyle="--",
            color="#4F4F4F",
            linewidth=1.45,
            zorder=2,
        )
        ax.scatter(
            method_x,
            y,
            s=28,
            color=marker_colors,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        ax.set_title(TASK_LABELS[task])
        ax.set_xticks(method_x, METHODS, rotation=35, ha="right")
        ax.set_ylabel("AUC")
        ax.set_ylim(0.91, 0.985)
        ax.grid(color="#E5E5E5", lw=0.8, alpha=0.85)

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=CURVE_COLORS[m],
            markeredgecolor="white",
            markersize=6,
            label=m,
        )
        for m in METHODS
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(METHODS),
        frameon=True,
        framealpha=0.92,
        edgecolor="#D8D8D8",
        fancybox=False,
        bbox_to_anchor=(0.5, 0.005),
    )
    fig.suptitle("AUC across classification tasks", y=0.995, fontsize=11)
    fig.tight_layout(rect=(0, 0.16, 1, 0.92), w_pad=1.1)
    add_panel_label(axes[0], "g")
    save_all(fig, "fig6_auc_tasks_curve")


def plot_auc_task_facets(df: pd.DataFrame) -> None:
    auc = df[df["metric"] == "AUC"].copy()
    task_order = auc.drop_duplicates("task")["task"].tolist()
    ncols = 2
    nrows = math.ceil(len(task_order) / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(7.2, 5.9), sharex=True)
    axes = np.asarray(axes).ravel()
    x = np.arange(len(METHODS))
    marker_colors = [CURVE_COLORS[m] for m in METHODS]

    global_min = float(np.nanmin(auc["mean"] - auc["std"].fillna(0.0)))
    global_max = float(np.nanmax(auc["mean"] + auc["std"].fillna(0.0)))
    pad = max(0.004, 0.10 * (global_max - global_min))
    ylo = max(0.90, global_min - pad)
    yhi = min(0.99, global_max + pad)

    for ax, task in zip(axes, task_order):
        task_df = auc[auc["task"] == task].set_index("method").loc[METHODS]
        y = task_df["mean"].to_numpy(dtype=float)
        std = np.nan_to_num(task_df["std"].to_numpy(dtype=float), nan=0.0)

        ax.fill_between(
            x,
            y - std,
            y + std,
            color="#3E7CB1",
            alpha=0.10,
            linewidth=0,
            zorder=1,
        )
        ax.plot(
            x,
            y,
            linestyle="--",
            color="#4F4F4F",
            linewidth=1.45,
            zorder=2,
        )
        ax.scatter(
            x,
            y,
            s=28,
            color=marker_colors,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        ax.set_title(TASK_LABELS[task])
        ax.set_ylabel("AUC")
        ax.set_ylim(ylo, yhi)
        ax.grid(color="#E5E5E5", lw=0.8, alpha=0.85)

    for ax in axes[len(task_order) :]:
        ax.axis("off")

    for ax in axes[-ncols:]:
        if ax.has_data():
            ax.set_xticks(x, METHODS, rotation=35, ha="right")

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=CURVE_COLORS[m],
            markeredgecolor="white",
            markersize=6,
            label=m,
        )
        for m in METHODS
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(METHODS),
        frameon=True,
        framealpha=0.92,
        edgecolor="#D8D8D8",
        fancybox=False,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.suptitle("AUC across classification tasks", y=0.995, fontsize=11)
    fig.tight_layout(rect=(0, 0.08, 1, 0.97), h_pad=1.5, w_pad=1.3)
    save_all(fig, "fig8_auc_task_facets")


def plot_mae_tasks_curves(df: pd.DataFrame) -> None:
    mae = df[df["metric"] == "MAE"].copy()
    task_order = mae.drop_duplicates("task")["task"].tolist()
    ncols = 2
    nrows = math.ceil(len(task_order) / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(7.6, 10.6), sharex=True)
    axes = np.asarray(axes).ravel()
    x = np.arange(len(METHODS))
    marker_colors = [CURVE_COLORS[m] for m in METHODS]

    for ax, task in zip(axes, task_order):
        task_df = mae[mae["task"] == task].set_index("method").loc[METHODS]
        y = task_df["mean"].to_numpy(dtype=float)
        std = task_df["std"].to_numpy(dtype=float)

        ax.fill_between(
            x,
            y - std,
            y + std,
            color="#3E7CB1",
            alpha=0.10,
            linewidth=0,
            zorder=1,
        )
        ax.plot(
            x,
            y,
            linestyle="--",
            color="#4F4F4F",
            linewidth=1.45,
            zorder=2,
        )
        ax.scatter(
            x,
            y,
            s=28,
            color=marker_colors,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        unit = task_df["unit"].iloc[0]
        ylabel = "MAE" if not unit else f"MAE ({unit})"
        ax.set_title(TASK_LABELS[task])
        ax.set_ylabel(ylabel)
        ax.grid(color="#E5E5E5", lw=0.8, alpha=0.85)

    for ax in axes[len(task_order) :]:
        ax.axis("off")

    for ax in axes[-ncols:]:
        ax.set_xticks(x, METHODS, rotation=35, ha="right")

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=CURVE_COLORS[m],
            markeredgecolor="white",
            markersize=6,
            label=m,
        )
        for m in METHODS
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(METHODS),
        frameon=True,
        framealpha=0.92,
        edgecolor="#D8D8D8",
        fancybox=False,
        bbox_to_anchor=(0.5, 0.005),
    )
    fig.suptitle("MAE across regression tasks", y=0.995, fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 0.985), h_pad=1.4, w_pad=1.3)
    save_all(fig, "fig7_mae_tasks_curves")


def plot_agmoe_vs_per_task_gate(df: pd.DataFrame) -> None:
    rows = []
    for task, task_df in df.groupby("task", sort=False):
        pt = task_df.loc[task_df["method"] == "Per-task gate"].iloc[0]
        ag = task_df.loc[task_df["method"] == "AG-MoE"].iloc[0]
        if bool(ag["higher_is_better"]):
            delta = (ag["mean"] - pt["mean"]) * 100.0
            unit = "AUC percentage points"
        else:
            delta = (pt["mean"] - ag["mean"]) / pt["mean"] * 100.0
            unit = "MAE reduction (%)"
        rows.append(
            {
                "task": task,
                "task_label": TASK_LABELS[task],
                "group": ag["group"],
                "delta": delta,
                "unit": unit,
            }
        )
    comp = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    x = np.arange(len(comp))
    colors = np.where(comp["delta"] >= 0, METHOD_COLORS["AG-MoE"], "#8A8A8A")
    ax.bar(x, comp["delta"], color=colors, width=0.7, edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="#333333", lw=0.8)
    ax.axvline(3.5, color="#D7D7D7", lw=1.0)
    ax.set_xticks(x, comp["task_label"], rotation=45, ha="right")
    ax.set_ylabel("AG-MoE gain over Per-task gate\n(MAE: % reduction; AUC: percentage-point gain)")
    ax.set_title("Effect of adaptive global sharing")
    ax.grid(axis="y", color="#ECECEC", lw=0.7)
    ax.text(1.5, ax.get_ylim()[1] * 0.94, "Composition", ha="center", va="top", color="#555555")
    ax.text(8.0, ax.get_ylim()[1] * 0.94, "Structure", ha="center", va="top", color="#555555")
    add_panel_label(ax, "d")
    save_all(fig, "fig3_agmoe_gain_over_per_task_gate")


def plot_best_method_counts(scored: pd.DataFrame) -> None:
    rows = []
    for task, task_df in scored.groupby("task", sort=False):
        high = bool(task_df["higher_is_better"].iloc[0])
        best_value = task_df["mean"].max() if high else task_df["mean"].min()
        winners = task_df[
            np.isclose(task_df["mean"], best_value, rtol=1e-9, atol=1e-12)
        ]["method"].tolist()
        for method in winners:
            rows.append({"task": task, "method": method})
    wins = pd.DataFrame(rows)
    counts = wins["method"].value_counts().reindex(METHODS, fill_value=0)

    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    ax.bar(
        np.arange(len(METHODS)),
        counts.loc[METHODS],
        color=[METHOD_COLORS[m] for m in METHODS],
        width=0.68,
        edgecolor="white",
        linewidth=0.6,
    )
    ax.set_xticks(np.arange(len(METHODS)), METHODS, rotation=35, ha="right")
    ax.set_ylabel("Number of best-task means")
    ax.set_title("Best mean performance count")
    ax.grid(axis="y", color="#ECECEC", lw=0.7)
    add_panel_label(ax, "e")
    save_all(fig, "fig4_best_method_counts")


def write_readme() -> None:
    text = """# Ablation visualization outputs

This folder contains NMI-style visualizations generated from the ablation table provided in the prompt.

Files:
- `ablation_raw_data.csv`: tidy version of the input table.
- `ablation_scored_data.csv`: table with task-wise normalized scores.
- `fig1_normalized_performance_heatmap.*`: normalized performance heatmap, where 1 is the best method within each task.
- `fig2_improvement_vs_no_sharing.*`: improvement over No-sharing; MAE tasks use percent MAE reduction, AUC tasks use percentage-point AUC gain.
- `fig3_agmoe_gain_over_per_task_gate.*`: AG-MoE gain relative to Per-task gate.
- `fig4_best_method_counts.*`: number of task-level best mean scores per method.
- `fig5_auc_tasks_only.*`: AUC-only grouped bar chart with standard-deviation error bars.
- `fig6_auc_tasks_curve.*`: AUC-only curve plot with standard-deviation bands.
- `fig7_mae_tasks_curves.*`: MAE-only facet curve plot with standard-deviation error bars.
- `fig8_auc_task_facets.*`: AUC-only facet curve plot using the same small-multiple style as Fig. 7.

Note: the prompt was truncated for `mp is metal` AG-MoE uncertainty (`0.9759 ± ...`), so the mean is kept and the standard deviation is stored as missing.
"""
    (OUT_DIR / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    set_style()
    df = build_data()
    scored = score_within_task(df)
    df.to_csv(OUT_DIR / "ablation_raw_data.csv", index=False)
    scored.to_csv(OUT_DIR / "ablation_scored_data.csv", index=False)
    plot_normalized_heatmap(scored)
    plot_vs_no_sharing(df)
    plot_auc_only(df)
    plot_auc_only_curves(df)
    plot_auc_task_facets(df)
    plot_mae_tasks_curves(df)
    plot_agmoe_vs_per_task_gate(df)
    plot_best_method_counts(scored)
    write_readme()
    print(f"Saved ablation figures to {OUT_DIR}")


if __name__ == "__main__":
    main()

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# =========================
# CONFIG
# =========================
LOG_PATH = Path("/home/tytadmin/gyx/model_agent_project/mat_models/checkpoints/v25_stage2_moe/train_log.jsonl")   # ← 改成你的 V25 日志路径
OUT_DIR = Path("v25_gate_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# LOAD JSONL
# =========================
records = []
with open(LOG_PATH, "r", encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

print(f"[INFO] Loaded {len(records)} epochs from {LOG_PATH}")

# =========================
# COLLECT GATE & ROUTER
# =========================
gate_mean = defaultdict(list)
gate_std = defaultdict(list)
router_avg = defaultdict(list)
epochs = []

for r in records:
    epoch = r["epoch"]
    epochs.append(epoch)

    gate = r.get("gate", {})
    router = r.get("router_avg", {})

    for task, g in gate.items():
        gate_mean[task].append(g["mean"])
        gate_std[task].append(g["std"])

    for task, w in router.items():
        # w is list of expert weights → mean routing entropy proxy
        router_avg[task].append(np.mean(w))

tasks = sorted(gate_mean.keys())
epochs = sorted(list(set(epochs)))

# =========================
# BUILD MATRICES
# =========================
G = np.zeros((len(tasks), len(epochs)))
S = np.zeros_like(G)
R = np.zeros_like(G)

for i, t in enumerate(tasks):
    G[i] = gate_mean[t]
    S[i] = gate_std[t]
    R[i] = router_avg.get(t, [0.0] * len(epochs))

# =========================
# 1️⃣ GATE HEATMAP
# =========================
plt.figure(figsize=(1.2 * len(epochs), 0.6 * len(tasks)))
im = plt.imshow(G, aspect="auto", cmap="viridis")
plt.colorbar(im, fraction=0.02, pad=0.02, label="Gate mean")
plt.yticks(range(len(tasks)), tasks)
plt.xticks(
    ticks=range(0, len(epochs), max(1, len(epochs)//10)),
    labels=[epochs[i] for i in range(0, len(epochs), max(1, len(epochs)//10))]
)
plt.xlabel("Epoch")
plt.ylabel("Task")
plt.title("V25 Adaptive Sharing Gate (mean)")
plt.tight_layout()
plt.savefig(OUT_DIR / "gate_heatmap.png", dpi=200)
plt.close()

# =========================
# 2️⃣ PER-TASK GATE CURVES
# =========================
for i, t in enumerate(tasks):
    plt.figure(figsize=(6, 3))
    plt.plot(epochs, G[i], label="mean", marker="o")
    plt.fill_between(
        epochs,
        G[i] - S[i],
        G[i] + S[i],
        alpha=0.3,
        label="±1 std"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Gate")
    plt.title(f"Gate Evolution: {t}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"gate_curve_{t}.png", dpi=200)
    plt.close()

# =========================
# 3️⃣ FINAL EPOCH RANKING
# =========================
final_gate = {t: gate_mean[t][-1] for t in tasks}
ranking = sorted(final_gate.items(), key=lambda x: x[1], reverse=True)

with open(OUT_DIR / "gate_final_ranking.txt", "w") as f:
    f.write("Final Epoch Gate Ranking (High → Low)\n")
    f.write("=" * 40 + "\n")
    for t, g in ranking:
        f.write(f"{t:25s} {g:.4f}\n")

# =========================
# 4️⃣ GATE STATS SUMMARY CSV
# =========================
rows = []
for t in tasks:
    rows.append({
        "task": t,
        "gate_final": gate_mean[t][-1],
        "gate_mean_all": float(np.mean(gate_mean[t])),
        "gate_std_all": float(np.mean(gate_std[t])),
    })

df_gate = pd.DataFrame(rows)
df_gate.to_csv(OUT_DIR / "gate_stats_summary.csv", index=False)

# =========================
# 5️⃣ GATE × ROUTER JOINT ANALYSIS
# =========================
rows = []
for t in tasks:
    rows.append({
        "task": t,
        "gate_final": gate_mean[t][-1],
        "router_mean": float(np.mean(router_avg.get(t, [0.0]))),
    })

df_joint = pd.DataFrame(rows)
df_joint.to_csv(OUT_DIR / "gate_router_joint.csv", index=False)

# =========================
# DONE
# =========================
print(f"[DONE] All gate analysis results saved to: {OUT_DIR.resolve()}")
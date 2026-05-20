# coNGN 基线复现

在 Matbench `log_gvrh` 上训练当前 SOTA 模型 coNGN，  
然后 **零样本迁移** 到 95 个 HEA 样本，与 AG-MoE 做对比。

## 目录结构

```
cogn_baseline/
├── setup_env.sh          # 安装 TF + kgcnn 到 mp310 环境
├── train_cogn_gvrh.py    # 5-fold 训练 coNGN on log_gvrh
├── infer_cogn_hea.py     # 推理 95 HEA + 计算排名指标
├── plot_cogn_vs_agmoe.py # 生成对比可视化图
├── run_all.sh            # 一键运行全流程
└── README.md
```

## 快速开始

### 第一步：安装依赖（只需做一次）

```bash
cd cogn_baseline
bash setup_env.sh
```

### 第二步：训练 coNGN

```bash
/Users/yuxia.guan/miniconda3/envs/mat_env/bin/python train_cogn_gvrh.py
```

**预计耗时**：
- CPU：12-24 小时
- GPU（单卡）：2-4 小时

输出：`cogn_weights/fold_{0..4}/weights.keras`

### 第三步：HEA 推理

```bash
/Users/yuxia.guan/miniconda3/envs/mat_env/bin/python infer_cogn_hea.py
```

输出：
- `cogn_hea_results/hea_predictions.json`
- `cogn_hea_results/hea_metrics.json`

### 第四步：生成对比图

```bash
/Users/yuxia.guan/miniconda3/envs/mat_env/bin/python plot_cogn_vs_agmoe.py
```

输出：`Article/fig_cogn_vs_agmoe.pdf`

### 一键运行

```bash
bash run_all.sh --setup   # 首次，包含依赖安装
bash run_all.sh            # 后续运行
```

## 预期结果

| 指标 | coNGN（预期） | AG-MoE |
|------|:---:|:---:|
| SpearmanR | ~0.7–0.8 | **0.918** |
| KendallTau | ~0.5–0.6 | **0.756** |
| Precision@9/10 | ~0.4–0.5 | **0.707** |
| PairwiseAcc | ~0.6–0.7 | **0.756** |

## 说明

- **为什么 coNGN 在 HEA 上会比 AG-MoE 差？**  
  coNGN 是单任务模型，只从 `log_gvrh`（剪切模量）学习。AG-MoE 通过多任务学习，  
  同时从 bulk/shear/elastic moduli 等相关任务获益，形成更鲁棒的结构表征。
  
- **为什么用 log_gvrh → 预测 Young's modulus？**  
  Young's modulus (E) 与 shear modulus (G) 高度相关：  
  E ≈ 2G(1 + ν)（ν≈0.3 for metals），因此 log_gvrh_head 做零样本迁移合理。

- **KGCNN 版本要求**：kgcnn >= 3.0.0
- **TF 版本**：2.13.x（Python 3.10 最稳定）

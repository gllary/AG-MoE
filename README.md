# AG-MoE

**Adaptive Gating Mixture-of-Experts for Multi-Task Materials Property Prediction**

This repository accompanies the manuscript

> *Learning when to share: disentangling representation formation and adaptive sharing for diverse materials property prediction.*
> Yuxia Guan, Weijiang Zhao, Hao Zou, Guangfang Chi, Yong Liu, Jianxin Wang.

## Overview

Materials property prediction across multiple tasks involves data with different distributions and physical mechanisms, so it is not obvious when representations should be shared. AG-MoE treats representation sharing as a **task- and input-dependent decision** rather than a fixed architectural choice. The framework is organised in two stages:

- **Stage 1 — specialist experts.** One expert is trained per task to learn task-grounded composition or structure representations.
- **Stage 2 — adaptive sharing.** The experts are frozen and a task-conditioned router with an instance-level gate learns, for each input, how strongly to mix specialised versus shared representations.

The model is evaluated under the standard Matbench v0.1 protocol on all 13 benchmark tasks and on two zero-shot transfer cohorts:

- a **95-alloy curated HEA** dataset for Young's-modulus screening, and
- an **18-alloy experimental B2 MPEI** cohort synthesised and characterised in this study.

## Repository layout

```
AG-MoE/
├── code/                    Implementation and analysis scripts
│   ├── mat_models/          Composition / structure encoders, MoE block, adaptive gate
│   ├── configs/             Training configuration files
│   ├── analysis/            Fold-level transfer evaluations
│   ├── cogn_baseline/       Reproduced coNGN baseline for the transfer experiments
│   └── results/             Aggregated per-task summary JSONs
├── results/                 Plot scripts and rendered figures
└── transfer data/           Curated transfer datasets
    ├── hea_dataset0330.json                  95-alloy HEA dataset (BCC, refractory)
    └── matbench_Young's_modulus/             18 B2 MPEI records (this work)
```

## Datasets

- **Matbench v0.1** — accessed through the official [`matbench`](https://matbench.materialsproject.org/) package; we follow the standard five-fold split.
- **HEA transfer cohort** — 95 BCC alloys aggregated from the literature, with sources documented alongside the JSON file.
- **B2 MPEI transfer cohort** — 18 ordered intermetallics measured in a single internally consistent workflow described in the manuscript.

## Citation

If you find this work useful, please cite the paper:

```bibtex
@article{guan2025agmoe,
  title   = {Learning when to share: disentangling representation formation and adaptive sharing for diverse materials property prediction},
  author  = {Guan, Yuxia and Zhao, Weijiang and Zou, Hao and Chi, Guangfang and Liu, Yong and Wang, Jianxin},
  year    = {2025}
}
```

## Contact

For questions about the work, please contact the corresponding authors at the affiliations listed in the manuscript.

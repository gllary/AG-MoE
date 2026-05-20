# Ablation visualization outputs

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

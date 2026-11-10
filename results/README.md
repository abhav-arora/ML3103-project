# Official test results

The comparison covers 16 model entries, four C-MAPSS subsets and three
seeds (42, 43 and 44). Each prediction is in engine cycles. Within each
subset, the comparison notebook verified that all models scored the same
engines, last observed cycles and official RUL labels.

- [Mean and sample standard deviation](test_metrics_mean_sd.csv) contains 128 rows.
- [Per-seed metrics and parameter counts](test_metrics_per_seed.csv) contains 384 rows.

Both tables include raw and capped target protocols. Use `target=raw` for
the main official test comparison. MAE and RMSE are in cycles, MSE is in
cycles squared, and RAE is relative absolute error in percent. Positive
signed bias means optimistic predictions. The C-MAPSS score is summed
over engines, so compare it within the same subset and target protocol.

The [comparison notebook](../notebooks/Test_Comparison.ipynb) recomputes the
tables from frozen predictions. It also produces the
[RMSE rankings](official_figures/rmse_rankings.pdf),
[RAE heatmap](official_figures/rae_heatmap.pdf),
[student comparison](official_figures/student_comparison.pdf)
and [residual plots](official_figures/residuals.pdf).
The [Drive comparison folder](https://drive.google.com/drive/folders/1f9AQGvz8me3L7h7ipy5WlOrTC7TrGGkw)
also has per-engine predictions. Training histories and checkpoints are in
the [baseline run](https://drive.google.com/drive/folders/1yykVSjciVI8hXerI1dvqoOby2kqvUB4J)
and [student run](https://drive.google.com/drive/folders/1OPVkcJRBbuDFMgmCZG90m0IWoUEovAfS).

The [ablation summary](ablations/ablation_mean_sd.csv), [per-seed results](ablations/ablation_per_seed.csv)
and [paired RMSE plot](ablations/ablation_rmse_deltas.png) show the separate hidden-state and
attention-loss removals. The matched ablation control was retrained and is a separate run
from the original official comparison above. See the [explanation figures](explanations/),
[seed-42 spike diagnostics](test_spike_diagnostics_seed42.json) and [data audit](data_audit/audit_summary.json).

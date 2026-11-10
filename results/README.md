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
[RMSE rankings](https://drive.google.com/file/d/1WunQ2A6FAxhCtEY5f4R6HlNNngNPgDhv/view),
[RAE heatmap](https://drive.google.com/file/d/1Hr5YxVDW7jcdCLjCBAuGoTqhJmXp1vlP/view),
[student comparison](https://drive.google.com/file/d/18BhegpC1mFKmOxF9SXeuIL_9Yq8tVwWj/view)
and [residual plots](https://drive.google.com/file/d/1nlTbHICTXBpMfx4bif7wUSQxf83V4GRM/view).
The [Drive comparison folder](https://drive.google.com/drive/folders/1f9AQGvz8me3L7h7ipy5WlOrTC7TrGGkw)
also has per-engine predictions. Training histories and checkpoints are in
the [baseline run](https://drive.google.com/drive/folders/1yykVSjciVI8hXerI1dvqoOby2kqvUB4J)
and [student run](https://drive.google.com/drive/folders/1OPVkcJRBbuDFMgmCZG90m0IWoUEovAfS).

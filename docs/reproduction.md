# Reproducing the experiments

Run commands from the repository root. The input folder must contain
`train_FD00X.txt`, `test_FD00X.txt` and `RUL_FD00X.txt` for each subset.
The [Drive C-MAPSS folder](https://drive.google.com/drive/folders/14fs5y_xmMtPGETm6vlt1R3vc9XbmvG78)
contains the archive used for the saved runs. The dataset creation and
asymmetric score are described in the [NASA benchmark paper](https://ntrs.nasa.gov/api/citations/20090029214/downloads/20090029214.pdf).

Install the data requirements from `requirements-data.txt`. The model
scripts additionally require PyTorch. The explanation script requires
Matplotlib. The original Colab notebooks record their runtime versions and
GPU in the Drive run logs.

```text
python -m spikedistill.data.prepare --data-dir /path/to/CMaps --output-dir artifacts/data
python -m spikedistill.training.baselines --data-root artifacts/data --output artifacts/baselines --subsets FD001 FD002 FD003 FD004
python -m spikedistill.training.distill --data-root artifacts/data --baselines artifacts/baselines --output artifacts/students
```

After training, repeat the corresponding baseline and distillation commands
with `--phase test`. Both runners select checkpoints by capped validation
RMSE and save raw and capped test metrics in engine cycles.

For an explanation run on frozen test checkpoints:

```text
python -m spikedistill.evaluation.explain --data-root artifacts/data --baselines artifacts/baselines --students artifacts/students
```

For the two one-loss ablations, rerun the full student alongside both removals
on the same runtime. The inference architecture, teacher, data, seed and
30-epoch budget match within each comparison. Only the named loss weight is
zeroed. Attention pooling stays in all three students.

```text
python -m spikedistill.training.distill --data-root artifacts/data --baselines artifacts/baselines --output artifacts/ablations --variants spikedistill spike_no_hidden spike_no_attention
python -m spikedistill.training.distill --data-root artifacts/data --baselines artifacts/baselines --output artifacts/ablations --variants spikedistill spike_no_hidden spike_no_attention --phase test
python -m spikedistill.evaluation.ablations --roots artifacts/ablations --output artifacts/ablation_report
```

If your prepared NPZ files come from a different snapshot of the same official
data, the original checkpoint byte hashes may differ. Add
`--verify-prepared-predictions` to training and explanation commands to require
matching saved teacher validation predictions and exact endpoints when the
hashes differ. The explanation command additionally checks the saved student
test predictions. Keep the corresponding prediction CSVs beside their metrics.
This check establishes numerical agreement on the evaluated windows.

The completed Drive runs store code and data fingerprints. Use the
[baseline notebook](../notebooks/Baselines.ipynb) with the original
`baselines_v1/git_revision.txt` for baseline checkpoint recovery. The
[distillation notebook](../notebooks/Distillation.ipynb) reconstructs the
original student script and reuses that pinned baseline revision for
`spikedistill_v1`. The reorganized Python modules are entry points for
new runs, with their own checkpoint directories.

The [official comparison notebook](../notebooks/Test_Comparison.ipynb)
reads the existing Drive predictions and requires only a CPU runtime.
It verifies held-out engine alignment and saved metrics before exporting
tables and plots.

# SpikeDistill-RUL: implementation roadmap

GitHub is the source of truth for project code and notebooks. The original
C-MAPSS data and experiment outputs are stored separately in Google Drive.

## Colab workflow

Open `ML_project.ipynb` in Colab, select an available GPU and run the cells
in order. The notebook clones this repository and mounts Drive using your
Google account. It expects `MyDrive/ML projec/CMaps` and saves logs, plots,
and checkpoints under `MyDrive/ML projec/SpikeDistill-RUL/colab_runs`.
Each pilot run records its Git commit so a replacement runtime can resume
with the same source code.

The first milestone is a two-epoch FD001 pilot followed by a checkpoint
recovery trial in a fresh runtime. GPU execution and Drive recovery are
pending until that trial runs. The pilot uses capped validation targets
and is separate from the final model comparisons.

To rebuild the tracked Colab notebook after changing its setup, run
`python build_colab.py` from the repository.

## Baseline experiments

Use `Baselines.ipynb` for the full comparison. It pins a Git commit for each
Drive run, prepares the selected subsets and saves resumable training
checkpoints. Start with FD001 and then run FD002–FD004. The pilot remains
separate.

The suite contains the mean predictor, DeepNN, WideNN, RNN, LSTM, GRU,
BiLSTM, CNN–LSTM, the attention BiLSTM teacher, an output-only distilled
small LSTM, a supervised recurrent LIF SNN and a rate-converted DeepNN.
DeepNN uses four ReLU layers (128, 96, 64, 32). WideNN uses one 256-unit
ReLU layer. The spiking baseline uses direct input current, fixed leak
and threshold, and a fast-sigmoid surrogate gradient. ANN-to-SNN
conversion reuses the selected DeepNN and calibrates thresholds using
training activations only. Its default is 32 simulation steps.

All neural fits use the same engine splits, 30-cycle inputs, batch size
128, Adam at 0.001, 30 epochs and seeds 42, 43 and 44 by default. The
teacher is trained first and frozen for output-only distillation. The
student minimizes equally weighted ground-truth and teacher-output MSE.
The mean predictor and converted model have no separate optimizer.
Model size and total compute can differ despite the common epoch budget.

Training targets are capped RUL divided by the configured cap. Before
scoring, the runner multiplies predictions by that cap to restore cycles.
It uses the same predictions for capped and uncapped comparisons, with
no prediction clipping. Validation checkpoint selection uses capped RMSE.
The official test phase runs separately after configuration is frozen,
with one prediction per engine. Report raw official labels as the primary
test protocol and capped scores separately. MAE/RMSE are in cycles,
MSE is in cycles squared, and R² can be negative. The asymmetric score
is summed over engines. Compare it within each subset and target protocol.

Run locally with PyTorch installed:

```powershell
python run_baselines.py --data-root artifacts/data --output artifacts/baselines --subsets FD001 --phase train
```

Use the same arguments with `--phase test` only after freezing the
experiment. JSON summaries include seed mean and sample standard
deviation. Per-engine prediction CSVs preserve both targets and the
unclipped prediction for independent metric checks.

## The 12 steps

| Step | Work | Status |
|---|---|---|
| 1 | Audit raw files, engine histories, RUL alignment and source hashes | Complete |
| 2 | Create engine-disjoint splits, chronological prefixes and capped/raw RUL labels | Complete |
| 3 | Fit train-only feature selection and normalization; operating regimes for FD002/FD004 | Complete |
| 4 | Build 30-cycle windows, padding masks and reproducible data artifacts | Complete |
| 5 | Exploratory plots and supervised-training environment setup | Next |
| 6 | Shared trainer, metrics and conventional baselines | Implemented; full sweep pending |
| 7 | Attention-BiLSTM teacher | Pending |
| 8 | Recurrent LIF spiking student without distillation | Pending |
| 9 | Output, membrane-state and attention distillation; sparsity loss | Pending |
| 10 | Three-seed comparisons, ablations and explainability | Pending |
| 11 | Local benchmarking and actual Jetson measurement if access is obtained | Pending |
| 12 | Reproducible notebook, eight-page report and ten-slide presentation | Pending |

## Run

Use Python with the packages in requirements-data.txt. The development runtime already contains them.
From this directory:

```powershell
python -m unittest -v test_prepare_data
python prepare_data.py --data-dir '..\CMaps'
```

Open Project.ipynb for an executed walkthrough. For FD001 only, append --subsets FD001.
The default output directory is relative to the current working directory; run from this project directory.

## Fixed starting protocol

- Split seed 42; 20% of official training engines held out for validation.
- Training and validation engine IDs never overlap; cycles remain chronological.
- Training uses all complete 30-cycle windows, stride one.
- Validation uses one incomplete prefix per held-out engine, selected reproducibly from 40%-90% of its lifetime. This is a fixed synthetic validation design, not a claim to reproduce the official test truncation distribution.
- Test uses one final observed window per official test engine. No test tuning or model selection.
- Labels at training/validation cycle t: final training trajectory cycle minus t.
- Test endpoint labels come exclusively from the official RUL file, not the final observed test cycle.
- Starting RUL cap 125. Both y (capped) and y_raw (uncapped) are saved; metric tables must explicitly name which protocol they use.
- Sensor selection uses training-only standard deviation > 1e-8. Engine ID and cycle are metadata, not inputs.
- FD001/FD003: training-only global sensor standardization.
- FD002/FD004: six operating regimes from training-only k-means on standardized operating settings, followed by per-regime sensor standardization. The same learned centers/statistics transform validation/test.
- Inputs include retained sensors and three standardized operating settings.
- Short evaluation histories are left-padded by repeating the first observed row. mask=False identifies padding. Future sequence models and attribution methods must respect masks; do not feed masked padding through recurrence unconditionally.
- Each subset has its own preprocessing and models; engine IDs are local to that subset.

## Artifacts

artifacts/data/<subset>/ contains train.npz, validation.npz, test.npz, preprocessing.json, split_manifest.json and audit.json.
Each archive contains X, mask, y, y_raw, engine_id, end_cycle and valid_length.
X is float32 with shape [examples, observed cycles, features].
preprocessing.json contains feature order and fitted normalization parameters.
split_manifest.json contains exact engine IDs, validation endpoints, configuration and SHA-256 hashes of source files.
audit_summary.json combines all four subsets.

## Data finding requiring provenance confirmation

The supplied FD004 files contain 248 test engines and 248 matching RUL labels. The bundled readme says 249 test trajectories. Preparation retains the actual files and records their hashes; it does not invent an engine or label. Confirm the archive provenance against an authoritative copy before claiming exact official-benchmark coverage.

## Verification and remaining limits

Seven tests cover split isolation/reproducibility, training-only scaling, window chronology, target construction, prefix leakage, padding, external test labels and malformed trajectories. All four actual subsets were prepared successfully. These checks verify data preparation, not predictive accuracy.

No model has been trained. PyTorch is not installed in the runtime used for preparation. A training environment and plotting dependencies still need setup. Actual Jetson measurements remain unavailable without device access or an explicitly approved alternative.

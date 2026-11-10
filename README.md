# SpikeDistill-RUL

Predict turbofan remaining useful life from 30-cycle NASA C-MAPSS sensor
windows. The project compares recurrent, feed-forward and spiking baselines
with a recurrent spiking student distilled from an attention BiLSTM.

**Start here for the presentation.** The [demo guide](docs/demo-guide.md)
lists the Colab notebooks in order, the exact code to show, and where to find
the trained models and saved results. The [final report](docs/SpikeDistill-RUL_Final_Report.pdf)
contains the diagrams and experiment discussion.

| Folder | Contents |
|---|---|
| [spikedistill/data](spikedistill/data/) | C-MAPSS audit, engine splits, normalization and windowing |
| [spikedistill/models](spikedistill/models/) | Baseline architectures, attention teacher and learnable LIF student |
| [spikedistill/training](spikedistill/training/) | Pilot, baseline and student training, checkpoint selection and held-out scoring |
| [spikedistill/evaluation](spikedistill/evaluation/) | Regression metrics and student explanation methods |
| [notebooks](notebooks/) | Colab data walkthrough, baseline and student training, official comparison, ablations and XAI |
| [models](models/) | Original validation-selected seed-42 teacher and student weights for FD001–FD004 |
| [results](results/) | Official test tables and plots, ablations, validation histories and explanation figures |
| [deployment/jetson](deployment/jetson/) | ONNX model and tests, TensorRT script and PyTorch board benchmark |
| [docs](docs/) | Model configurations and reproducibility notes |
| [tests](tests/) | Data, metric, model and checkpoint tests |
| [tools](tools/) | Pilot notebook generator |

Start with the [official test results](results/README.md) or the
[model configuration table](docs/model-configurations.md).
Raw C-MAPSS files, prepared training arrays and complete epoch-by-epoch
run archives remain in [Google Drive](https://drive.google.com/drive/folders/1VPsHX0HvLAXKWoisPdk71c_kLW3fuFtX).

## Colab workflow

Follow the [presentation order](docs/demo-guide.md) to inspect the finished
project. The [pilot notebook](notebooks/ML_project.ipynb) is an optional
FD001 recovery trial. For a fresh pilot, select a GPU and run the cells
in order. The notebook clones this repository and mounts Drive using your
Google account. It expects `MyDrive/ML projec/CMaps` and saves logs, plots,
and checkpoints under `MyDrive/ML projec/SpikeDistill-RUL/colab_runs`.
Each pilot run records its Git commit so a replacement runtime can resume
with the same source code.

The two-epoch FD001 pilot ran on a Colab GPU and saved checkpoints to
Drive. The baseline sweep later resumed after an interruption and finished
across all four subsets. The pilot uses capped validation targets and is
separate from the baseline comparison.

To rebuild the tracked Colab notebook after changing its setup, run
`python tools/build_colab.py` from the repository.

## Baseline experiments

Use [the baseline notebook](notebooks/Baselines.ipynb) for training. It pins a Git commit for each
Drive run, prepares the selected subsets and saves resumable training
checkpoints. It runs FD001–FD004 by default. To extend an existing run,
keep its run name, seeds, epochs and other training settings fixed and
add the missing subsets. Completed models reuse their saved weights.
The pilot remains separate.

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
python -m spikedistill.training.baselines --data-root artifacts/data --output artifacts/baselines --subsets FD001 --phase train
```

Use the same arguments with `--phase test` only after freezing the
experiment. JSON summaries include seed mean and sample standard
deviation. Per-engine prediction CSVs preserve both targets and the
unclipped prediction for independent metric checks.

## Baseline validation results

The sweep finished on FD001–FD004 with seeds 42, 43, and 44. Drive contains
all 120 final training checkpoints and epoch histories, plus 144 validation
metric and prediction pairs. Each trainable run completed 30 epochs. The mean
predictor and ANN-to-SNN conversion contribute evaluation results without
separate training loops.

The table reports capped validation RMSE in engine cycles as mean ± sample
standard deviation across the three seeds. Lower is better. Bold marks the
lowest mean for each subset. The completed [official test results](results/README.md)
include these baselines and the four spiking student variants.

| Model | FD001 | FD002 | FD003 | FD004 |
|---|---:|---:|---:|---:|
| Mean predictor | 33.19 ± 0.00 | 33.01 ± 0.00 | 34.82 ± 0.00 | 35.57 ± 0.00 |
| DeepNN | 13.61 ± 0.18 | 18.28 ± 0.09 | **10.53 ± 1.27** | 14.92 ± 0.14 |
| WideNN | **13.27 ± 0.13** | **17.58 ± 0.93** | 11.76 ± 0.57 | 14.70 ± 0.89 |
| RNN | 16.51 ± 0.79 | 19.34 ± 1.13 | 15.71 ± 1.51 | 16.18 ± 0.77 |
| LSTM | 17.86 ± 0.89 | 18.24 ± 1.48 | 14.84 ± 1.92 | 14.86 ± 0.87 |
| GRU | 15.02 ± 0.65 | 17.65 ± 0.92 | 12.64 ± 0.59 | 14.23 ± 1.17 |
| BiLSTM | 16.61 ± 0.97 | 17.59 ± 0.73 | 11.42 ± 0.52 | 20.36 ± 1.12 |
| CNN–LSTM | 18.38 ± 1.57 | 18.37 ± 0.17 | 15.43 ± 2.18 | 16.22 ± 0.72 |
| Attention BiLSTM teacher | 14.73 ± 0.71 | 18.07 ± 0.59 | 12.36 ± 0.32 | 18.71 ± 0.69 |
| Small LSTM with output distillation | 16.86 ± 1.61 | 18.06 ± 0.58 | 14.15 ± 0.23 | **13.41 ± 0.93** |
| Supervised recurrent LIF SNN | 21.19 ± 2.25 | 20.60 ± 0.60 | 22.97 ± 0.81 | 22.04 ± 1.07 |
| ANN-to-SNN conversion | 39.10 ± 5.03 | 36.40 ± 8.09 | 26.27 ± 4.32 | 34.61 ± 6.10 |

The [full validation summary](https://drive.google.com/file/d/1g8_gloA6GTzYuzYmOudwr4auW_rng26K/view)
contains MAE, MSE, RMSE, R² and the C-MAPSS asymmetric score under both capped
and raw target protocols. The [run folder](https://drive.google.com/drive/folders/1yykVSjciVI8hXerI1dvqoOby2kqvUB4J)
contains per-engine predictions and epoch reports.

## The 12 steps

| Step | Work | Status |
|---|---|---|
| 1 | Audit raw files, engine histories, RUL alignment and source hashes | Complete |
| 2 | Create engine-disjoint splits, chronological prefixes and capped/raw RUL labels | Complete |
| 3 | Fit train-only feature selection and normalization; operating regimes for FD002/FD004 | Complete |
| 4 | Build 30-cycle windows, padding masks and reproducible data artifacts | Complete |
| 5 | Exploratory plots and supervised-training environment setup | Complete: plots, GPU pilot and Drive checkpoints saved |
| 6 | Shared trainer, metrics and conventional baselines | Complete: four subsets, three seeds and all validation outputs |
| 7 | Attention-BiLSTM teacher | Complete: trained for every subset and seed |
| 8 | Recurrent LIF spiking student without distillation | Complete: fixed and learnable variants trained and evaluated |
| 9 | Output, membrane-state and attention distillation; sparsity loss | Complete: four student variants trained across all subsets and seeds |
| 10 | Three-seed comparisons, ablations and explainability | Complete. Separate hidden-state and attention loss removals evaluated on FD001–FD004 across three seeds with a matched full control. [Results, figures and reproduction package](https://drive.google.com/file/d/1eNAg-ofuiLBHtJ5IxPIQOnA-uDpXWHKA/view?usp=drivesdk) |
| 11 | Local benchmarking and actual Jetson measurement if access is obtained | FD001 ONNX export and CPU verification complete. Physical Orin Nano measurements pending. |
| 12 | Reproducible notebook, eight-page report and ten-slide presentation | Eight-page report and staged reproduction notebooks available. Single executed notebook and slide deck pending. |

## Run

Use Python with the packages in requirements-data.txt. The development runtime already contains them.
From this directory:

```powershell
python -m unittest -v tests.test_prepare_data tests.test_metrics
python -m spikedistill.data.prepare --data-dir '..\CMaps'
```

Open [the data preparation walkthrough](notebooks/Project.ipynb) for an overview.
For FD001 only, append `--subsets FD001`.
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

The completed sweep includes baseline and spiking-student validation and
official test results. Jetson measurements and final submission artifacts
remain pending.

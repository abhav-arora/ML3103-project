# Research extensions

These experiments extend the assignment's prescribed SpikeDistill architecture with additional questions about prediction stability, student capacity and missing sensors. Their outcomes belong to a separate experimental phase. The ensemble and width-sweep methods have established precedents in machine learning. A contribution from this project is the defined experiment, its controlled protocol and the evidence it produces.

## 1. Ensemble stability

**Gap in our existing evaluation.** We trained separate seeds and reported their spread. We had not tested whether averaging their predictions produces a better deployable predictor.

**Question.** Can averaging frozen students improve raw RUL accuracy and the NASA asymmetric score over a single model selected using validation data?

Compare equal averaging and inverse-validation-MSE weighting. Both use nonnegative weights summing to one. The reference is the single member with lowest capped validation MSE. This is useful because a test-ranked choice would give the single model access to evaluation labels.

The first experiment uses three full students from the matched ablation runs, seeds 42, 43 and 44. It produces one ensemble for each subset. It reuses existing trained models. It adds no ensemble training epochs. The ensemble holds three sets of inference parameters. Its speed and energy cost require measurement.

The same runner also supports a heterogeneous ensemble. Supply the prediction folders of an attention BiLSTM, GRU and SpikeDistill from the same subset and seed. Fix this member list before scoring the test set. Examine its validation complementarity first.

Fit the weights before running the test phase. The fit command reads validation files only.

```bash
python -m research_extensions.ensemble.run --members /path/to/seed_42/spikedistill /path/to/seed_43/spikedistill /path/to/seed_44/spikedistill --output research_runs/ensemble/FD001 --phase fit
python -m research_extensions.ensemble.run --members /path/to/seed_42/spikedistill /path/to/seed_43/spikedistill /path/to/seed_44/spikedistill --output research_runs/ensemble/FD001 --phase test
```

The plan records weights and validation hashes. The test phase verifies matching members, endpoint/label alignment and unchanged validation files. Raw and capped scores are computed from cycle-scale predictions without clipping.

[model.py](ensemble/model.py) loads those same selected checkpoints as an executable PyTorch ensemble. Its forward method averages the normalized member outputs. Multiply the result by the shared cap to obtain cycles. Use `load_ensemble(member_folders, plan, weighting="validation")` with the fitted plan and the original member folders. The loader supports the recurrent student variants and trainable conventional baselines. Mean prediction and ANN-to-SNN conversion use their dedicated paths in the original benchmark.

[Hinton et al., knowledge distillation and ensembles](https://arxiv.org/abs/1503.02531) · [Lakshminarayanan et al., deep ensembles](https://arxiv.org/abs/1612.01474)

## 2. Student capacity

**Gap in our existing evaluation.** The 48-neuron width was fixed. The earlier results do not establish whether that size offers a good accuracy-size trade-off.

**Question.** How much capacity does a spiking student need to benefit from the same teacher?

Train widths 16, 32, 48 and 64 with the same full loss, teacher, data, 30 epochs, batch size 128 and Adam learning rate 0.001. The membrane adapter changes from width → 96. It is excluded from inference parameters. At each width, choose the checkpoint using capped validation RMSE.

For F input features and W neurons, inference parameters are W² + (F + 5)W + 1. FD001 has 18 features, giving 625, 1,761, 3,409 and 5,569 parameters for the four widths. The experiment tests both smaller students and additional capacity.

```bash
python -m research_extensions.student_capacity.run --data-root artifacts/data --baselines artifacts/baselines --output research_runs/capacity --verify-prepared-predictions
python -m research_extensions.student_capacity.run --data-root artifacts/data --baselines artifacts/baselines --output research_runs/capacity --verify-prepared-predictions --phase test
```

Defaults cover all four subsets and seeds 42–44. Checkpoints save every epoch and resume with a matching configuration. The first local sweep covers FD001 and seed 42. Multi-seed, multi-subset conclusions require completing the full sweep.

## 3. Sensor-outage robustness

**Gap in our existing evaluation.** The official test comparison assumes every retained sensor is available. Sensor permutation explained importance, but did not directly compare teacher and student under a missing-sensor scenario.

**Question.** How do clean performance and outage sensitivity differ between the teacher and student?

For each retained sensor, replace its standardized trace with zero throughout the window. This represents imputation using the training mean, including regime-specific sensor means. Keep the other sensors, operating settings and cycle validity mask fixed. Evaluate the frozen models on identical affected windows.

Report absolute raw/capped scores and changes from each model's clean RMSE and NASA score. A smaller change can coexist with a worse absolute score, so inspect both. Test every sensor using the fixed protocol. The first run uses seed 42 across all four subsets.

```bash
python -m research_extensions.robustness.run --data-root artifacts/data --baselines artifacts/baselines --students artifacts/students --output research_runs/robustness --verify-prepared-predictions
```

## Showing and reproducing the work

Use [Research_Extensions.ipynb](notebooks/Research_Extensions.ipynb) for the saved displays and optional Colab runs. The [results folder](results/) holds evidence for the local experiments. These results are reported separately from the original experiment in the project report.

Use the evidence to explain which hypothesis was supported, its size or accuracy cost, and which conclusions need more seeds. A successful research extension can reveal that an expected improvement fails.

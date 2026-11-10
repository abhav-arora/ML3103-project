# Initial extension results

The extension protocol was fixed before its test runs. All test values below use uncapped official labels and predictions in engine cycles. The member checkpoints and the student-width checkpoints were selected using capped validation RMSE. These experiments extend the earlier results and have their own coverage limits.

## Seed ensembles

The three members are the full student controls from the matched ablation experiment, seeds 42, 43 and 44. Each ensemble predicts by averaging three frozen models. Weights use capped validation MSE. Each subset contributes one ensemble result. The reference is the single member with the lowest validation MSE.

| Subset | Validation-selected single | Equal ensemble | Validation-weighted ensemble |
|---|---:|---:|---:|
| FD001 | 15.970 | 16.014 | 15.899 |
| FD002 | 29.646 | 28.879 | 28.886 |
| FD003 | 20.365 | 19.033 | 19.060 |
| FD004 | 28.652 | 28.130 | 28.128 |

The validation-weighted ensemble had lower raw test RMSE than the validation-selected single member on all four subsets. Equal averaging increased FD001 RMSE slightly. Averaging required three inference models. The run does not establish statistical reliability across independent ensembles. Compare the [complete metrics](ensemble_summary.csv), including NASA scores and inference parameter counts, and the frozen plans in [seed_ensemble](seed_ensemble/).

## Student width

Four new FD001 students were trained for 30 epochs at seed 42 with the same frozen teacher and full loss. This is a single-seed capacity experiment.

| Width | Inference parameters | Selected epoch | Raw test RMSE | NASA score |
|---|---:|---:|---:|---:|
| 16 | 625 | 24 | 17.750 | 641.24 |
| 32 | 1,761 | 28 | 17.302 | 527.11 |
| 48 | 3,409 | 27 | 15.970 | 689.19 |
| 64 | 5,569 | 14 | 18.322 | 735.22 |

The 48-neuron run had the lowest validation and test RMSE. The 32-neuron run had the lowest NASA score and used about 48% fewer inference parameters than the 48-neuron control. Increasing width to 64 increased RMSE in this run. Complete the three-seed, four-subset sweep before concluding that a width is generally preferable. [Metric table](capacity_summary.csv) · [Selected weights](../models/student_capacity/)

## Simulated sensor outages

These runs use the original teacher and original student from seed 42 on all four subsets. Each retained sensor is replaced in turn by its standardized training mean for the entire window. Operating settings and validity masks are kept.

| Subset | Teacher clean RMSE | Teacher largest RMSE increase | Student clean RMSE | Student largest RMSE increase |
|---|---:|---:|---:|---:|
| FD001 | 15.729 | +1.945, sensor 02 | 17.548 | +4.138, sensor 14 |
| FD002 | 29.698 | +2.666, sensor 14 | 28.962 | +3.932, sensor 02 |
| FD003 | 14.166 | +5.696, sensor 12 | 20.402 | +7.258, sensor 12 |
| FD004 | 27.695 | +3.201, sensor 11 | 28.492 | +5.975, sensor 11 |

The student's largest increase was greater than the teacher's on every subset under this imputation scenario. The most harmful sensor differed on FD001 and FD002. These are sensitivity findings from frozen models and a specified outage simulation. The columns report each model's largest increase, which can involve different sensors. Use the [per-sensor tables and plots](sensor_outage/) to compare them under identical outages.

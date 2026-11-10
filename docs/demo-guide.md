# SpikeDistill-RUL demo guide

**Start with [Project.ipynb in Colab](https://colab.research.google.com/github/abhav-arora/ML3103-project/blob/main/notebooks/Project.ipynb).** Open the remaining tabs in the order below. The completed figures, scores and selected models are available in this repo. For a short demo, inspect the original training cells and run the lightweight displays.

## Open the Colab tabs in this order

| Order | Open in Colab | What to show |
|---|---|---|
| 1 · Data | [Project.ipynb](https://colab.research.google.com/github/abhav-arora/ML3103-project/blob/main/notebooks/Project.ipynb) | Run the lightweight cells. Show the four subsets, engine-level split, train-only normalization, 30-cycle window, and left-padding mask. The FD001 window has 18 features per cycle. |
| 2 · Baselines | [Baselines.ipynb](https://colab.research.google.com/github/abhav-arora/ML3103-project/blob/main/notebooks/Baselines.ipynb) | Show the run setup and checkpoint commands. The notebook prepares data and trains the baseline suite, including the attention BiLSTM teacher. Point to FD001-FD004 and seeds 42, 43 and 44. Training caps and scales RUL. Checkpoints are chosen on validation data. |
| 3 · Proposed model | [Distillation.ipynb](https://colab.research.google.com/github/abhav-arora/ML3103-project/blob/main/notebooks/Distillation.ipynb) | Show the teacher and student definitions, then the loss calculation. The frozen teacher supplies its prediction, hidden states and attention. The 48-neuron spiking student learns from these signals and the ground truth. Show that the teacher and the training adapter are outside the deployed student. |
| 4 · Official results | [Test_Comparison.ipynb](https://colab.research.google.com/github/abhav-arora/ML3103-project/blob/main/notebooks/Test_Comparison.ipynb) | With your Drive mounted, this CPU notebook reads saved predictions, checks the official test engines and recomputes all metrics in cycles. For a short demo, show the metric code and open the saved [RMSE ranking](../results/official_figures/rmse_rankings.pdf) and [results table](../results/test_metrics_mean_sd.csv) from GitHub. |
| 5 · Ablations and XAI | [Ablations_and_Explanations.ipynb](https://colab.research.google.com/github/abhav-arora/ML3103-project/blob/main/notebooks/Ablations_and_Explanations.ipynb) | Run the first two code cells on CPU. Show that the full model and two variants differ in one loss at a time. Show the FD003 RMSE change after removing hidden matching, then the FD001 teacher/student attention and spike figures. Leave the optional reproduction flags disabled during the demo. |

[ML_project.ipynb](https://colab.research.google.com/github/abhav-arora/ML3103-project/blob/main/notebooks/ML_project.ipynb) is the earlier FD001 pilot. Open it only if asked about Colab setup or resuming after a runtime disconnect. It demonstrates Drive persistence and a short two-epoch run.

## Finish in GitHub

1. Open [the saved teacher and student checkpoints](../models/README.md). The FD001 student checkpoint was selected at epoch 14 on validation data. The other subsets each have their own model and feature layout.
2. Open [the deployment folder](../deployment/jetson/README.md). Show the FD001 ONNX model, its inputs and the CPU verification report. The ONNX file accepts a `[1, 30, 18]` window and a `[1, 30]` mask, then returns RUL in cycles. The physical Orin Nano measurements are pending.
3. Open [the final PDF](SpikeDistill-RUL_Final_Report.pdf). Page 3 has baseline architecture diagrams, page 4 has the teacher-student diagram, and pages 5-8 cover results, ablations, explanations and deployment.

For the additional research, open [Research_Extensions.ipynb](https://colab.research.google.com/github/abhav-arora/ML3103-project/blob/main/research_extensions/notebooks/Research_Extensions.ipynb)
after the original experiment. It shows the [research questions](../research_extensions/README.md),
frozen student ensembles, the four-width FD001 experiment and sensor-outage tests.
The width sweep is a single-seed initial study, and the sensor study uses seed 42.

## Code map if the TA asks how it works

| Question | File |
|---|---|
| How are engines split and input windows made? | [prepare.py](../spikedistill/data/prepare.py) |
| Where are the baseline layers defined? | [models/baselines.py](../spikedistill/models/baselines.py) |
| Where does the student fire and use attention? | [models/distillation.py](../spikedistill/models/distillation.py) |
| How are the baseline and student trained? | [training/baselines.py](../spikedistill/training/baselines.py) and [training/distill.py](../spikedistill/training/distill.py) |
| Where are the metrics and explanations calculated? | [metrics.py](../spikedistill/evaluation/metrics.py), [ablations.py](../spikedistill/evaluation/ablations.py), and [explain.py](../spikedistill/evaluation/explain.py) |
| Which files test deployment? | [test_onnx.py](../deployment/jetson/onnx/test_onnx.py) for a CPU check, [test_jetson.sh](../deployment/jetson/onnx/test_jetson.sh) for TensorRT on the board, and [benchmark.py](../deployment/jetson/benchmark.py) for the PyTorch board benchmark |

**One sentence to close with.** SpikeDistill improved on the fixed LIF baseline across all four subsets and used about 89% fewer parameters than the teacher, while conventional models led the official mean test RMSE.

## Where the work is saved

GitHub holds the code, notebooks, report, official metric tables and plots, ablation tables and histories, explanation figures, selected seed-42 models, and the runnable ONNX demo. Drive holds [the raw dataset](https://drive.google.com/drive/folders/14fs5y_xmMtPGETm6vlt1R3vc9XbmvG78), prepared data, [the complete Colab runs](https://drive.google.com/drive/folders/1VPsHX0HvLAXKWoisPdk71c_kLW3fuFtX), and the [full ablation archive](https://drive.google.com/file/d/1eNAg-ofuiLBHtJ5IxPIQOnA-uDpXWHKA/view). The [reproduction guide](reproduction.md) covers rerunning the experiments.

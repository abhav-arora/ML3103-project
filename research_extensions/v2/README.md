# Research Models V2

Can stronger teachers improve a fixed 64-unit recurrent spiking model? We compare a Li-style DCNN, an attention BiGRU, and their equal prediction average. The same spiking architecture is trained four ways: supervised, DCNN guidance, BiGRU guidance, and both teachers. Guided loss is 60% MSE against the true training label and 40% MSE against frozen teacher predictions on training windows.

| Paper | FD001 | FD002 | FD003 | FD004 |
| --- | ---: | ---: | ---: | ---: |
| Li, Ding & Sun, original DCNN experiment | 12.61 capped; 13.32 raw | — | — | — |
| Fu et al., BiGRU with self-attention, Table 8 | 11.8 | 14.2 | 13.9 | 15.6 |

These are published reference values in cycles, not V2 measurements. [Li et al. (2018)](https://doi.org/10.1016/j.ress.2017.11.021) use 14 selected features, 30-cycle windows, four length-10 convolution layers followed by a length-3 layer, and a 100-unit head. The commonly reproduced four-number DCNN row appears in [a later benchmark, Table 4](https://www.sciencedirect.com/science/article/pii/S156849462501316X); the original Li paper reports one benchmark experiment. The V2 model retains the layer layout and uses our train-fitted features. [Fu et al. (2026)](https://doi.org/10.1038/s41598-026-46237-y) also use a constructed health index and trajectory alignment. V2 adapts bidirectional GRU and temporal self-attention to our shared sensor inputs. Bidirectionality operates only inside the observed window.

Both teachers and all students use capped training labels and MSE. Selection uses uncapped RMSE on held-out training engines. V2 and V1 differ in training recipe as well as teacher architecture; compare guided vs. supervised **within V2** to isolate teacher guidance. The official test stage checks the frozen selection and full engine coverage. Predictions are converted to cycles before computing RMSE, MAE, MSE, RAE, R², bias and the NASA asymmetric score. Raw and capped targets are kept separate.

Run the [Colab notebook](../notebooks/Research_Models_V2.ipynb) with a GPU, or run these commands from the repository:

```bash
python -m research_extensions.v2.run --data-root artifacts/data --output research_runs/teachers_v2 --phase train
python -m research_extensions.v2.run --data-root artifacts/data --output research_runs/teachers_v2 --phase test
```

Defaults: FD001–FD004, seed 42, 50 teacher epochs, 30 student epochs. Epochs are saved to Drive in Colab. Keep the source, run options and output directory consistent when resuming. A new seed set or training budget should use a fresh output folder.

## Teacher tuning for FD002 and FD004

[Teacher_Tuning_FD002_FD004.ipynb](../notebooks/Teacher_Tuning_FD002_FD004.ipynb) runs a teacher-only experiment with Conv1D, a two-layer bidirectional GRU, and additive attention. The anchor uses the supplied 96 units per direction, dropout 0.25 and a 30-cycle input. The sweep tests longer windows, causal smoothing, hidden width, dropout, learning rate, and training label caps. The long windows and smoothing are intended to help the model track sensor drift across changes in operating condition.

Normalization and operating regime clustering are fitted on training engines only. The smoothing at each cycle uses that cycle and its observed history. Each held-out validation engine contributes up to four observed prefixes at fixed fractions of its trajectory, and the official test contributes one last observed window per engine. Checkpoint and hyperparameter selection minimize uncapped validation RMSE. On test, uncapped RMSE is the primary metric, and capped RMSE uses the project-wide cap of 125 for comparison. Candidate configurations are bundled, so the sweep can identify a useful recipe but cannot isolate the contribution of every change.

```bash
python -m research_extensions.v2.tune_bigru --data-root /path/to/CMAPSS --output research_runs/bigru_teacher --phase tune
python -m research_extensions.v2.tune_bigru --data-root /path/to/CMAPSS --output research_runs/bigru_teacher --phase test
```

The default sweep uses five candidates, 24 epochs each, and seed 42. Use a fresh output directory for a different sweep. Drive-backed Colab checkpoints resume with the same code, runtime and settings.

The [completed CPU sweep](../results/additive_bigru_teacher/local/) used three candidates (anchor30, smooth60, raw60), 12 epochs each, and seed 42. FD002 selected smooth60 at epoch 11 and scored 21.61 raw cycles against 28.20 for WideNN and 25.63 for the prior V1 GRU. FD004 selected raw60 at epoch 12 and scored 31.58 raw cycles against 26.29 for the original LSTM and 26.90 for the V2 BiGRU. Keep the original LSTM as the strongest measured FD004 model. Each subset used one seed, and the recipes change multiple settings.

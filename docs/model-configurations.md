# Model configurations

The official experiment used 30-cycle windows with stride 1. FD001–FD004
retain 18, 24, 19 and 24 features per cycle. Trainable runs used 30 epochs,
batch size 128, Adam at 0.001, gradient norm clipping at 1.0, and seeds 42,
43 and 44. The engine split was fixed at seed 42. Targets were capped at
125 cycles and divided by 125 during training. Checkpoint selection used
capped validation RMSE. The main test comparison uses raw RUL in cycles.

| Model key | Configuration | FD001 | FD002 | FD003 | FD004 |
|---|---|---:|---:|---:|---:|
| `mean` | Constant training-label mean | 0 | 0 | 0 | 0 |
| `deepnn` | Flatten → ReLU 128 → 96 → 64 → 32 → scalar | 89,953 | 112,993 | 93,793 | 112,993 |
| `widenn` | Flatten → ReLU 256 → scalar | 138,753 | 184,833 | 146,433 | 184,833 |
| `rnn` | One 48-unit tanh RNN | 3,313 | 3,601 | 3,361 | 3,601 |
| `lstm` | One 48-unit LSTM | 13,105 | 14,257 | 13,297 | 14,257 |
| `gru` | One 48-unit GRU | 9,841 | 10,705 | 9,985 | 10,705 |
| `bilstm` | One BiLSTM with 48 units per direction | 26,209 | 28,513 | 26,593 | 28,513 |
| `cnn_lstm` | Conv1D 48 channels, kernel 3, padding 1 → ReLU → 48-unit LSTM | 21,505 | 22,369 | 21,649 | 22,369 |
| `attention_bilstm` | 48 units per direction; attention scorer 96 → 48 → 1 | 30,914 | 33,218 | 31,298 | 33,218 |
| `small_lstm_kd` | 16-unit LSTM; equal label and teacher-output MSE weights | 2,321 | 2,705 | 2,385 | 2,705 |
| `snn` | 48 LIF neurons; fixed leak 0.9, threshold 1.0 | 3,265 | 3,553 | 3,313 | 3,553 |
| `ann_to_snn` | DeepNN weights, training-only threshold calibration, 32 simulation steps | 89,953 | 112,993 | 93,793 | 112,993 |
| `learnable_snn` | 48 recurrent LIF neurons, learnable leak and threshold; supervised | 3,409 | 3,697 | 3,457 | 3,697 |
| `spike_output_kd` | Same student with output matching | 3,409 | 3,697 | 3,457 | 3,697 |
| `spike_temporal_kd` | Same student with output, membrane and attention matching | 3,409 | 3,697 | 3,457 | 3,697 |
| `spikedistill` | Temporal KD plus mean firing-rate penalty | 3,409 | 3,697 | 3,457 | 3,697 |

Counts are inference parameters verified from the [saved per-seed table](../results/test_metrics_per_seed.csv).
The converted SNN shares DeepNN weights and keeps calibrated thresholds as
buffers. The student starts at leak 0.9 and threshold 1.0 and learns both
per neuron. It uses direct sensor current, recurrent spikes, and temporal
attention over membrane states. Both recurrent SNN designs use a hard
forward threshold and a fast-sigmoid surrogate with slope 10. One state
update corresponds to one observed engine cycle.

| Student | Label MSE | Teacher-output MSE | Membrane MSE | Attention MSE | Firing-rate penalty |
|---|---:|---:|---:|---:|---:|
| Learnable supervised | 1.0 | 0 | 0 | 0 | 0 |
| Output-only KD | 0.5 | 0.5 | 0 | 0 | 0 |
| Temporal KD | 0.5 | 0.5 | 0.1 | 0.1 | 0 |
| SpikeDistill-RUL | 0.5 | 0.5 | 0.1 | 0.1 | 0.01 |

Temporal matching trains a 48 → 96 projection with 4,704 parameters.
The projection and frozen teacher are excluded from student inference.

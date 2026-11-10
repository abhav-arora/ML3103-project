# Selected trained models

These are the original validation-selected Colab checkpoints for seed 42. Each `.pt` file contains the saved weights and training state. The adjacent JSON records the selected epoch, and the CSV gives the corresponding predictions in engine cycles.

| Subset | Attention BiLSTM teacher | SpikeDistill-RUL student |
|---|---|---|
| FD001 | [Teacher](baselines/FD001/seed_42/attention_bilstm/) | [Student](students/FD001/seed_42/spikedistill/) |
| FD002 | [Teacher](baselines/FD002/seed_42/attention_bilstm/) | [Student](students/FD002/seed_42/spikedistill/) |
| FD003 | [Teacher](baselines/FD003/seed_42/attention_bilstm/) | [Student](students/FD003/seed_42/spikedistill/) |
| FD004 | [Teacher](baselines/FD004/seed_42/attention_bilstm/) | [Student](students/FD004/seed_42/spikedistill/) |

The executable FD001 student export and its inputs are in [ONNX deployment](../deployment/jetson/onnx/). The complete training archive, including other seeds, intermediate epochs and baseline families, is in [baseline runs](https://drive.google.com/drive/folders/1yykVSjciVI8hXerI1dvqoOby2kqvUB4J) and [student runs](https://drive.google.com/drive/folders/1OPVkcJRBbuDFMgmCZG90m0IWoUEovAfS).

# Jetson Orin Nano deployment

| Path | Use |
|---|---|
| [ONNX deployment](onnx/README.md) | Original FD001 seed 42 student export, preprocessing settings, test inputs, and TensorRT smoke test |
| [benchmark.py](benchmark.py) | Batch-one PyTorch accuracy, latency, memory, power and spike diagnostics |
| [jetson_power.py](jetson_power.py) | Power sampling from the board's readable input rail |

From the repository root, you can run the ONNX correctness test with Python, NumPy and ONNX Runtime. This works on a CPU and requires neither training nor a Drive mount.

```bash
python -m pip install numpy onnxruntime
python deployment/jetson/onnx/test_onnx.py
```

On a physical Orin Nano with JetPack and TensorRT tools, run `bash deployment/jetson/onnx/test_jetson.sh` to build and time the TensorRT engine. The [saved test report](onnx/local_test_report.json) records Windows CPU ONNX Runtime results.

For the original PyTorch teacher and student, use the included [selected models](../../models/README.md) and the prepared FD001 `test.npz` and `split_manifest.json` under `MyDrive/ML projec/SpikeDistill-RUL/artifacts/data/FD001`. Run from the repository root in a JetPack-compatible PyTorch environment.

```bash
python3 -m deployment.jetson.benchmark \
  --data-root /path/to/artifacts/data \
  --baselines-root models/baselines \
  --students-root models/students \
  --subset FD001 --seed 42 --models attention_bilstm spikedistill
```

For a CPU portability check, add `--device cpu --power-seconds 0`. For a broader sweep, point the roots at the complete [baseline runs](https://drive.google.com/drive/folders/1yykVSjciVI8hXerI1dvqoOby2kqvUB4J) and [student runs](https://drive.google.com/drive/folders/1OPVkcJRBbuDFMgmCZG90m0IWoUEovAfS) and use `--models all`. The mean baseline also requires the matching original training archive. On-board timing, memory and power measurements are pending.

# SpikeDistill-RUL ONNX deployment

This export uses the original FD001 SpikeDistill-RUL student, seed 42, selected at validation epoch 14. The ONNX file contains the trained weights and the complete prediction graph. It uses FP32, ONNX opset 17 and a fixed 30-cycle window for one engine at a time. The other C-MAPSS subsets require their own trained weights and feature layouts.

## Input and output

| Tensor | Type and shape | Meaning |
| --- | --- | --- |
| `sensor_window` | float32, `[1, 30, 18]` | Prepared FD001 features, oldest to newest. Follow the feature order and training-set normalization in `FD001_preprocessing.json`. |
| `valid_mask` | float32, `[1, 30]` | Use `1` for observed cycles and `0` for left padding. Supply at least one observed cycle. For short histories, repeat the first observed row to fill the padding, as in the project data preparation. |
| `rul_cycles` | float32, `[1]` | Estimated remaining useful life in cycles. Output scaling is inside the ONNX graph. |

The Jetson benchmark package includes the original preprocessing script. For inference on raw sensor readings, reproduce that feature preparation before calling this ONNX model. The model expects standardized features in the saved order.

## Repeat the local correctness check

```bash
python -m pip install numpy onnxruntime
python test_onnx.py --output my_cpu_test.json
```

The test compares ONNX Runtime with PyTorch on all 100 official FD001 windows and an extra left-padding case. It computes raw and capped test metrics and measures CPU batch-1 latency over 1,000 calls after 100 warm-ups. The included `local_test_report.json` records the Windows CPU run. Its maximum absolute prediction difference was about 0.000046 cycles against a 0.001-cycle tolerance. The raw test RMSE was 17.54765 cycles.

## Build and test on Jetson Orin Nano

On the board, unzip this package and run:

```bash
bash test_jetson.sh
```

The script uses the board's JetPack TensorRT tools to build an FP32 engine with TF32 disabled. It prints PyTorch reference outputs and runs two real-input smoke cases, one using left padding. Compare the reported `rul_cycles` values to the references before using the engine. The batch-1 latency run uses the real FD001 input and saves the median and p95 in the log along with the timing samples and available board metadata. The TensorRT latency scope and backend differ from the earlier PyTorch benchmark, so label them separately in the report. Build the engine on the target board with its installed TensorRT version.

The ONNX graph and the CPU predictions were verified locally. TensorRT parsing, prediction parity and performance still need to be checked on a physical Orin Nano.

[NVIDIA TensorRT benchmarking and real input files](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/benchmarking.html)

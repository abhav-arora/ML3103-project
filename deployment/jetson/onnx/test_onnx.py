"""Verify FD001 predictions and optionally measure batch-1 ONNX Runtime latency."""
import argparse
import json
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import onnxruntime as ort


HERE = Path(__file__).resolve().parent


def metrics(y, prediction):
    y = y.astype(np.float64)
    error = prediction - y
    mse = float(np.mean(error ** 2))
    centered = y - y.mean()
    return dict(MAE=float(np.abs(error).mean()), MSE=mse, RMSE=mse ** 0.5,
                R2=1 - float((error ** 2).sum() / (centered ** 2).sum()),
                RAE_pct=float(100 * np.abs(error).sum() / np.abs(centered).sum()),
                cmapss_score=float(np.expm1(np.where(error < 0, -error / 13, error / 10)).sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=1000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.runs < 1000:
        parser.error("Use at least 1000 timed calls")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(HERE / "SpikeDistill_RUL_FD001_seed42.onnx"),
                                   sess_options=options, providers=["CPUExecutionProvider"])
    with np.load(HERE / "test_vectors.npz", allow_pickle=False) as data:
        feeds = [dict(sensor_window=x[None], valid_mask=mask[None])
                 for x, mask in zip(data["sensor_window"], data["valid_mask"])]

        def predict(feed):
            return session.run(["rul_cycles"], feed)[0].item()

        prediction = np.array([predict(feed) for feed in feeds], dtype=np.float64)
        reference = data["expected_cycles"]
        np.testing.assert_allclose(prediction, reference, rtol=0, atol=0.001)
        real_count = len(data["engine_id"])
        result = dict(provider=session.get_providers()[0], onnxruntime=ort.__version__,
                      official_test_cases=real_count, extra_mask_cases=len(feeds)-real_count,
                      max_absolute_difference_cycles=float(np.abs(prediction - reference).max()),
                      tolerance_cycles=0.001,
                      raw=metrics(data["y_raw"], prediction[:real_count]),
                      capped=metrics(data["y_capped"], prediction[:real_count]))
    for i in range(100):
        predict(feeds[i % real_count])
    timings = []
    for i in range(args.runs):
        start = perf_counter_ns()
        predict(feeds[i % real_count])
        timings.append((perf_counter_ns() - start) / 1e6)
    median, p95, p99 = np.percentile(timings, [50, 95, 99]).tolist()
    result["cpu_latency_ms"] = dict(warmup=100, measured=args.runs, median=median, p95=p95, p99=p99)
    report = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()

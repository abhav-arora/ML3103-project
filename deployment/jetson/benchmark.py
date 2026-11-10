"""Benchmark frozen C-MAPSS models on Jetson Orin Nano."""

import argparse
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter_ns

import numpy as np
import torch
from torch import nn

from .jetson_power import find_rail, measure
from spikedistill.evaluation.metrics import regression_metrics
from spikedistill.models.baselines import MODELS, ConvertedSNN, make_model
from spikedistill.models.distillation import Student


STUDENTS = ("learnable_snn", "spike_output_kd", "spike_temporal_kd", "spikedistill")
SUBSETS = ("FD001", "FD002", "FD003", "FD004")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, columns, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)


def file_hash(path):
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_report(folder):
    # The test report records the validation-selected epoch too. Use it for parity when present.
    for name in ("test_metrics.json", "validation_metrics.json"):
        path = folder / name
        if path.is_file():
            return read_json(path)
    raise FileNotFoundError(f"No saved metrics in {folder}")


def load_test(directory):
    manifest = read_json(directory / "split_manifest.json")
    cap = float(manifest["rul_cap"])
    path = directory / "test.npz"
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in
                ("X", "mask", "y", "y_raw", "engine_id", "end_cycle")}
    x, mask, engines = data["X"], data["mask"], data["engine_id"]
    if (x.ndim != 3 or x.shape[1] != manifest["window"] or not len(x)
            or mask.shape != x.shape[:2] or mask.dtype != np.bool_
            or not mask.any(axis=1).all() or (mask[:, :-1] & ~mask[:, 1:]).any()
            or not np.isfinite(x).all() or not np.array_equal(engines, np.arange(1, len(x)+1))
            or not math.isfinite(cap) or cap <= 0):
        raise ValueError(f"Invalid test windows or manifest in {directory}")
    if (any(data[key].shape != (len(x),) for key in ("y", "y_raw", "end_cycle"))
            or not np.isfinite(data["y_raw"]).all() or (data["y_raw"] < 0).any()
            or not np.array_equal(data["y"], np.minimum(cap, data["y_raw"]))):
        raise ValueError(f"Invalid RUL labels in {directory}")
    return data, cap, file_hash(path)


def check_config(config, name, args, cap, shape):
    expected = dict(model=name, subset=args.subset, seed=args.seed, cap=cap)
    if name in MODELS:
        expected.update(window=shape[0], features=shape[1])
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"{name} checkpoint metadata differs for {key}: {config.get(key)} vs {value}")


def selected_weights(folder, report, model):
    epoch = report["selected_epoch"]
    if type(epoch) is not int or epoch < 1:
        raise ValueError(f"Invalid selected epoch in {folder}")
    path = folder / f"epoch_{epoch:04d}.pt"
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["epoch"] != epoch or state["config"] != report["config"]:
        raise ValueError(f"Checkpoint does not match the selected epoch and config in {folder}")
    model.load_state_dict(state["model"], strict=True)
    return file_hash(path)


class MeanRUL(nn.Module):
    def __init__(self, cycles):
        super().__init__()
        self.register_buffer("cycles", torch.tensor(cycles, dtype=torch.float32))

    def forward(self, x, mask):
        return self.cycles.expand(x.shape[0])


def load_model(args, name, cap, shape):
    root = args.students_root if name in STUDENTS else args.baselines_root
    if root is None:
        raise ValueError(f"Set --{'students' if name in STUDENTS else 'baselines'}-root for {name}")
    folder = root / args.subset / f"seed_{args.seed}" / name
    report = read_report(folder)
    config = report["config"]
    check_config(config, name, args, cap, shape)
    epoch = report["selected_epoch"]
    fingerprint = None
    if name == "mean":
        path = args.data_root / args.subset / "train.npz"
        if file_hash(path) != config["source_sha256"]["train.npz"]:
            raise ValueError("Use the original prepared train.npz for the saved mean baseline")
        with np.load(path, allow_pickle=False) as train:
            model = MeanRUL(train["y"].mean(dtype=np.float64))
    elif name == "ann_to_snn":
        source_folder = folder.with_name("deepnn")
        source = read_report(source_folder)
        check_config(source["config"], "deepnn", args, cap, shape)
        if (config["source_model"] != "deepnn" or epoch != source["selected_epoch"]
                or config["source_sha256"] != source["config"]["source_sha256"]):
            raise ValueError("Conversion metadata does not match the DeepNN source checkpoint")
        dense = make_model("deepnn", *shape)
        fingerprint = selected_weights(source_folder, source, dense)
        thresholds = torch.tensor(config["calibration_thresholds"], dtype=torch.float32)
        steps = config["conversion_steps"]
        if (thresholds.shape != (len(dense.hidden),) or not torch.isfinite(thresholds).all()
                or not (thresholds > 0).all() or type(steps) is not int or steps < 1):
            raise ValueError("Invalid saved ANN-to-SNN calibration")
        model = ConvertedSNN(dense, thresholds, steps)
    else:
        model = Student(shape[1]) if name in STUDENTS else make_model(name, *shape)
        fingerprint = selected_weights(folder, report, model)
    if sum(p.numel() for p in model.parameters()) != report["parameters"]:
        raise ValueError(f"Model parameter count differs from saved report in {folder}")
    return model.eval(), folder, epoch, fingerprint


def read_text(path):
    try:
        return Path(path).read_text(errors="replace").strip("\x00\n ") or None
    except OSError:
        return None


def command_output(*command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=5, check=True)
        return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def metadata(args, cuda):
    board = read_text("/proc/device-tree/model")
    if board is None:
        board = read_text("/sys/firmware/devicetree/base/model")
    orin_nano = bool(board and "orin nano" in board.lower())
    return dict(utc=datetime.now(timezone.utc).isoformat(), board=board,
                jetson_orin_nano_cuda=orin_nano and cuda, platform=platform.platform(),
                python=platform.python_version(), torch=str(torch.__version__),
                cuda_build=torch.version.cuda, cudnn=torch.backends.cudnn.version() if cuda else None,
                device=args.device, gpu=torch.cuda.get_device_name() if cuda else None,
                jetson_linux=read_text("/etc/nv_tegra_release"),
                jetpack_package=command_output("dpkg-query", "-W", "nvidia-jetpack"),
                power_mode=command_output("nvpmodel", "-q") or args.power_mode_label,
                power_mode_operator_label=args.power_mode_label,
                precision="FP32 (TF32 disabled)", batch_size=1, cpu_threads=args.cpu_threads)


def accuracy(data, predictions, folder):
    results = {}
    for label, target in (("capped", data["y"]), ("raw", data["y_raw"])):
        scores = regression_metrics(target, predictions)
        denominator = np.abs(target.astype(np.float64) - target.mean(dtype=np.float64)).sum()
        scores["RAE_pct"] = float(100 * np.abs(predictions - target).sum()/denominator) if denominator else None
        scores["bias_cycles"] = float(np.mean(predictions - target))
        results[label] = scores
    reference_path = folder / "test_predictions.csv"
    parity = {"status": "reference CSV unavailable"}
    if reference_path.is_file():
        with reference_path.open(newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
        expected = dict(zip(data["engine_id"], zip(data["end_cycle"], data["y"], data["y_raw"], predictions)))
        errors = []
        for row in rows:
            engine = int(row["engine_id"])
            if engine not in expected:
                raise ValueError("Reference test engine IDs differ from test.npz")
            end, capped, raw, actual = expected.pop(engine)
            if (int(row["end_cycle"]) != end or float(row["target_capped_cycles"]) != capped
                    or float(row["target_raw_cycles"]) != raw):
                raise ValueError("Reference test endpoints or labels differ from test.npz")
            errors.append(abs(float(row["prediction_cycles"]) - actual))
        if expected or not np.isfinite(errors).all():
            raise ValueError("Reference predictions are incomplete or non-finite")
        parity = dict(status="compared", max_abs_difference_cycles=float(max(errors)),
                      mean_abs_difference_cycles=float(np.mean(errors)))
    return results, parity


def spike_diagnostics(name, model, x, mask):
    """Count logical events separately from timing. Counts are not dense kernel work."""
    batch, window, _ = x.shape
    if name in STUDENTS:
        events = model.signals(x, mask)[-1]
        total = int(events.sum().item())
        fanout = model.recurrent.out_features
        slots = int(mask.sum().item()) * fanout
        synops = int((events[:, :-1] * mask[:, 1:, None]).sum().item()) * fanout
    elif name == "snn":
        previous, pooled = [], []
        recurrent = model.recurrent.register_forward_pre_hook(lambda layer, inputs: previous.append(inputs[0]))
        readout = model.readout.register_forward_pre_hook(lambda layer, inputs: pooled.append(inputs[0]))
        try:
            model(x, mask)
        finally:
            recurrent.remove()
            readout.remove()
        lengths = mask.sum(dim=1, keepdim=True)
        total = int(torch.round(pooled[0] * lengths).sum().item())
        fanout = model.recurrent.out_features
        slots = int(lengths.sum().item()) * fanout
        synops = int((torch.stack(previous, dim=1) * mask[..., None]).sum().item()) * fanout
    elif name == "ann_to_snn":
        counts, pooled = [], []

        def count_hidden(layer, inputs):
            counts.append((inputs[0].count_nonzero(), layer.out_features))

        hooks = [layer.register_forward_pre_hook(count_hidden) for layer in model.source.hidden[1:]]
        hooks.append(model.source.readout.register_forward_pre_hook(lambda layer, inputs: pooled.append(inputs[0])))
        try:
            model(x, mask)
        finally:
            for hook in hooks:
                hook.remove()
        window = model.steps
        last = torch.round(pooled[0] * window / model.thresholds[-1]).sum()
        total = int((sum(events for events, _ in counts) + last).item())
        slots = batch * window * sum(layer.out_features for layer in model.source.hidden)
        synops = int(sum(events * fanout for events, fanout in counts).item())
    else:
        return None
    return dict(simulation_steps=window, spikes=total, neuron_step_slots=slots,
                spike_rate=total/slots, estimated_hidden_synops_per_inference=synops/batch,
                synops_scope="spike-driven hidden connections; excludes input and readout")


@torch.inference_mode()
def benchmark(args, name, directory):
    cuda = args.device == "cuda"
    if cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Install a matching NVIDIA Jetson PyTorch build or explicitly use --device cpu.")
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    hardware = metadata(args, cuda)
    if not hardware["jetson_orin_nano_cuda"]:
        print("Hardware and device will be recorded as a portability run.", flush=True)
    data, cap, test_hash = load_test(args.data_root / args.subset)
    model, folder, epoch, checkpoint_hash = load_model(args, name, cap, data["X"].shape[1:])
    model.to(device=args.device, dtype=torch.float32)
    for module in model.modules():
        if isinstance(module, nn.RNNBase):
            module.flatten_parameters()
    x = torch.as_tensor(data["X"], device=args.device, dtype=torch.float32)
    mask = torch.as_tensor(data["mask"], device=args.device, dtype=torch.bool)
    order = np.random.default_rng(args.seed).permutation(len(x)).tolist()
    inputs = [(x[index:index+1], mask[index:index+1]) for index in order]

    def predict(features, valid):
        output = model(features, valid)
        return output if name == "mean" else output * cap

    def infer(index):
        predict(*inputs[index % len(inputs)])
        if cuda:
            torch.cuda.synchronize()

    for i in range(args.warmup):
        infer(i)
    memory = {}
    if cuda:
        torch.cuda.reset_peak_memory_stats()
        memory.update(cuda_allocated_after_warmup_bytes=torch.cuda.memory_allocated(),
                      cuda_reserved_after_warmup_bytes=torch.cuda.memory_reserved())
    durations = []
    for i in range(args.runs):
        start = perf_counter_ns()
        infer(i)
        durations.append((perf_counter_ns() - start) / 1e6)
    if cuda:
        memory.update(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    if platform.system() == "Linux":
        import resource
        memory["linux_process_peak_rss_through_latency_bytes"] = 1024 * resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    rail = find_rail(args.power_rail) if args.power_seconds else None
    if rail:
        power, samples = measure(rail, infer, args.power_seconds)
        write_csv(directory / "power_samples.csv", ("seconds_from_run_start", "watts"), samples)
    else:
        status = "rail unavailable or unreadable" if args.power_seconds else "disabled"
        power = dict(status=status, rail=args.power_rail)

    predictions = np.asarray([predict(features, valid).item()
                              for features, valid in zip(x.split(1), mask.split(1))])
    scores, parity = accuracy(data, predictions, folder)
    # Bound diagnostic workspace without changing the test cases.
    diagnostics = [spike_diagnostics(name, model, features, valid)
                   for features, valid in zip(x.split(32), mask.split(32))]
    spike = diagnostics[0]
    if spike:
        spikes = sum(row["spikes"] for row in diagnostics)
        slots = sum(row["neuron_step_slots"] for row in diagnostics)
        synops = sum(row["estimated_hidden_synops_per_inference"] * len(features)
                     for row, features in zip(diagnostics, x.split(32)))
        spike.update(spikes=spikes, neuron_step_slots=slots, spike_rate=spikes/slots,
                     estimated_hidden_synops_per_inference=synops/len(x))
    median, p95, p99 = (float(v) for v in np.percentile(durations, (50, 95, 99)))
    latency = dict(median_ms=median, p95_ms=p95, p99_ms=p99, mean_ms=float(np.mean(durations)),
                   inferences_per_second=1000/float(np.mean(durations)), runs=args.runs, warmup=args.warmup)
    write_csv(directory / "latency_samples.csv", ("run", "engine_id", "latency_ms"),
              ((i + 1, data["engine_id"][order[i % len(order)]], value)
               for i, value in enumerate(durations)))
    write_csv(directory / "test_predictions.csv",
              ("engine_id", "end_cycle", "target_capped_cycles", "target_raw_cycles", "prediction_cycles"),
              zip(data["engine_id"], data["end_cycle"], data["y"], data["y_raw"], predictions))
    report = dict(model=name, subset=args.subset, seed=args.seed, selected_epoch=epoch,
                  parameters=sum(p.numel() for p in model.parameters()), test_sha256=test_hash,
                  checkpoint_sha256=checkpoint_hash, input_shape=list(x.shape[1:]), rul_cap_cycles=cap,
                  hardware=hardware, latency=latency, memory=memory, power=power, spike=spike,
                  test_metrics_cycles=scores, saved_prediction_comparison=parity)
    write_json(directory / "report.json", report)
    print(f"{name}: median {median:.3f} ms, p95 {p95:.3f} ms, power {power['status']}", flush=True)


def summarize(directory, models):
    rows = []
    for name in models:
        path = directory / name / "report.json"
        if not path.is_file():
            rows.append(dict(model=name, status="failed"))
            continue
        r = read_json(path)
        row = {key: r[key] for key in ("model", "subset", "seed", "parameters", "selected_epoch")}
        row.update(status="ok", device=r["hardware"]["device"],
                   jetson_orin_nano_cuda=r["hardware"]["jetson_orin_nano_cuda"],
                   power_mode=r["hardware"]["power_mode"], **r["latency"], **r["memory"])
        row.update(power_status=r["power"]["status"], power_rail=r["power"]["rail"],
                   mean_w=r["power"].get("mean_w"), mj_per_inference=r["power"].get("mj_per_inference"))
        if r["spike"]:
            for key in ("simulation_steps", "spike_rate", "estimated_hidden_synops_per_inference"):
                row[key] = r["spike"][key]
        for target, metrics in r["test_metrics_cycles"].items():
            row.update({f"{target}_{key}": value for key, value in metrics.items()})
        rows.append(row)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (directory / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--baselines-root", type=Path)
    parser.add_argument("--students-root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("jetson_results"))
    parser.add_argument("--subset", choices=SUBSETS, default="FD001")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models", nargs="+", choices=(*MODELS, *STUDENTS, "all"), default=["all"])
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--runs", type=int, default=1000)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--power-seconds", type=float, default=10)
    parser.add_argument("--power-rail", default="VDD_IN")
    parser.add_argument("--power-mode-label")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (args.warmup < 1 or args.runs < 1000 or args.cpu_threads < 1
            or not math.isfinite(args.power_seconds)
            or (args.power_seconds != 0 and args.power_seconds < 1)):
        parser.error("Use at least 1 warm-up, 1000 measured runs and 1 CPU thread. Power duration is zero or at least 1 second.")
    if args.worker:
        benchmark(args, args.models[0], args.output)
        return
    models = (*MODELS, *STUDENTS) if "all" in args.models else tuple(dict.fromkeys(args.models))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    directory = args.output / args.subset / f"seed_{args.seed}" / stamp
    directory.mkdir(parents=True)
    print(f"Results directory: {directory.resolve()}", flush=True)
    failed = []
    for model in models:
        output = directory / model
        output.mkdir()
        print(f"Benchmarking {args.subset} {model}", flush=True)
        result = subprocess.run([sys.executable, "-m", "deployment.jetson.benchmark", *sys.argv[1:],
                                 "--models", model, "--output", str(output), "--worker"])
        if result.returncode:
            failed.append(model)
    summarize(directory, models)
    if failed:
        sys.exit(f"Models with errors: {', '.join(failed)}. See the per-model output above.")


if __name__ == "__main__":
    main()

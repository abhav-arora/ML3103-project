"""Compare a frozen teacher and student under one missing sensor at a time."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from spikedistill.evaluation.metrics import regression_metrics
from spikedistill.models.distillation import Student
from spikedistill.training.baselines import batches, predict_cycles, read_split, write_json
from spikedistill.training.distill import prediction_parity, teacher_for
from spikedistill.training.pilot import digest


def evaluate(args, subset):
    prepared = args.data_root / subset
    manifest = json.loads((prepared / "split_manifest.json").read_text())
    prep = json.loads((prepared / "preprocessing.json").read_text())
    cap = float(manifest["rul_cap"])
    device = torch.device(args.device)
    hashes = [digest(prepared / f"{split}.npz") for split in ("train", "validation")]
    validation = read_split(prepared, "validation")
    teacher, fingerprint = teacher_for(args.baselines, subset, args.seed, *hashes, cap, device,
                                       verify_with=validation if args.verify_prepared_predictions else None)
    folder = args.students / subset / f"seed_{args.seed}" / "spikedistill"
    report = json.loads((folder / "test_metrics.json").read_text())
    path = folder / f"epoch_{report['selected_epoch']:04d}.pt"
    state = torch.load(path, map_location="cpu", weights_only=True)
    config = state["config"]
    if (config != report["config"] or state["epoch"] != report["selected_epoch"]
            or (config["subset"], config["seed"], config["cap"]) != (subset, args.seed, cap)
            or config["teacher_sha256"] != fingerprint):
        raise ValueError(f"Student checkpoint or teacher mismatch in {folder}")
    if ([config["train_sha256"], config["validation_sha256"]] != hashes
            and not args.verify_prepared_predictions):
        raise ValueError("Prepared data hashes differ; use the explicit prediction-parity option")
    data = read_split(prepared, "test")
    if len(np.unique(data["engine_id"])) != len(data["engine_id"]):
        raise ValueError("Expected one test window per engine")
    student = Student(data["X"].shape[-1]).to(device).eval()
    student.load_state_dict(state["model"])
    models = dict(teacher=teacher, student=student)
    clean = {name: predict_cycles(model, batches(data, cap, 128), device, cap) for name, model in models.items()}
    difference = prediction_parity(folder, "test", data, clean["student"])
    rows = []
    for sensor in ("clean", *prep["sensors"]):
        current = data
        if sensor != "clean":
            x = data["X"].copy()
            x[:, :, prep["feature_names"].index(sensor)] = 0
            current = dict(data, X=x)
        for name, model in models.items():
            pred = clean[name] if sensor == "clean" else predict_cycles(model, batches(current, cap, 128), device, cap)
            for target, truth in (("raw", data["y_raw"]), ("capped", data["y"])):
                metrics = regression_metrics(truth, pred)
                reference = regression_metrics(truth, clean[name])
                rows.append(dict(subset=subset, seed=args.seed, target=target, model=name, missing_sensor=sensor,
                                 **metrics, delta_RMSE=metrics["RMSE"] - reference["RMSE"],
                                 delta_cmapss_score=metrics["cmapss_score"] - reference["cmapss_score"]))
    output = args.output / subset / f"seed_{args.seed}"
    write_json(output / "provenance.json", dict(subset=subset, seed=args.seed,
        student_checkpoint_sha256=digest(path), teacher_checkpoint_sha256=fingerprint,
        test_sha256=digest(prepared / "test.npz"), student_prediction_parity_max_cycles=difference,
        protocol="one sensor replaced by its standardized training mean for the entire observed window",
        observed_inputs="operating settings and validity masks preserved", tuning="none"))
    with (output / "sensor_outage.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot(rows, output, subset)
    print(f"{subset}: evaluated {len(prep['sensors'])} sensor outages for teacher and student", flush=True)


def plot(rows, output, subset):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sensors = list(dict.fromkeys(r["missing_sensor"] for r in rows if r["missing_sensor"] != "clean"))
    fig, ax = plt.subplots(figsize=(10, 4.8), layout="constrained")
    for name, marker, color in (("teacher", "o", "#176b87"), ("student", "s", "#ad5542")):
        values = [next(r["delta_RMSE"] for r in rows if r["target"] == "raw" and r["model"] == name
                       and r["missing_sensor"] == sensor) for sensor in sensors]
        ax.plot(range(len(sensors)), values, marker=marker, label=name, color=color)
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.set_xticks(range(len(sensors)), [s.replace("sensor_", "S") for s in sensors], rotation=45)
    ax.set(title=f"{subset} · one sensor unavailable throughout the window",
           ylabel="Increase in raw test RMSE (cycles)", xlabel="Sensor replaced by training mean")
    ax.legend()
    fig.savefig(output / "sensor_outage.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-root", "baselines", "students", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--subsets", nargs="+", choices=("FD001", "FD002", "FD003", "FD004"),
                        default=["FD001", "FD002", "FD003", "FD004"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--verify-prepared-predictions", action="store_true")
    args = parser.parse_args()
    if args.output.resolve() in (args.data_root.resolve(), args.baselines.resolve(), args.students.resolve()):
        parser.error("Use a separate results folder")
    torch.set_num_threads(1)
    for subset in args.subsets:
        evaluate(args, subset)

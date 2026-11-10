"""Summarize paired one-loss removals on the official C-MAPSS test engines."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from spikedistill.evaluation.metrics import regression_metrics
from spikedistill.training.distill import ABLATIONS, VARIANTS


MODELS = ("spikedistill", *ABLATIONS)
METRICS = ("MAE", "MSE", "RMSE", "RAE_pct", "R2", "cmapss_score", "bias_cycles")
SUBSETS = ("FD001", "FD002", "FD003", "FD004")


def load_runs(roots, subset, seed):
    folders = [root / subset / f"seed_{seed}" for root in roots]
    reports = {}
    endpoints = None
    for name in MODELS:
        paths = [folder / name / "test_metrics.json" for folder in folders]
        matches = [path for path in paths if path.is_file()]
        if not matches:
            raise FileNotFoundError(paths[0])
        if len(matches) > 1:
            raise ValueError(f"Ambiguous artifacts for {subset} seed {seed} {name}")
        path, = matches
        report = json.loads(path.read_text(encoding="utf-8"))
        config = report["config"]
        if ((config["subset"], config["seed"], config["model"]) != (subset, seed, name)
                or config["loss_weights"] != list((VARIANTS | ABLATIONS)[name])
                or report["split"] != "test" or report["prediction_units"] != "cycles"):
            raise ValueError(f"Invalid test report {path}")
        with path.with_name("test_predictions.csv").open(newline="", encoding="utf-8") as stream:
            columns = ("engine_id", "end_cycle", "target_capped_cycles", "target_raw_cycles", "prediction_cycles")
            predictions = sorted(tuple(float(row[key]) for key in columns) for row in csv.DictReader(stream))
        current = [row[:-1] for row in predictions]
        if len(current) != report["raw"]["n"] or len({row[0] for row in current}) != len(current):
            raise ValueError(f"Invalid test endpoints in {path}")
        for target, index in (("capped", 2), ("raw", 3)):
            true = np.array([row[index] for row in predictions])
            pred = np.array([row[-1] for row in predictions])
            calculated = regression_metrics(true, pred)
            for metric, value in calculated.items():
                saved = report[target][metric]
                if value != saved and (value is None or saved is None or not np.isclose(value, saved, rtol=1e-9)):
                    raise ValueError(f"Metric mismatch for {target} {metric} in {path}")
            denominator = np.abs(true - true.mean()).sum()
            report[target]["RAE_pct"] = float(100 * np.abs(pred - true).sum() / denominator) if denominator else None
            report[target]["bias_cycles"] = float(np.mean(pred - true))
        if endpoints is not None and current != endpoints:
            raise ValueError(f"Different test cases in {path}")
        endpoints = current
        history = json.loads(path.with_name("history.json").read_text(encoding="utf-8"))
        selected = min(history, key=lambda row: row["validation"]["RMSE"])
        if len(history) != config["epochs"] or selected["epoch"] != report["selected_epoch"]:
            raise ValueError(f"Incomplete training or selection mismatch in {path}")
        reports[name] = report, selected["spike_rate"]
    settings = [{k: v for k, v in r["config"].items() if k not in {"model", "loss_weights"}}
                for r, _ in reports.values()]
    if settings.count(settings[0]) != len(settings) or len({r["parameters"] for r, _ in reports.values()}) != 1:
        raise ValueError(f"Unmatched settings for {subset} seed {seed}")
    return reports


def save_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(roots, output, subsets, seeds):
    runs, means = [], []
    keys = (*METRICS, "delta_RMSE_vs_full", "training_spike_rate_at_selected_epoch")
    for subset in subsets:
        group = []
        for seed in seeds:
            reports = load_runs(roots, subset, seed)
            for target in ("raw", "capped"):
                full_rmse = reports["spikedistill"][0][target]["RMSE"]
                for name, (report, spike_rate) in reports.items():
                    metrics = {key: report[target][key] for key in METRICS}
                    group.append(dict(subset=subset, seed=seed, target=target, model=name,
                                      selected_epoch=report["selected_epoch"], engines=report[target]["n"],
                                      **metrics, delta_RMSE_vs_full=metrics["RMSE"] - full_rmse,
                                      training_spike_rate_at_selected_epoch=spike_rate))
        runs.extend(group)
        for target in ("raw", "capped"):
            for name in MODELS:
                rows = [row for row in group if (row["target"], row["model"]) == (target, name)]
                row = dict(subset=subset, target=target, model=name, seeds=len(rows))
                for key in keys:
                    values = [r[key] for r in rows]
                    row[f"{key}_mean"] = float(np.mean(values)) if None not in values else None
                    row[f"{key}_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 and None not in values else None
                means.append(row)
    output.mkdir(parents=True, exist_ok=True)
    save_csv(output / "ablation_per_seed.csv", runs)
    save_csv(output / "ablation_mean_sd.csv", means)
    plot(means, output, subsets)


def plot(means, output, subsets):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 4.6), layout="constrained")
    positions = np.arange(len(subsets))
    for i, (name, label, color) in enumerate(zip(ABLATIONS,
            ("Without hidden matching", "Without attention matching"), ("#196e88", "#ac573b"))):
        rows = [r for r in means if r["model"] == name and r["target"] == "raw"]
        ax.bar(positions + (i - 0.5) * 0.32, [r["delta_RMSE_vs_full_mean"] for r in rows],
               yerr=[r["delta_RMSE_vs_full_sd"] or 0 for r in rows], width=0.32, capsize=4,
               label=label, color=color)
    ax.set_xticks(positions, subsets)
    ax.axhline(0, color="#333333", linewidth=1)
    ax.set(title="Effect of removing one distillation loss", ylabel="Raw test RMSE change (cycles)")
    ax.legend()
    fig.savefig(output / "ablation_rmse_deltas.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subsets", nargs="+", choices=SUBSETS, default=list(SUBSETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = parser.parse_args()
    summarize(args.roots, args.output, args.subsets, args.seeds)

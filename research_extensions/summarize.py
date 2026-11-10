"""Render research-extension tables and plots from completed experiment artifacts."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def write_table(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(root):
    ensembles = []
    for path in sorted((root / "seed_ensemble").glob("FD*/test_summary.json")):
        for report in json.loads(path.read_text()):
            ensembles.append(dict(subset=report["config"]["subset"], model=report["config"]["model"],
                                  parameters=report["parameters"], **report["raw"]))
    if ensembles:
        write_table(root / "ensemble_summary.csv", ensembles)
        fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
        subsets = list(dict.fromkeys(r["subset"] for r in ensembles))
        names = ("validation_best_single", "equal_ensemble", "validation_weighted_ensemble")
        for i, name in enumerate(names):
            values = [next(r["RMSE"] for r in ensembles if r["subset"] == subset and r["model"] == name)
                      for subset in subsets]
            ax.bar(np.arange(len(subsets)) + (i-1)*0.24, values, width=0.24, label=name.replace("_", " "))
        ax.set_xticks(range(len(subsets)), subsets)
        ax.set(ylabel="Raw test RMSE (cycles)", title="Three frozen student seeds averaged at inference")
        ax.legend(fontsize=8)
        fig.savefig(root / "ensemble_comparison.png", dpi=170)
        plt.close(fig)
    capacity = []
    for path in sorted((root / "student_capacity").glob("FD*/seed_*/*/test_metrics.json")):
        report = json.loads(path.read_text())
        config = report["config"]
        history = json.loads(path.with_name("history.json").read_text())
        selected = next(row for row in history if row["epoch"] == report["selected_epoch"])
        capacity.append(dict(subset=config["subset"], seed=config["seed"], width=config["width"],
                             selected_epoch=report["selected_epoch"], parameters=report["parameters"],
                             validation_RMSE=selected["validation"]["RMSE"],
                             train_spike_rate_at_selected_epoch=selected["spike_rate"], **report["raw"]))
    if capacity:
        capacity.sort(key=lambda r: (r["subset"], r["seed"], r["width"]))
        write_table(root / "capacity_summary.csv", capacity)
        fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
        groups = sorted({(r["subset"], r["seed"]) for r in capacity})
        for subset, seed in groups:
            rows = [r for r in capacity if (r["subset"], r["seed"]) == (subset, seed)]
            ax.plot([r["parameters"] for r in rows], [r["RMSE"] for r in rows], "o-", label=f"{subset}, seed {seed}")
            for index, row in enumerate(rows):
                ax.annotate(f"W={row['width']}", (row["parameters"], row["RMSE"]),
                            xytext=(-5 if index == len(rows)-1 else 4, 5),
                            ha="right" if index == len(rows)-1 else "left",
                            textcoords="offset points", fontsize=8)
        ax.set(xlabel="Inference parameters", ylabel="Raw test RMSE (cycles)",
               title="Student capacity with the same teacher and full loss")
        ax.legend(fontsize=8)
        fig.savefig(root / "student_capacity.png", dpi=170)
        plt.close(fig)
    print(f"Summarized {len(ensembles)} ensemble scores and {len(capacity)} capacity runs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(__file__).resolve().parent / "results")
    summarize(parser.parse_args().results)

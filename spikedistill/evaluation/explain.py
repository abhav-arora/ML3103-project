"""Explain the frozen student on each subset using one predeclared seed."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from spikedistill.training.distill import Student, teacher_for
from spikedistill.training.baselines import batches, predict_cycles, read_split
from spikedistill.training.pilot import digest


def integrated_gradients(model, x, mask, cap, steps=64):
    """Approximate path attribution using the training surrogate derivative."""
    alpha = (torch.arange(steps, device=x.device) + 0.5) / steps
    path = (alpha[:, None, None] * x).requires_grad_()
    prediction = model(path, mask.expand(steps, -1)) * cap
    gradient, = torch.autograd.grad(prediction.sum(), path)
    return (x * gradient.mean(0)).squeeze(0).detach().cpu().numpy()


def sensor_permutation(model, data, sensors, cap, device, seed):
    y = data["y_raw"]
    original = predict_cycles(model, batches(data, cap, 128), device, cap)
    rmse = lambda pred: float(np.sqrt(np.mean((pred - y) ** 2)))
    reference = rmse(original)
    lengths = data["mask"].sum(axis=1)
    rng = np.random.default_rng(seed)
    rows = []
    for index, name in enumerate(sensors):
        differences = []
        for _ in range(5):
            x = data["X"].copy()
            for length in np.unique(lengths):
                group = np.flatnonzero(lengths == length)
                x[group, :, index] = data["X"][rng.permutation(group), :, index]
            prediction = predict_cycles(model, batches(dict(data, X=x), cap, 128), device, cap)
            differences.append(rmse(prediction) - reference)
        rows.append((name, np.mean(differences), np.std(differences, ddof=1)))
    return rows


def explain(args, subset):
    device = torch.device(args.device)
    prepared = args.data_root / subset
    folder = args.students / subset / f"seed_{args.seed}" / "spikedistill"
    report = json.loads((folder / "test_metrics.json").read_text())
    state = torch.load(folder / f"epoch_{report['selected_epoch']:04d}.pt",
                       map_location="cpu", weights_only=True)
    if state["config"] != report["config"] or state["epoch"] != report["selected_epoch"]:
        raise ValueError(f"Selected student mismatch: {folder}")
    cap = state["config"]["cap"]
    data = read_split(prepared, "test")
    features = json.loads((prepared / "preprocessing.json").read_text())
    teacher, fingerprint = teacher_for(args.baselines, subset, args.seed, digest(prepared / "train.npz"),
                                       digest(prepared / "validation.npz"), cap, device)
    if fingerprint != state["config"]["teacher_sha256"]:
        raise ValueError(f"Teacher fingerprint mismatch: {folder}")
    model = Student(data["X"].shape[-1]).to(device).eval()
    model.load_state_dict(state["model"])
    pred = predict_cycles(model, batches(data, cap, 128), device, cap)
    errors = np.abs(pred - data["y_raw"])
    target = args.students / "explanations" / subset
    target.mkdir(parents=True, exist_ok=True)

    rows = sensor_permutation(model, data, features["sensors"], cap, device, args.seed)
    with (target / "sensor_permutation.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sensor", "rmse_change_cycles_mean", "rmse_change_cycles_sample_sd"])
        writer.writerows(rows)
    rows.sort(key=lambda row: row[1])
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    ax.barh([r[0] for r in rows], [r[1] for r in rows], xerr=[r[2] for r in rows])
    ax.set(title=f"{subset} held-out sensor permutation", xlabel="Change in raw test RMSE (cycles)")
    fig.savefig(target / "sensor_permutation.png", dpi=160)
    plt.close(fig)

    for label, index in (("lowest_error", errors.argmin()), ("highest_error", errors.argmax())):
        x = torch.from_numpy(data["X"][index:index+1]).to(device)
        mask = torch.from_numpy(data["mask"][index:index+1]).to(device)
        ig = integrated_gradients(model, x, mask, cap)
        with torch.no_grad():
            _, _, teacher_attention = teacher.signals(x, mask)
            _, _, student_attention, spikes = model.signals(x, mask)
            # Exact contribution to the linear readout with attention held fixed.
            contribution = spikes[0] * student_attention[0, :, None] * model.readout.weight[0] * cap
        valid = mask[0].cpu().numpy()
        cycles = np.arange(int(data["end_cycle"][index]) - valid.sum() + 1, int(data["end_cycle"][index]) + 1)
        fig, axes = plt.subplots(4, 1, figsize=(10, 12), constrained_layout=True)
        fig.suptitle(f"{subset}, engine {data['engine_id'][index]} | prediction {pred[index]:.1f} vs true {data['y_raw'][index]:.1f} cycles")
        fig.colorbar(axes[0].imshow(ig[valid].T, aspect="auto", cmap="coolwarm"), ax=axes[0], label="Cycles")
        axes[0].set(title="Surrogate Integrated Gradients", ylabel="Input feature", xlabel="Observed window position")
        axes[1].plot(cycles, teacher_attention[0].cpu().numpy()[valid], label="Teacher")
        axes[1].plot(cycles, student_attention[0].cpu().numpy()[valid], label="Student")
        axes[1].set(title="Attention", xlabel="Engine cycle", ylabel="Weight")
        axes[1].legend()
        spike_map = spikes[0].cpu().numpy()[valid].T
        axes[2].imshow(spike_map, aspect="auto", cmap="Greys", vmin=0, vmax=1)
        axes[2].set(title="Spike raster", ylabel="Neuron", xlabel="Observed window position")
        axes[3].bar(cycles, contribution.sum(1).cpu().numpy()[valid])
        axes[3].set(title="Signed spike contribution with fixed attention", xlabel="Engine cycle", ylabel="Cycles")
        fig.savefig(target / f"{label}.png", dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for path in ("data-root", "baselines", "students"):
        parser.add_argument(f"--{path}", type=Path, required=True)
    parser.add_argument("--subsets", nargs="+", choices=["FD001", "FD002", "FD003", "FD004"],
                        default=["FD001", "FD002", "FD003", "FD004"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    for subset in args.subsets:
        explain(args, subset)

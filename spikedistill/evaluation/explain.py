"""Explain the frozen student on each subset using one predeclared seed."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from spikedistill.training.distill import Student, prediction_parity, teacher_for
from spikedistill.training.baselines import batches, predict_cycles, read_split
from spikedistill.training.pilot import digest


def integrated_gradients(model, x, mask, cap, steps=64):
    """Approximate path attribution using the training surrogate derivative."""
    alpha = (torch.arange(steps, device=x.device) + 0.5) / steps
    path = (alpha[:, None, None] * x).requires_grad_()
    prediction = model(path, mask.expand(steps, -1)) * cap
    gradient, = torch.autograd.grad(prediction.sum(), path)
    return (x * gradient.mean(0)).squeeze(0).detach().cpu().numpy()


@torch.inference_mode()
def temporal_shap(model, x, mask, cap, sensor_indices, seed, permutations=32, block_size=5):
    """Estimate Shapley values for sensor-by-time blocks with a standardized zero reference."""
    observed = mask[0].nonzero(as_tuple=True)[0].cpu().numpy()
    blocks = [observed[start:start + block_size] for start in range(0, len(observed), block_size)]
    groups = [(sensor, block) for sensor in sensor_indices for block in blocks]
    original = x[0].cpu().numpy()
    reference = original.copy()
    reference[np.ix_(observed, sensor_indices)] = 0
    rng = np.random.default_rng(seed)
    values = np.zeros(len(groups), dtype=np.float64)
    endpoints = None
    for permutation in range(permutations):
        # Pair each ordering with its reverse to balance early and late insertions.
        order = rng.permutation(len(groups)) if permutation % 2 == 0 else order[::-1]
        sequence = np.repeat(reference[None], len(groups) + 1, axis=0)
        for position, index in enumerate(order, start=1):
            sensor, block = groups[index]
            sequence[position:, block, sensor] = original[block, sensor]
        tensor = torch.as_tensor(sequence, device=x.device)
        output = (model(tensor, mask.expand(len(sequence), -1)) * cap).cpu().numpy().astype(np.float64)
        values[order] += np.diff(output)
        endpoints = output[[0, -1]]
    values /= permutations
    return values.reshape(len(sensor_indices), len(blocks)), blocks, endpoints


def signed_heatmap(ax, values, labels, title, xlabels, xlabel):
    bound = max(float(np.max(np.abs(values))), 1e-8)
    plot = ax.imshow(values, aspect="auto", cmap="coolwarm", vmin=-bound, vmax=bound)
    ax.set(title=title, xlabel=xlabel)
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    ax.set_xticks(range(len(xlabels)), xlabels, fontsize=8)
    ax.figure.colorbar(plot, ax=ax, label="Cycles")


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
    config = state["config"]
    train_hash, val_hash = (digest(prepared / f"{name}.npz") for name in ("train", "validation"))
    if (config != report["config"] or state["epoch"] != report["selected_epoch"]
            or (config["model"], config["subset"], config["seed"]) != ("spikedistill", subset, args.seed)):
        raise ValueError(f"Selected student mismatch: {folder}")
    data_hash_matches = (config["train_sha256"], config["validation_sha256"]) == (train_hash, val_hash)
    if not data_hash_matches and not args.verify_prepared_predictions:
        raise ValueError(f"Student prepared data fingerprint mismatch: {folder}")
    cap = config["cap"]
    data = read_split(prepared, "test")
    if len(np.unique(data["engine_id"])) != len(data["engine_id"]):
        raise ValueError("Official test requires one final window per engine")
    features = json.loads((prepared / "preprocessing.json").read_text())
    validation = read_split(prepared, "validation") if args.verify_prepared_predictions else None
    teacher, fingerprint = teacher_for(
        args.baselines, subset, args.seed, train_hash, val_hash, cap, device, verify_with=validation)
    if fingerprint != config["teacher_sha256"]:
        raise ValueError(f"Teacher fingerprint mismatch: {folder}")
    model = Student(data["X"].shape[-1]).to(device).eval()
    model.load_state_dict(state["model"])
    pred = predict_cycles(model, batches(data, cap, 128), device, cap)
    parity = None
    if args.verify_prepared_predictions:
        parity = prediction_parity(folder, "test", data, pred)
    errors = np.abs(pred - data["y_raw"])
    target = (args.output or args.students / "explanations") / subset
    target.mkdir(parents=True, exist_ok=True)

    rows = sensor_permutation(model, data, features["sensors"], cap, device, args.seed)
    with (target / "sensor_permutation.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sensor", "rmse_change_cycles_mean", "rmse_change_cycles_sample_sd"])
        writer.writerows(rows)
    rows.sort(key=lambda row: row[1])
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    ax.barh([r[0] for r in rows], [r[1] for r in rows], xerr=[r[2] for r in rows],
            color=["#176b87" if row[1] >= 0 else "#ad5542" for row in rows])
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set(title=f"{subset} held-out sensor permutation", xlabel="Change in raw test RMSE (cycles)")
    fig.savefig(target / "sensor_permutation.png", dpi=160)
    plt.close(fig)

    cases = []
    for label, index in (("lowest_error", errors.argmin()), ("highest_error", errors.argmax())):
        x = torch.from_numpy(data["X"][index:index+1]).to(device)
        mask = torch.from_numpy(data["mask"][index:index+1]).to(device)
        ig = integrated_gradients(model, x, mask, cap)
        with torch.no_grad():
            _, _, teacher_attention = teacher.signals(x, mask)
            _, _, student_attention, spikes = model.signals(x, mask)
            # Exact contribution to the linear readout with attention held fixed.
            contribution = spikes[0] * student_attention[0, :, None] * model.readout.weight[0] * cap
            bias = float(model.readout.bias.item() * cap)
            ig_baseline = float(model(torch.zeros_like(x), mask).item() * cap)
        valid = mask[0].cpu().numpy()
        cycles = np.arange(int(data["end_cycle"][index]) - valid.sum() + 1, int(data["end_cycle"][index]) + 1)
        fig, axes = plt.subplots(4, 1, figsize=(10, 12), constrained_layout=True)
        fig.suptitle(f"{subset}, engine {data['engine_id'][index]} | prediction {pred[index]:.1f} vs true {data['y_raw'][index]:.1f} cycles")
        ticks = np.arange(0, len(cycles), max(1, len(cycles) // 6))
        signed_heatmap(axes[0], ig[valid].T, features["feature_names"], "Surrogate Integrated Gradients",
                       cycles, "Engine cycle")
        axes[0].set_xticks(ticks, cycles[ticks])
        axes[1].plot(cycles, teacher_attention[0].cpu().numpy()[valid], label="Teacher")
        axes[1].plot(cycles, student_attention[0].cpu().numpy()[valid], label="Student")
        axes[1].set(title="Attention", xlabel="Engine cycle", ylabel="Weight")
        axes[1].legend()
        spike_map = spikes[0].cpu().numpy()[valid].T
        axes[2].imshow(spike_map, aspect="auto", cmap="Greys", vmin=0, vmax=1)
        axes[2].set(title="Spike raster", ylabel="Neuron", xlabel="Engine cycle")
        axes[2].set_xticks(ticks, cycles[ticks])
        axes[3].bar(cycles, contribution.sum(1).cpu().numpy()[valid])
        axes[3].set(title="Signed spike contribution with fixed attention", xlabel="Engine cycle", ylabel="Cycles")
        fig.savefig(target / f"{label}.png", dpi=160)
        plt.close(fig)
        sensors = features["sensors"]
        values, blocks, endpoints = temporal_shap(
            model, x, mask, cap, [features["feature_names"].index(s) for s in sensors], args.seed)
        first_cycle = int(data["end_cycle"][index]) - x.shape[1] + 1
        spans = [(first_cycle + int(block[0]), first_cycle + int(block[-1])) for block in blocks]
        fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
        signed_heatmap(ax, values, sensors, f"{subset} · engine {data['engine_id'][index]} · temporal SHAP",
                       [f"{first}–{last}" for first, last in spans], "Engine cycle block")
        fig.savefig(target / f"{label}_temporal_shap.png", dpi=160)
        plt.close(fig)
        with (target / f"{label}_temporal_shap.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["sensor", "first_cycle", "last_cycle", "shap_cycles"])
            writer.writerows((sensor, *span, value)
                             for sensor, row in zip(sensors, values) for span, value in zip(spans, row))
        cases.append(dict(case=label, engine_id=int(data["engine_id"][index]), true_raw_cycles=float(data["y_raw"][index]),
                          predicted_cycles=float(pred[index]), absolute_error_cycles=float(errors[index]),
                          shap_reference_cycles=float(endpoints[0]), shap_prediction_cycles=float(endpoints[1]),
                          shap_efficiency_residual_cycles=float(endpoints[1] - endpoints[0] - values.sum()),
                          ig_reference_cycles=ig_baseline,
                          surrogate_ig_residual_cycles=float(pred[index] - ig_baseline - ig[valid].sum()),
                          readout_reconstruction_residual_cycles=float(pred[index] - bias - contribution.sum().item())))
    (target / "cases.json").write_text(json.dumps(dict(subset=subset, seed=args.seed,
        selected_epoch=report["selected_epoch"], teacher_sha256=fingerprint,
        student_data_hash_matches=data_hash_matches, student_test_prediction_max_difference_cycles=parity,
        test_sha256=digest(prepared / "test.npz"), cases=cases), indent=2), encoding="utf-8")
    print(f"{subset}: saved explanations for engines {[case['engine_id'] for case in cases]}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for path in ("data-root", "baselines", "students"):
        parser.add_argument(f"--{path}", type=Path, required=True)
    parser.add_argument("--subsets", nargs="+", choices=["FD001", "FD002", "FD003", "FD004"],
                        default=["FD001", "FD002", "FD003", "FD004"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-prepared-predictions", action="store_true",
                        help="Verify saved teacher validation and student test predictions for a re-created snapshot.")
    args = parser.parse_args()
    torch.set_num_threads(1)
    for subset in args.subsets:
        explain(args, subset)

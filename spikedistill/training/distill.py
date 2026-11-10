"""Train matched spiking students and compare their frozen checkpoints."""
import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from spikedistill.models import distillation
from spikedistill.models.distillation import Student, Teacher
from spikedistill.evaluation.metrics import regression_metrics
from spikedistill.training.baselines import aggregate, batches, predict_cycles, read_split, save_result, write_json
from spikedistill.training.pilot import digest, load_latest, restore_rng, rng_state, save_checkpoint

VARIANTS = {
    "learnable_snn": (1.0, 0.0, 0.0, 0.0, 0.0),
    "spike_output_kd": (0.5, 0.5, 0.0, 0.0, 0.0),
    "spike_temporal_kd": (0.5, 0.5, 0.1, 0.1, 0.0),
    "spikedistill": (0.5, 0.5, 0.1, 0.1, 0.01),
}
ABLATIONS = {
    "spike_no_hidden": (0.5, 0.5, 0.0, 0.1, 0.01),
    "spike_no_attention": (0.5, 0.5, 0.1, 0.0, 0.01),
}


def prediction_parity(folder, split, data, prediction, atol=0.001):
    """Require exact saved endpoints and labels, with a cycle-scale prediction tolerance."""
    with (folder / f"{split}_predictions.csv").open(newline="", encoding="utf-8") as stream:
        rows = sorted(csv.DictReader(stream), key=lambda row: int(row["engine_id"]))
    order = np.argsort(data["engine_id"])
    for column, key in (("engine_id", "engine_id"), ("end_cycle", "end_cycle"),
                        ("target_capped_cycles", "y"), ("target_raw_cycles", "y_raw")):
        if not np.array_equal([float(row[column]) for row in rows], data[key][order]):
            raise ValueError(f"Different {split} {column} in {folder}")
    saved = np.array([float(row["prediction_cycles"]) for row in rows])
    difference = float(np.abs(saved - prediction[order]).max())
    if not np.isfinite(difference) or difference > atol:
        raise ValueError(f"{split} prediction parity failed in {folder}: {difference:.6g} cycles")
    return difference


def teacher_for(root, subset, seed, train_hash, validation_hash, cap, device, verify_with=None):
    folder = root / subset / f"seed_{seed}" / "attention_bilstm"
    metrics = json.loads((folder / "validation_metrics.json").read_text())
    path = folder / f"epoch_{metrics['selected_epoch']:04d}.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    sources = config["source_sha256"]
    if (config != metrics["config"] or checkpoint["epoch"] != metrics["selected_epoch"]
            or (config["subset"], config["seed"], config["cap"]) != (subset, seed, cap)):
        raise ValueError(f"Teacher checkpoint or prepared data mismatch for {subset} seed {seed}")
    teacher = Teacher(config["features"]).to(device)
    teacher.load_state_dict(checkpoint["model"])
    teacher.eval().requires_grad_(False)
    if (sources["train.npz"], sources["validation.npz"]) != (train_hash, validation_hash):
        if verify_with is None:
            raise ValueError(f"Teacher prepared data fingerprint mismatch for {subset} seed {seed}")
        prediction = predict_cycles(teacher, batches(verify_with, cap, 128), device, cap)
        difference = prediction_parity(folder, "validation", verify_with, prediction)
        print(f"{subset} seed {seed}: teacher validation parity max difference {difference:.6g} cycles", flush=True)
    return teacher, digest(path)


def train(model, adapter, teacher, config, directory, train_data, validation, device):
    directory.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.Adam([*model.parameters(), *adapter.parameters()], lr=config["lr"])
    generator = torch.Generator().manual_seed(config["seed"])
    loader = batches(train_data, config["cap"], config["batch_size"], generator)
    val_loader = batches(validation, config["cap"], config["batch_size"])
    history = []
    prior, _ = load_latest(directory)
    if prior is not None:
        if prior["config"] != config:
            raise ValueError(f"Run configuration changed: {directory}")
        model.load_state_dict(prior["model"])
        adapter.load_state_dict(prior["adapter"])
        optimizer.load_state_dict(prior["optimizer"])
        restore_rng(prior["rng"], generator)
        history = prior["history"]
    supervised, output, hidden, attention, sparse = config["loss_weights"]
    for epoch in range(len(history) + 1, config["epochs"] + 1):
        start = time.perf_counter()
        model.train()
        loss_sum = event_count = event_slots = 0
        for x, mask, target in loader:
            x, mask, target = x.to(device), mask.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred, membrane, weights, spikes = model.signals(x, mask)
            rate = spikes.sum() / (mask.sum() * spikes.size(-1))
            loss = supervised * F.mse_loss(pred, target) + sparse * rate
            if output or hidden or attention:
                with torch.no_grad():
                    reference, states, reference_weights = teacher.signals(x, mask)
            if output:
                loss = loss + output * F.mse_loss(pred, reference)
            if hidden:
                loss = loss + hidden * F.mse_loss(adapter(membrane[mask]), states[mask])
            if attention:
                loss = loss + attention * F.mse_loss(weights[mask], reference_weights[mask])
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss at {directory}, epoch {epoch}")
            loss.backward()
            nn.utils.clip_grad_norm_([*model.parameters(), *adapter.parameters()], 1.0, error_if_nonfinite=True)
            optimizer.step()
            loss_sum += loss.item() * len(x)
            event_count += spikes.detach().sum().item()
            event_slots += mask.sum().item() * spikes.size(-1)
        val = predict_cycles(model, val_loader, device, config["cap"])
        history.append(dict(epoch=epoch, loss=loss_sum / len(train_data["y"]),
                            spike_rate=event_count / event_slots, seconds=time.perf_counter() - start,
                            validation=regression_metrics(validation["y"], val)))
        save_checkpoint(dict(epoch=epoch, config=config, model=model.state_dict(), adapter=adapter.state_dict(),
                             optimizer=optimizer.state_dict(), rng=rng_state(generator), history=history), directory)
        write_json(directory / "history.json", history)
        print(f'{directory}: epoch {epoch}/{config["epochs"]}, RMSE {history[-1]["validation"]["RMSE"]:.3f} cycles', flush=True)


def run(args):
    device = torch.device(args.device)
    variants = VARIANTS | ABLATIONS
    results = []
    for subset in args.subsets:
        prepared = args.data_root / subset
        data = {split: read_split(prepared, split) for split in ("train", "validation")}
        manifest = json.loads((prepared / "split_manifest.json").read_text())
        cap = float(manifest["rul_cap"])
        train_hash, validation_hash = (digest(prepared / f"{split}.npz") for split in data)
        for seed in args.seeds:
            teacher, teacher_hash = teacher_for(
                args.baselines, subset, seed, train_hash, validation_hash, cap, device,
                verify_with=data["validation"] if args.verify_prepared_predictions else None)
            for name in args.variants:
                directory = args.output / subset / f"seed_{seed}" / name
                config = dict(model=name, subset=subset, seed=seed, epochs=args.epochs, batch_size=128, lr=0.001,
                              cap=cap, loss_weights=list(variants[name]), teacher_sha256=teacher_hash,
                              train_sha256=train_hash, validation_sha256=validation_hash, code_sha256=digest(__file__),
                              model_sha256=digest(distillation.__file__))
                if args.verify_prepared_predictions:
                    config["prepared_data_check"] = "hash or saved teacher validation prediction parity"
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                model = Student(data["train"]["X"].shape[-1]).to(device)
                adapter = nn.Linear(48, teacher.readout.in_features).to(device)
                if args.phase == "train":
                    train(model, adapter, teacher, config, directory, data["train"], data["validation"], device)
                latest, _ = load_latest(directory)
                if latest is None or latest["config"] != config or latest["epoch"] != args.epochs:
                    raise RuntimeError(f"Complete the matching training run first: {directory}")
                best = min(latest["history"], key=lambda row: row["validation"]["RMSE"])["epoch"]
                state = torch.load(directory / f"epoch_{best:04d}.pt", map_location="cpu", weights_only=True)
                if state["config"] != config or state["epoch"] != best:
                    raise RuntimeError(f"Selected checkpoint mismatch: {directory}")
                model.load_state_dict(state["model"])
                split = "validation" if args.phase == "train" else "test"
                if split not in data:
                    data[split] = read_split(prepared, split)
                    if len(np.unique(data[split]["engine_id"])) != len(data[split]["engine_id"]):
                        raise ValueError("Official test requires one prediction per engine")
                prediction = predict_cycles(model, batches(data[split], cap, 128), device, cap)
                results.append(save_result(directory, config, split, data[split], prediction,
                                           best, sum(p.numel() for p in model.parameters())))
    write_json(args.output / f"{args.phase}_summary.json", aggregate(results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for path in ("data-root", "baselines", "output"):
        parser.add_argument(f"--{path}", type=Path, required=True)
    parser.add_argument("--subsets", nargs="+", choices=["FD001", "FD002", "FD003", "FD004"],
                        default=["FD001", "FD002", "FD003", "FD004"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--variants", nargs="+", choices=[*VARIANTS, *ABLATIONS], default=list(VARIANTS))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--phase", choices=["train", "test"], default="train")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--verify-prepared-predictions", action="store_true",
                        help="For a re-created snapshot, accept matching teacher validation predictions and endpoints.")
    args = parser.parse_args()
    if args.epochs < 1 or args.cpu_threads < 1 or args.output.resolve() == args.baselines.resolve():
        parser.error("Use positive epochs and a separate student output directory")
    torch.set_num_threads(args.cpu_threads)
    run(args)

"""Shared baseline protocol. Training never opens the official test archive."""
import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from baseline_models import MODELS, ConvertedSNN, calibrate_conversion, make_model
from metrics import regression_metrics
from train_pilot import digest, load_latest, restore_rng, rng_state, save_checkpoint


def read_split(directory, split):
    with np.load(directory / f"{split}.npz", allow_pickle=False) as archive:
        keys = ["X", "mask", "y", "y_raw", "engine_id", "end_cycle"]
        return {key: archive[key] for key in keys}


def batches(data, cap, size, generator=None):
    dataset = TensorDataset(torch.from_numpy(data["X"]), torch.from_numpy(data["mask"]),
                            torch.from_numpy(data["y"] / cap))
    return DataLoader(dataset, batch_size=size, shuffle=generator is not None, generator=generator)


@torch.no_grad()
def predict_cycles(model, loader, device, cap):
    model.eval()
    normalized = [model(x.to(device), mask.to(device)).cpu().numpy() for x, mask, _ in loader]
    return np.concatenate(normalized).astype(np.float64) * cap


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def configuration(args, name, subset, seed, fingerprints, x, cap):
    return dict(model=name, subset=subset, seed=seed, window=x.shape[1], features=x.shape[2],
                epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                cap=cap, conversion_steps=args.conversion_steps,
                training_target="capped_rul_divided_by_cap", selection="minimum_capped_validation_rmse",
                source_sha256=fingerprints)


def checkpoint_model(directory, config, device):
    latest, _ = load_latest(directory)
    if latest is None or latest["config"] != config or latest["epoch"] != config["epochs"]:
        raise RuntimeError(f"Complete the matching training run first: {directory}")
    best_epoch = min(latest["history"], key=lambda row: row["validation"]["RMSE"])["epoch"]
    state = torch.load(directory / f"epoch_{best_epoch:04d}.pt", map_location="cpu", weights_only=True)
    if state["config"] != config or state["epoch"] != best_epoch:
        raise RuntimeError(f"Selected checkpoint does not match its run: {directory}")
    model = make_model(config["model"], config["window"], config["features"]).to(device)
    model.load_state_dict(state["model"])
    return model.eval(), best_epoch


def fit(config, directory, train_data, validation, device, teacher=None):
    directory.mkdir(parents=True, exist_ok=True)
    seed, cap = config["seed"], config["cap"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    train_loader = batches(train_data, cap, config["batch_size"], generator)
    val_loader = batches(validation, cap, config["batch_size"])
    model = make_model(config["model"], config["window"], config["features"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    previous, _ = load_latest(directory)
    history = []
    if previous is not None:
        if previous["config"] != config:
            raise ValueError(f"Configuration changed. Use another output directory: {directory}")
        model.load_state_dict(previous["model"])
        optimizer.load_state_dict(previous["optimizer"])
        restore_rng(previous["rng"], generator)
        history = previous["history"]
    for epoch in range(len(history) + 1, config["epochs"] + 1):
        model.train()
        total = 0.0
        for x, mask, target in train_loader:
            x, mask, target = x.to(device), mask.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x, mask)
            loss = nn.functional.mse_loss(prediction, target)
            if teacher is not None:
                with torch.no_grad():
                    guidance = teacher(x, mask)
                loss = 0.5 * loss + 0.5 * nn.functional.mse_loss(prediction, guidance)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss in {directory}, epoch {epoch}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            total += loss.item() * len(x)
        cycles = predict_cycles(model, val_loader, device, cap)
        row = dict(epoch=epoch, training_loss_scaled=total / len(train_data["y"]),
                   validation=regression_metrics(validation["y"], cycles))
        history.append(row)
        save_checkpoint(dict(epoch=epoch, config=config, model=model.state_dict(),
                             optimizer=optimizer.state_dict(), history=history, rng=rng_state(generator)), directory)
        print(f'{config["subset"]} {config["model"]} seed={seed} epoch={epoch} '
              f'validation RMSE={row["validation"]["RMSE"]:.4f} cycles', flush=True)
    write_json(directory / "history.json", history)


def save_result(directory, config, split, data, prediction, best_epoch, parameters):
    result = dict(config=config, split=split, selected_epoch=best_epoch, parameters=parameters,
                  capped=regression_metrics(data["y"], prediction),
                  raw=regression_metrics(data["y_raw"], prediction),
                  prediction_units="cycles", predictions_clipped=False)
    write_json(directory / f"{split}_metrics.json", result)
    with (directory / f"{split}_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["engine_id", "end_cycle", "target_capped_cycles", "target_raw_cycles", "prediction_cycles"])
        writer.writerows(zip(data["engine_id"], data["end_cycle"], data["y"], data["y_raw"], prediction))
    return result


def aggregate(results):
    summary = []
    groups = {(row["config"]["subset"], row["config"]["model"]) for row in results}
    for subset, name in sorted(groups):
        rows = [row for row in results if (row["config"]["subset"], row["config"]["model"]) == (subset, name)]
        for target in ("capped", "raw"):
            row = dict(subset=subset, model=name, target=target, seeds=[r["config"]["seed"] for r in rows])
            for key in ("MAE", "MSE", "RMSE", "R2", "cmapss_score"):
                values = [r[target][key] for r in rows]
                row[key] = dict(mean=float(np.mean(values)), sample_sd=float(np.std(values, ddof=1)) if len(values) > 1 else None) if None not in values else None
            summary.append(row)
    return summary


def run(args):
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    results = []
    for subset in args.subsets:
        data_dir = args.data_root / subset
        manifest = json.loads((data_dir / "split_manifest.json").read_text(encoding="utf-8"))
        cap = float(manifest["rul_cap"])
        if not np.isfinite(cap) or cap <= 0:
            raise ValueError(f"Invalid RUL scale in {data_dir}")
        train_data = read_split(data_dir, "train")
        validation = read_split(data_dir, "validation")
        sources = ("train.npz", "validation.npz", "preprocessing.json", "split_manifest.json")
        fingerprints = {item: digest(data_dir / item) for item in sources}
        for filename in ("run_baselines.py", "baseline_models.py", "metrics.py", "train_pilot.py"):
            fingerprints[filename] = digest(Path(__file__).with_name(filename))
        test = None
        for seed in args.seeds:
            root = args.output / subset / f"seed_{seed}"
            configs = {name: configuration(args, name, subset, seed, fingerprints, train_data["X"], cap)
                       for name in MODELS}
            for name in MODELS:
                if name not in args.models:
                    continue
                config = configs[name]
                directory = root / name
                model, best_epoch = None, None
                if name == "ann_to_snn":
                    source, best_epoch = checkpoint_model(root / "deepnn", configs["deepnn"], device)
                    thresholds = calibrate_conversion(source, batches(train_data, cap, args.batch_size), device)
                    model = ConvertedSNN(source, thresholds, args.conversion_steps).to(device)
                    config = dict(config, source_model="deepnn", calibration_thresholds=thresholds.tolist())
                elif name != "mean":
                    if args.phase == "train":
                        teacher = None
                        if name == "small_lstm_kd":
                            teacher, _ = checkpoint_model(root / "attention_bilstm", configs["attention_bilstm"], device)
                        fit(config, directory, train_data, validation, device, teacher)
                    model, best_epoch = checkpoint_model(directory, config, device)
                if args.phase == "test" and test is None:
                    test = read_split(data_dir, "test")
                    if len(np.unique(test["engine_id"])) != len(test["engine_id"]):
                        raise ValueError("Official evaluation requires one window per engine")
                data = validation if args.phase == "train" else test
                prediction = (np.full(len(data["y"]), train_data["y"].mean(dtype=np.float64))
                              if model is None else predict_cycles(model, batches(data, cap, args.batch_size), device, cap))
                parameters = 0 if model is None else sum(p.numel() for p in model.parameters())
                split = "validation" if args.phase == "train" else "test"
                results.append(save_result(directory, config, split, data, prediction, best_epoch, parameters))
    write_json(args.output / f"{args.phase}_summary.json", aggregate(results))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("train", "test"), default="train")
    parser.add_argument("--subsets", nargs="+", choices=("FD001", "FD002", "FD003", "FD004"), default=["FD001"])
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--conversion-steps", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.conversion_steps) < 1 or not np.isfinite(args.lr) or args.lr <= 0:
        parser.error("Epochs, batch size, conversion steps and learning rate must be positive")
    if any(seed < 0 or seed >= 2**32 for seed in args.seeds) or len(set(args.seeds)) != len(args.seeds):
        parser.error("Seeds must be distinct unsigned 32-bit integers")
    run(args)


if __name__ == "__main__":
    main()

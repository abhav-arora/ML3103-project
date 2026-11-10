"""Train a controlled width sweep using the existing student and distillation trainer."""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from spikedistill.models import distillation
from spikedistill.models.distillation import Student
from spikedistill.training.baselines import aggregate, batches, predict_cycles, read_split, save_result, write_json
from spikedistill.training.distill import VARIANTS, teacher_for, train
from spikedistill.training.pilot import digest, load_latest


def run(args):
    device = torch.device(args.device)
    results = []
    for subset in args.subsets:
        prepared = args.data_root / subset
        data = {split: read_split(prepared, split) for split in ("train", "validation")}
        manifest = json.loads((prepared / "split_manifest.json").read_text())
        cap = float(manifest["rul_cap"])
        train_hash, val_hash = (digest(prepared / f"{split}.npz") for split in data)
        for seed in args.seeds:
            teacher, teacher_hash = teacher_for(
                args.baselines, subset, seed, train_hash, val_hash, cap, device,
                verify_with=data["validation"] if args.verify_prepared_predictions else None)
            for width in args.widths:
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                model = Student(data["train"]["X"].shape[-1], width=width).to(device)
                adapter = nn.Linear(width, teacher.readout.in_features).to(device)
                name = f"spikedistill_w{width}"
                directory = args.output / subset / f"seed_{seed}" / name
                config = dict(model=name, subset=subset, seed=seed, width=width,
                              window=manifest["window"], features=data["train"]["X"].shape[-1],
                              epochs=args.epochs, batch_size=128, lr=0.001, cap=cap,
                              loss_weights=list(VARIANTS["spikedistill"]), device=args.device,
                              teacher_sha256=teacher_hash, train_sha256=train_hash, validation_sha256=val_hash,
                              code_sha256=digest(__file__), model_sha256=digest(distillation.__file__),
                              trainer_sha256=digest(Path(__file__).resolve().parents[2] / "spikedistill/training/distill.py"),
                              parameters=sum(p.numel() for p in model.parameters()))
                if args.phase == "train":
                    train(model, adapter, teacher, config, directory, data["train"], data["validation"], device)
                latest, _ = load_latest(directory)
                if (latest is None or latest["config"] != config or latest["epoch"] != args.epochs
                        or len(latest["history"]) != args.epochs):
                    raise RuntimeError(f"Complete the matching training run first: {directory}")
                best = min(latest["history"], key=lambda row: row["validation"]["RMSE"])["epoch"]
                state = torch.load(directory / f"epoch_{best:04d}.pt", map_location="cpu", weights_only=True)
                if state["config"] != config or state["epoch"] != best:
                    raise ValueError(f"Selected checkpoint metadata mismatch in {directory}")
                model.load_state_dict(state["model"])
                split = "validation" if args.phase == "train" else "test"
                if split not in data:
                    data[split] = read_split(prepared, split)
                    if len(np.unique(data[split]["engine_id"])) != len(data[split]["engine_id"]):
                        raise ValueError("Official test requires one window per engine")
                prediction = predict_cycles(model, batches(data[split], cap, 128), device, cap)
                result = save_result(directory, config, split, data[split], prediction, best, config["parameters"])
                results.append(result)
                print(f"{subset} seed {seed} width {width}: {split} raw RMSE {result['raw']['RMSE']:.4f}, "
                      f"parameters {config['parameters']}", flush=True)
    write_json(args.output / f"{args.phase}_summary.json", aggregate(results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("data-root", "baselines", "output"):
        parser.add_argument("--" + arg, type=Path, required=True)
    parser.add_argument("--subsets", nargs="+", choices=("FD001", "FD002", "FD003", "FD004"),
                        default=["FD001", "FD002", "FD003", "FD004"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--widths", nargs="+", type=int, default=[16, 32, 48, 64])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--phase", choices=("train", "test"), default="train")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--verify-prepared-predictions", action="store_true")
    args = parser.parse_args()
    if (args.epochs < 1 or any(width < 1 for width in args.widths)
            or len(set(args.widths)) != len(args.widths)
            or args.output.resolve() in (args.data_root.resolve(), args.baselines.resolve())):
        parser.error("Use positive unique widths and epochs, with a separate output folder")
    torch.set_num_threads(1)
    run(args)

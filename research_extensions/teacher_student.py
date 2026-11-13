"""Compare convolutional and GRU teachers and their distilled SNN students."""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, TensorDataset

from spikedistill.models import distillation
from spikedistill.models.distillation import Student
from spikedistill.evaluation.metrics import regression_metrics
from spikedistill.training.baselines import batches, predict_cycles, read_split, save_result, write_json
from spikedistill.training.pilot import digest, load_latest, restore_rng, rng_state, save_checkpoint


SUBSETS = ("FD001", "FD002", "FD003", "FD004")
TEACHERS = ("temporal_cnn", "gru_memory")
GUIDANCE = {
    "snn_supervised": (),
    "snn_from_cnn": (0,),
    "snn_from_gru": (1,),
    "snn_from_both": (0, 1),
}


class TemporalCNN(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.input = nn.Conv1d(features, 48, 1)
        self.blocks = nn.ModuleList(nn.Conv1d(48, 48, 3, padding=d, dilation=d) for d in (1, 2, 4))
        self.readout = nn.Sequential(nn.Linear(96 + features, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 1))

    def forward(self, x, mask):
        valid = mask[:, None, :]
        state = self.input(x.transpose(1, 2) * valid) * valid
        for layer in self.blocks:
            state = (state + F.gelu(layer(state))) * valid
        pooled = state.sum(2) / mask.sum(1, keepdim=True)
        # Prepared windows end at the latest observed cycle.
        return self.readout(torch.cat((pooled, state[:, :, -1], x[:, -1]), dim=1)).squeeze(1)


class GRUMemory(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.input = nn.Linear(features, 32)
        self.memory = nn.GRU(32, 64, batch_first=True)
        self.readout = nn.Sequential(nn.Linear(64 + features, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 1))

    def forward(self, x, mask):
        state = F.gelu(self.input(x))
        if not bool(mask.all()):
            order = torch.argsort(~mask, dim=1, stable=True)
            state = state.gather(1, order.unsqueeze(-1).expand_as(state))
            state = pack_padded_sequence(state, mask.sum(1).cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.memory(state)
        return self.readout(torch.cat((hidden[-1], x[:, -1]), dim=1)).squeeze(1)


def build(name, features):
    if name == "temporal_cnn":
        return TemporalCNN(features)
    if name == "gru_memory":
        return GRUMemory(features)
    return Student(features, width=64)


def raw_loader(data, cap, size, generator=None):
    return batches(dict(data, y=data["y_raw"]), cap, size, generator)


def fit(model, config, directory, training, validation, device, guidance):
    directory.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    generator = torch.Generator().manual_seed(config["seed"])
    dataset = raw_loader(training, config["cap"], config["batch_size"]).dataset
    if guidance:
        teacher_loader = DataLoader(dataset, batch_size=256)
        predictions = [predict_cycles(teacher, teacher_loader, device, config["cap"]) for teacher in guidance]
        soft_target = torch.from_numpy(np.mean(predictions, axis=0) / config["cap"]).float()
        dataset = TensorDataset(*dataset.tensors, soft_target)
    train_loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, generator=generator)
    val_loader = batches(validation, config["cap"], 256)
    history = []
    previous, _ = load_latest(directory)
    if previous is not None:
        if previous["config"] != config:
            raise ValueError(f"Run configuration changed in {directory}")
        model.load_state_dict(previous["model"])
        optimizer.load_state_dict(previous["optimizer"])
        restore_rng(previous["rng"], generator)
        history = previous["history"]
    for epoch in range(len(history) + 1, config["epochs"] + 1):
        model.train()
        total = 0.0
        for x, mask, target, *soft_target in train_loader:
            x, mask, target = x.to(device), mask.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x, mask)
            loss = F.smooth_l1_loss(prediction, target, beta=0.2)
            if soft_target:
                loss = 0.6 * loss + 0.4 * F.smooth_l1_loss(prediction, soft_target[0].to(device), beta=0.2)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss in {directory}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            total += loss.item() * len(x)
        prediction = predict_cycles(model, val_loader, device, config["cap"])
        metrics = regression_metrics(validation["y_raw"], prediction)
        history.append(dict(epoch=epoch, validation=metrics, loss=total / len(training["y_raw"])))
        save_checkpoint(dict(epoch=epoch, config=config, model=model.state_dict(),
                             optimizer=optimizer.state_dict(), history=history, rng=rng_state(generator)), directory)
        print(f"{config['subset']} {config['model']} seed {config['seed']} epoch {epoch}: "
              f"raw validation RMSE {metrics['RMSE']:.3f}", flush=True)
    write_json(directory / "history.json", history)


def selected(model, config, directory, device):
    latest, _ = load_latest(directory)
    if latest is None or latest["config"] != config or latest["epoch"] != config["epochs"]:
        raise RuntimeError(f"Complete the matching training run first: {directory}")
    epoch = min(latest["history"], key=lambda row: row["validation"]["RMSE"])["epoch"]
    path = directory / f"epoch_{epoch:04d}.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["config"] != config or checkpoint["epoch"] != epoch:
        raise ValueError(f"Selected checkpoint metadata mismatch in {directory}")
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval().requires_grad_(False), epoch, digest(path)


def read_inputs(path):
    training, validation = (read_split(path, split) for split in ("train", "validation"))
    if np.intersect1d(training["engine_id"], validation["engine_id"]).size:
        raise ValueError(f"Training and validation overlap in {path}")
    if np.unique(validation["engine_id"]).size != len(validation["engine_id"]):
        raise ValueError(f"Expected one validation window per engine in {path}")
    for data in (training, validation):
        if not data["mask"][:, -1].all():
            raise ValueError(f"Expected latest cycle in final position in {path}")
    manifest = json.loads((path / "split_manifest.json").read_text())
    for data, key in ((training, "train_engine_ids"), (validation, "validation_engine_ids")):
        if set(data["engine_id"]) != set(manifest[key]):
            raise ValueError(f"Prepared arrays disagree with split manifest in {path}")
    return training, validation, float(manifest["rul_cap"])


def run(args):
    device = torch.device(args.device)
    completed = []
    for subset in args.subsets:
        path = args.data_root / subset
        training, validation, cap = read_inputs(path)
        fingerprints = {name: digest(path / name) for name in
                        ("train.npz", "validation.npz", "split_manifest.json", "preprocessing.json")}
        for seed in args.seeds:
            teachers, teacher_hashes = [], []
            for name in (*TEACHERS, *GUIDANCE):
                indexes = GUIDANCE.get(name, ())
                config = dict(model=name, subset=subset, seed=seed, cap=cap,
                              epochs=args.teacher_epochs if name in TEACHERS else args.student_epochs,
                              batch_size=256, features=training["X"].shape[-1], device=args.device,
                              cpu_threads=args.cpu_threads, selection="minimum raw validation RMSE",
                              training_target="raw RUL divided by cap for scale only",
                              teachers={TEACHERS[i]: teacher_hashes[i] for i in indexes},
                              source_sha256=fingerprints, code_sha256=digest(__file__),
                              student_code_sha256=digest(distillation.__file__))
                directory = args.output / subset / f"seed_{seed}" / name
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                model = build(name, config["features"]).to(device)
                if args.phase == "train":
                    fit(model, config, directory, training, validation, device, [teachers[i] for i in indexes])
                model, epoch, sha = selected(model, config, directory, device)
                if name in TEACHERS:
                    teachers.append(model)
                    teacher_hashes.append(sha)
                completed.append((path, directory, config, model if name in TEACHERS else model.cpu(), epoch, sha))
            for teacher in teachers:
                teacher.cpu()
    # All configurations and checkpoint selections are fixed before loading any test data.
    manifest = args.output / "frozen_checkpoints.json"
    frozen = [
        dict(config=config, selected_epoch=epoch, checkpoint_sha256=sha)
        for _, _, config, _, epoch, sha in completed]
    if args.phase == "train":
        write_json(manifest, frozen)
    elif json.loads(manifest.read_text()) != frozen:
        raise ValueError("Selected checkpoints differ from the validation manifest")
    split = "validation" if args.phase == "train" else "test"
    current_path = None
    for path, directory, config, model, epoch, _ in completed:
        if path != current_path:
            data = read_split(path, split)
            if not data["mask"][:, -1].all() or len(np.unique(data["engine_id"])) != len(data["engine_id"]):
                raise ValueError(f"Expected one observed endpoint per engine in {path}/{split}")
            current_path = path
        prediction = predict_cycles(model.to(device), batches(data, config["cap"], 256), device, config["cap"])
        result = save_result(directory, config, split, data, prediction, epoch,
                             sum(parameter.numel() for parameter in model.parameters()))
        model.to("cpu")
        print(f"{config['subset']} {config['model']} seed {config['seed']}: "
              f"raw {split} RMSE {result['raw']['RMSE']:.3f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--subsets", nargs="+", choices=SUBSETS, default=list(SUBSETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--teacher-epochs", type=int, default=20)
    parser.add_argument("--student-epochs", type=int, default=20)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--phase", choices=("train", "test"), default="train")
    args = parser.parse_args()
    if min(args.teacher_epochs, args.student_epochs, args.cpu_threads) < 1:
        parser.error("Epochs and CPU threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    run(args)

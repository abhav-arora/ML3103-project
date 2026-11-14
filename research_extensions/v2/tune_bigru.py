"""Tune convolutional additive-attention BiGRU teachers on FD002 and FD004."""
import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset

from spikedistill.data.prepare import fit_preprocessor, load_trajectories, split_engines, transform
from spikedistill.evaluation.metrics import regression_metrics
from spikedistill.training.baselines import save_result, write_json
from spikedistill.training.pilot import digest, load_latest, restore_rng, rng_state, save_checkpoint


SUBSETS = {"FD002": 259, "FD004": 248}
# Predeclare the validation sweep, including an anchor close to the proposed model.
CANDIDATES = {
    "anchor30": dict(window=30, hidden=96, dropout=0.25, lr=0.0005, cap=125, smooth=False),
    "long60": dict(window=60, hidden=96, dropout=0.25, lr=0.0005, cap=125, smooth=False),
    "smooth60": dict(window=60, hidden=96, dropout=0.25, lr=0.0005, cap=145, smooth=True),
    "smooth90": dict(window=90, hidden=64, dropout=0.20, lr=0.0008, cap=145, smooth=True),
    "raw60": dict(window=60, hidden=96, dropout=0.30, lr=0.00035, cap=None, smooth=True),
}
SCALE = 125.0


class AttnBiGRU(nn.Module):
    """Local convolution, bidirectional memory, and masked additive attention."""

    def __init__(self, n_feat, hid=96, layers=2, p=0.25):
        super().__init__()
        self.front = nn.Sequential(nn.Conv1d(n_feat, 64, 3, padding=1), nn.GELU())
        self.rnn = nn.GRU(64, hid, layers, batch_first=True, bidirectional=True,
                          dropout=p if layers > 1 else 0.0)
        self.att = nn.Sequential(nn.Linear(2 * hid, hid), nn.Tanh(), nn.Linear(hid, 1))
        self.head = nn.Sequential(nn.Dropout(p), nn.Linear(2 * hid, 64),
                                  nn.GELU(), nn.Linear(64, 1))

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        x = x.masked_fill(~mask.unsqueeze(-1), 0)
        x = self.front(x.transpose(1, 2)).transpose(1, 2)
        # Windows are right-padded. Packing prevents the backward GRU seeing padding.
        if not bool(mask.all()):
            packed = pack_padded_sequence(x, mask.sum(1).cpu(), batch_first=True, enforce_sorted=False)
            h, _ = self.rnn(packed)
            h, _ = pad_packed_sequence(h, batch_first=True, total_length=x.shape[1])
        else:
            h, _ = self.rnn(x)
        weights = self.att(h).masked_fill(~mask.unsqueeze(-1), -torch.inf).softmax(dim=1)
        return self.head((weights * h).sum(1)).squeeze(-1)


def observed_features(rows, prep, smooth):
    features = transform(rows, prep)
    if not smooth:
        return features
    sensors = features[:, :-3]
    averaged = sensors.copy()
    for cycle in range(1, len(sensors)):
        averaged[cycle] = 0.2 * sensors[cycle] + 0.8 * averaged[cycle - 1]
    return np.concatenate((features, averaged), axis=1)


class Windows(Dataset):
    def __init__(self, frame, prep, config, split, official_rul=None):
        self.window = config["window"]
        self.cap = config["cap"]
        self.engines = {}
        self.endpoints = []
        for engine, rows in frame.groupby("engine_id", sort=True):
            features = observed_features(rows, prep, config["smooth"])
            self.engines[engine] = features
            length = len(features)
            if split == "train":
                # Include short observed prefixes, then sample every fourth cycle.
                ends = range(min(12, length), length + 1, 4)
                ends = sorted(set(ends) | {length})
            elif split == "validation":
                # Multiple fixed horizons per held-out engine, independent of test.
                ends = sorted({max(1, min(length - 1, int(length * f)))
                               for f in (0.30, 0.50, 0.70, 0.90)})
            else:
                ends = (length,)
            for end in ends:
                raw = length - end if official_rul is None else official_rul[engine - 1]
                self.endpoints.append((engine, end, raw))

    def __len__(self):
        return len(self.endpoints)

    def __getitem__(self, index):
        engine, end, raw = self.endpoints[index]
        observed = self.engines[engine][max(0, end - self.window):end]
        x = np.zeros((self.window, observed.shape[1]), np.float32)
        x[:len(observed)] = observed
        mask = np.arange(self.window) < len(observed)
        label = raw if self.cap is None else min(self.cap, raw)
        return x, mask, np.float32(label / SCALE)

    def metric_inputs(self):
        engines, ends, raw = map(np.asarray, zip(*self.endpoints))
        return dict(engine_id=engines, end_cycle=ends, y_raw=raw, y=np.minimum(125, raw))


def model_for(data, config, device):
    return AttnBiGRU(data[0][0].shape[-1], hid=config["hidden"], p=config["dropout"]).to(device)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    outputs = [model(x.to(device), mask.to(device)).cpu().numpy() for x, mask, _ in loader]
    return np.concatenate(outputs).astype(np.float64) * SCALE


def fit(config, train, validation, folder, device):
    folder.mkdir(parents=True, exist_ok=True)
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = model_for(train, config, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(train, batch_size=config["batch_size"], shuffle=True, generator=generator)
    validation_loader = DataLoader(validation, batch_size=config["batch_size"])
    targets = validation.metric_inputs()["y_raw"]
    checkpoint, _ = load_latest(folder)
    history = []
    if checkpoint is not None:
        if checkpoint["config"] != config:
            raise ValueError(f"Settings changed. Use a fresh run folder: {folder}")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        restore_rng(checkpoint["rng"], generator)
        history = checkpoint["history"]
    for epoch in range(len(history) + 1, config["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        for x, mask, y in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x.to(device), mask.to(device))
            loss = F.mse_loss(prediction, y.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            epoch_loss += loss.detach().item() * len(x)
        rmse = regression_metrics(targets, predict(model, validation_loader, device))["RMSE"]
        history.append(dict(epoch=epoch, rmse=rmse, loss=epoch_loss / len(train)))
        save_checkpoint(dict(epoch=epoch, config=config, model=model.state_dict(),
                             optimizer=optimizer.state_dict(), history=history,
                             rng=rng_state(generator)), folder)
        print(f"{config['subset']} {config['candidate']} epoch {epoch}: "
              f"raw validation RMSE {rmse:.3f}", flush=True)
    best = min(history, key=lambda row: row["rmse"])
    path = folder / f"epoch_{best['epoch']:04d}.pt"
    return dict(config=config, **best, checkpoint_sha256=digest(path))


def training_inputs(source, subset, split_seed):
    path = source / f"train_{subset}.txt"
    original = load_trajectories(path)
    train_ids, validation_ids = split_engines(original, split_seed)
    train = original[original.engine_id.isin(train_ids)]
    validation = original[original.engine_id.isin(validation_ids)]
    return train, validation, fit_preprocessor(train, subset, split_seed), digest(path)


def sweep(args, device):
    selected = []
    for subset in args.subsets:
        train, validation, prep, source_digest = training_inputs(args.data_root, subset, args.split_seed)
        results = []
        for name in args.candidates:
            config = dict(CANDIDATES[name], candidate=name, subset=subset, seed=args.seed,
                          split_seed=args.split_seed, epochs=args.epochs, batch_size=args.batch_size,
                          source_sha256=source_digest, code_sha256=digest(__file__),
                          device=args.device, torch_version=str(torch.__version__))
            folder = args.output / subset / name
            result = fit(config, Windows(train, prep, config, "train"),
                         Windows(validation, prep, config, "validation"), folder, device)
            results.append(result)
        write_json(args.output / subset / "validation_candidates.json", results)
        selected.append(min(results, key=lambda row: row["rmse"]))
    write_json(args.output / "frozen_teachers.json", selected)


def evaluate(args, device):
    selections = json.loads((args.output / "frozen_teachers.json").read_text(encoding="utf-8"))
    comparisons = []
    for best in selections:
        config = best["config"]
        subset = config["subset"]
        _, _, prep, source_digest = training_inputs(args.data_root, subset, config["split_seed"])
        folder = args.output / subset / config["candidate"]
        checkpoint = folder / f"epoch_{best['epoch']:04d}.pt"
        if (digest(checkpoint) != best["checkpoint_sha256"]
                or source_digest != config["source_sha256"] or digest(__file__) != config["code_sha256"]):
            raise ValueError(f"Frozen inputs, code or checkpoint changed for {subset}")
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if state["config"] != config or state["epoch"] != best["epoch"]:
            raise ValueError(f"Checkpoint metadata mismatch for {subset}")
        official = load_trajectories(args.data_root / f"test_{subset}.txt")
        targets = np.loadtxt(args.data_root / f"RUL_{subset}.txt", ndmin=1)
        count = SUBSETS[subset]
        if (not np.array_equal(np.sort(official.engine_id.unique()), np.arange(1, count + 1))
                or targets.shape != (count,) or not np.isfinite(targets).all() or (targets < 0).any()):
            raise ValueError(f"Incomplete official test set for {subset}")
        data = Windows(official, prep, config, "test", targets)
        model = model_for(data, config, device)
        model.load_state_dict(state["model"])
        prediction = predict(model, DataLoader(data, batch_size=config["batch_size"]), device)
        result = save_result(folder, config, "test", data.metric_inputs(), prediction,
                             best["epoch"], sum(p.numel() for p in model.parameters()))
        comparisons.append(dict(subset=subset, candidate=config["candidate"], seed=config["seed"],
                                validation_rmse=best["rmse"], test_raw_rmse=result["raw"]["RMSE"],
                                test_capped125_rmse=result["capped"]["RMSE"], engines=count))
        print(comparisons[-1], flush=True)
    with (args.output / "teacher_test_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subsets", nargs="+", choices=SUBSETS, default=list(SUBSETS))
    parser.add_argument("--candidates", nargs="+", choices=CANDIDATES, default=list(CANDIDATES))
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--phase", choices=("tune", "test"), default="tune")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.threads) < 1:
        parser.error("Epochs, batch size and thread count must be positive")
    torch.set_num_threads(args.threads)
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    (sweep if args.phase == "tune" else evaluate)(args, device)


if __name__ == "__main__":
    main()

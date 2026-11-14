"""Teacher-only benchmark on the original prepared FD002 and FD004 archives."""
import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from research_extensions.teacher_student import read_inputs, selected
from research_extensions.v2.tune_bigru import AttnBiGRU
from spikedistill.evaluation.metrics import regression_metrics
from spikedistill.training.baselines import batches, predict_cycles, read_split, save_result, write_json
from spikedistill.training.pilot import digest, load_latest, restore_rng, rng_state, save_checkpoint


COUNTS = {"FD002": 259, "FD004": 248}
TRAIN_FILES = ("train.npz", "validation.npz", "split_manifest.json", "preprocessing.json")
SETTINGS = {
    "additive_bigru": (512, 0.001),
    "gru_skip": (256, 0.001),
    "residual_cnn": (256, 0.001),
}


class GRUSkip(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.memory = nn.GRU(features, 64, batch_first=True)
        self.head = nn.Sequential(nn.Linear(64 + 2 * features, 64), nn.GELU(),
                                  nn.Dropout(0.15), nn.Linear(64, 1))

    def forward(self, x, mask):
        average = x.masked_fill(~mask.unsqueeze(-1), 0).sum(1) / mask.sum(1, keepdim=True)
        observed = x
        if not bool(mask.all()):
            order = torch.argsort(~mask, dim=1, stable=True)
            observed = x.gather(1, order.unsqueeze(-1).expand_as(x))
            observed = pack_padded_sequence(observed, mask.sum(1).cpu(), batch_first=True,
                                            enforce_sorted=False)
        _, hidden = self.memory(observed)
        return self.head(torch.cat((hidden[-1], average, x[:, -1]), dim=-1)).squeeze(-1)


class ResidualCNN(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.project = nn.Conv1d(features, 48, 1)
        self.blocks = nn.ModuleList(nn.Conv1d(48, 48, 3, padding=d, dilation=d)
                                    for d in (1, 2, 4, 8))
        self.activation = nn.GELU()
        self.head = nn.Sequential(nn.Linear(96 + features, 64), nn.GELU(),
                                  nn.Dropout(0.15), nn.Linear(64, 1))

    def forward(self, x, mask):
        valid = mask[:, None, :]
        state = self.project(x.transpose(1, 2).masked_fill(~valid, 0)).masked_fill(~valid, 0)
        for conv in self.blocks:
            state = (state + self.activation(conv(state))).masked_fill(~valid, 0)
        average = state.sum(-1) / mask.sum(1, keepdim=True)
        return self.head(torch.cat((average, state[:, :, -1], x[:, -1]), dim=-1)).squeeze(-1)


def build(name, features):
    if name == "additive_bigru":
        # Convert the shared left padding to the packing convention of AttnBiGRU.
        return AdditiveLeftPadded(features, hid=48, p=0.20)
    return {"gru_skip": GRUSkip, "residual_cnn": ResidualCNN}[name](features)


class AdditiveLeftPadded(AttnBiGRU):
    def forward(self, x, mask):
        if not bool(mask.all()):
            order = torch.argsort(~mask, dim=1, stable=True)
            x = x.gather(1, order.unsqueeze(-1).expand_as(x))
            mask = mask.gather(1, order)
        return super().forward(x, mask)


def fit(model, config, folder, train, validation, device):
    folder.mkdir(parents=True, exist_ok=True)
    cap = config["cap"]
    batch_size = config["batch_size"]
    generator = torch.Generator().manual_seed(config["seed"])
    training = batches(train, cap, batch_size, generator)
    held_out = batches(validation, cap, batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config["epochs"], eta_min=1e-4)
    history = []
    previous, _ = load_latest(folder)
    if previous is not None:
        if previous["config"] != config:
            raise ValueError(f"Run settings changed: {folder}")
        model.load_state_dict(previous["model"])
        optimizer.load_state_dict(previous["optimizer"])
        scheduler.load_state_dict(previous["scheduler"])
        restore_rng(previous["rng"], generator)
        history = previous["history"]
    for epoch in range(len(history) + 1, config["epochs"] + 1):
        model.train()
        total = 0.0
        for x, mask, y in training:
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(x.to(device), mask.to(device)), y.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            total += loss.detach().item() * len(x)
        scheduler.step()
        prediction = predict_cycles(model, held_out, device, cap)
        metrics = regression_metrics(validation["y"], prediction)
        raw = regression_metrics(validation["y_raw"], prediction)
        history.append(dict(epoch=epoch, validation=metrics, raw_validation=raw,
                            training_loss_scaled=total / len(train["y"])))
        save_checkpoint(dict(epoch=epoch, config=config, model=model.state_dict(),
                             optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                             history=history, rng=rng_state(generator)), folder)
        print(f"{config['subset']} {config['model']} epoch {epoch}: "
              f"validation RMSE capped {metrics['RMSE']:.3f}, raw {raw['RMSE']:.3f}", flush=True)


def configure(args, cap, features, name):
    batch, lr = SETTINGS[name]
    return dict(subset=args.subset, model=name, epochs=args.epochs, seed=args.seed,
                batch_size=batch, lr=lr, cap=cap, features=features, device=args.device,
                threads=args.threads, pytorch=str(torch.__version__),
                training_target="baseline capped RUL", selection="minimum capped validation RMSE",
                source_sha256={key: digest(args.data_root / args.subset / key) for key in TRAIN_FILES},
                code_sha256=digest(__file__), additive_code_sha256=digest(
                    Path(__file__).with_name("tune_bigru.py")))


def candidates(predictions):
    yield from predictions.items()
    names = list(predictions)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            yield f"{a}+{b}", (predictions[a] + predictions[b]) / 2
    yield "all_teachers", np.mean(list(predictions.values()), axis=0)


def selection(predictions, validation):
    scores = {}
    for name, pred in candidates(predictions):
        scores[name] = {
            "capped_rmse": regression_metrics(validation["y"], pred)["RMSE"],
            "raw_rmse": regression_metrics(validation["y_raw"], pred)["RMSE"],
        }
    return min(scores, key=lambda name: scores[name]["capped_rmse"]), scores


def run(args):
    device = torch.device(args.device)
    path = args.data_root / args.subset
    expected = json.loads(Path(__file__).with_name("baseline_data_sha256.json")
                          .read_text(encoding="utf-8-sig"))[args.subset]
    files = (*TRAIN_FILES, "test.npz") if args.phase == "test" else TRAIN_FILES
    for name in files:
        if digest(path / name) != expected[name]:
            raise ValueError(f"Prepared baseline file does not match its reference: {path / name}")
    train, val, cap = read_inputs(path)
    if cap != 125 or train["X"].shape[1] != 30:
        raise ValueError("Expected the baseline 30-cycle arrays with cap 125")
    output = args.output / args.subset
    entries, models, validation_predictions = {}, {}, {}
    for name in ([args.member] if args.member else SETTINGS):
        config = configure(args, cap, train["X"].shape[-1], name)
        directory = output / name
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        model = build(name, config["features"]).to(device)
        if args.phase == "train":
            fit(model, config, directory, train, val, device)
            continue
        model, epoch, sha = selected(model, config, directory, device)
        entries[name] = dict(config=config, epoch=epoch, sha256=sha)
        models[name] = model
        validation_predictions[name] = predict_cycles(model, batches(val, cap, 512), device, cap)
    if args.phase == "train":
        return
    winner, scores = selection(validation_predictions, val)
    frozen = dict(winner=winner, validation=scores, members=entries)
    index = output / "frozen_teacher.json"
    if args.phase == "freeze":
        write_json(index, frozen)
        print(f"{args.subset} validation selection: {winner}, {scores[winner]}", flush=True)
        return
    # Compare checkpoint and preprocessing fingerprints. Do not require bitwise
    # equality of floating-point metrics from an independent GPU process.
    saved = json.loads(index.read_text(encoding="utf-8"))
    if saved["members"] != entries:
        raise ValueError("Frozen checkpoint or input fingerprints changed")
    test_path = path / "test.npz"
    test = read_split(path, "test")
    count = COUNTS[args.subset]
    if len(test["engine_id"]) != count or set(test["engine_id"]) != set(range(1, count + 1)):
        raise ValueError(f"Missing official test engine endpoints: {path}")
    predictions = {name: predict_cycles(model, batches(test, cap, 512), device, cap)
                   for name, model in models.items()}
    test_scores = []
    for name, pred in candidates(predictions):
        members = list(models) if name == "all_teachers" else name.split("+")
        parameters = sum(p.numel() for member in members for p in models[member].parameters())
        config = dict(subset=args.subset, seed=args.seed, model=name, members=members,
                      test_sha256=digest(test_path), validation_selected=name == saved["winner"])
        epoch = entries[name]["epoch"] if name in entries else None
        result = save_result(output / name, config, "test", test, pred, epoch, parameters)
        test_scores.append(dict(model=name, validation_selected=config["validation_selected"],
                                raw_rmse=result["raw"]["RMSE"], capped_rmse=result["capped"]["RMSE"],
                                nasa_score=result["raw"]["cmapss_score"], engines=count))
    with (output / "test_scores.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(test_scores[0]))
        writer.writeheader()
        writer.writerows(test_scores)
    print(f"{args.subset} test scores: {test_scores}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--subset", required=True, choices=COUNTS)
    parser.add_argument("--phase", choices=("train", "freeze", "test"), default="train")
    parser.add_argument("--member", choices=SETTINGS)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.epochs < 1 or args.threads < 1 or (args.member and args.phase != "train"):
        parser.error("Use positive epochs and threads. --member applies only during training.")
    torch.set_num_threads(args.threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    run(args)


if __name__ == "__main__":
    main()

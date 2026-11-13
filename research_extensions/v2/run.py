"""Train paper-inspired teachers and matched SNNs; test only frozen selections."""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from research_extensions import teacher_student as protocol
from research_extensions.teacher_student_report import report
from research_extensions.v2 import models
from spikedistill.evaluation.metrics import regression_metrics
from spikedistill.models import distillation
from spikedistill.training import baselines, pilot


TEACHERS = ("li_dcnn", "attention_bigru")
GUIDANCE = {"snn_supervised": (), "snn_from_li_dcnn": (0,),
            "snn_from_bigru": (1,), "snn_from_both": (0, 1)}
TEST_COUNTS = dict(zip(protocol.SUBSETS, (100, 259, 100, 248)))


def fit(model, config, directory, training, validation, teachers, device):
    directory.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)
    generator = torch.Generator().manual_seed(config["seed"])
    history = []
    previous, _ = pilot.load_latest(directory)
    if previous is not None:
        if previous["config"] != config:
            raise ValueError(f"Run configuration changed: {directory}")
        model.load_state_dict(previous["model"])
        if previous["epoch"] == config["epochs"]:
            return
        optimizer.load_state_dict(previous["optimizer"])
        history = previous["history"]
    cap = config["cap"]
    dataset = baselines.batches(training, cap, config["batch_size"]).dataset
    if teachers:
        ordered = DataLoader(dataset, batch_size=512)
        predictions = [baselines.predict_cycles(teacher, ordered, device, cap) for teacher in teachers]
        soft = torch.tensor(np.mean(predictions, axis=0) / cap, dtype=torch.float32)
        dataset = TensorDataset(*dataset.tensors, soft)
    if previous is not None:
        pilot.restore_rng(previous["rng"], generator)
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, generator=generator)
    val_loader = baselines.batches(validation, cap, 512)
    for epoch in range(len(history) + 1, config["epochs"] + 1):
        model.train()
        total = 0.0
        for x, mask, y, *soft in loader:
            x, mask, y = x.to(device), mask.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x, mask)
            loss = F.mse_loss(prediction, y)
            if soft:
                loss = 0.6 * loss + 0.4 * F.mse_loss(prediction, soft[0].to(device))
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss: {directory}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            total += loss.item() * len(x)
        pred = baselines.predict_cycles(model, val_loader, device, cap)
        metrics = regression_metrics(validation["y_raw"], pred)
        history.append(dict(epoch=epoch, validation=metrics, loss=total / len(dataset)))
        pilot.save_checkpoint(dict(epoch=epoch, config=config, model=model.state_dict(),
                                   optimizer=optimizer.state_dict(), history=history,
                                   rng=pilot.rng_state(generator)), directory)
        print(f"{config['subset']} {config['model']} seed {config['seed']} "
              f"epoch {epoch}: raw validation RMSE {metrics['RMSE']:.3f}", flush=True)


def prepare_run(args):
    device = torch.device(args.device)
    fingerprints = {Path(module.__file__).name: pilot.digest(module.__file__)
                    for module in (models, protocol, distillation, baselines, pilot)}
    fingerprints["v2/run.py"] = pilot.digest(__file__)
    for subset in args.subsets:
        data = args.data_root / subset
        train, validation, cap = protocol.read_inputs(data)
        source = {name: pilot.digest(data / name) for name in
                  ("train.npz", "validation.npz", "split_manifest.json", "preprocessing.json")}
        window, features = train["X"].shape[1:]
        for seed in args.seeds:
            teachers, selected = [], []
            for name in (*TEACHERS, *GUIDANCE):
                indexes = GUIDANCE.get(name, ())
                is_teacher = name in TEACHERS
                config = dict(model=name, subset=subset, seed=seed, window=window, features=features,
                              cap=cap, epochs=args.teacher_epochs if is_teacher else args.student_epochs,
                              batch_size=512 if name == "li_dcnn" else 256,
                              lr=0.001 if name != "attention_bigru" else 0.0005,
                              device=args.device, cpu_threads=args.cpu_threads,
                              selection="minimum raw validation RMSE", torch_version=str(torch.__version__),
                              training_target="capped RUL divided by cap", loss="MSE",
                              teacher_fraction=0.4 if indexes else 0.0,
                              source_sha256=source, code_sha256=fingerprints,
                              teachers={TEACHERS[i]: selected[i]["checkpoint_sha256"] for i in indexes})
                directory = args.output / subset / f"seed_{seed}" / name
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                model = models.build(name, features, window).to(device)
                if args.phase == "train":
                    fit(model, config, directory, train, validation, [teachers[i] for i in indexes], device)
                model, epoch, sha = protocol.selected(model, config, directory, device)
                if is_teacher:
                    teachers.append(model)
                selected.append(dict(config=config, selected_epoch=epoch, checkpoint_sha256=sha))
                yield data, directory, selected[-1], model
            for teacher in teachers:
                teacher.cpu()


def evaluate(directory, entry, model, data, split, device):
    config = entry["config"]
    cap = config["cap"]
    loader = baselines.batches(data, cap, 512)
    prediction = baselines.predict_cycles(model.to(device), loader, device, cap)
    result = baselines.save_result(directory, config, split, data, prediction, entry["selected_epoch"],
                                   sum(p.numel() for p in model.parameters()))
    print(f"{config['subset']} {config['model']} seed {config['seed']}: "
          f"raw {split} RMSE {result['raw']['RMSE']:.3f}", flush=True)


def run(args):
    device = torch.device(args.device)
    index = args.output / "frozen_checkpoints.json"
    completed = []
    for data, directory, entry, model in prepare_run(args):
        if args.phase == "train":
            evaluate(directory, entry, model, baselines.read_split(data, "validation"), "validation", device)
        completed.append((data, directory, entry, model))
        if entry["config"]["model"] not in TEACHERS:
            model.cpu()
    frozen = [entry for _, _, entry, _ in completed]
    if args.phase == "train":
        baselines.write_json(index, frozen)
        return
    if json.loads(index.read_text(encoding="utf-8")) != frozen:
        raise ValueError("Checkpoints or settings differ from the frozen validation manifest")
    last = None
    for path, directory, entry, model in completed:
        if path != last:
            test = baselines.read_split(path, "test")
            expected = TEST_COUNTS[entry["config"]["subset"]]
            if (len(test["engine_id"]) != expected or not test["mask"][:, -1].all()
                    or not np.array_equal(np.sort(test["engine_id"]), np.arange(1, expected + 1))):
                raise ValueError(f"Expected one final window for every official test engine: {path}")
            last = path
        evaluate(directory, entry, model, test, "test", device)
        model.cpu()
    report(args.output, args.subsets, args.seeds, args.originals,
           display_tables=False, teachers=TEACHERS, guidance=GUIDANCE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--originals", type=Path, default=Path("results/test_metrics_per_seed.csv"))
    parser.add_argument("--subsets", nargs="+", choices=protocol.SUBSETS, default=list(protocol.SUBSETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--teacher-epochs", type=int, default=50)
    parser.add_argument("--student-epochs", type=int, default=30)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--phase", choices=("train", "test"), default="train")
    args = parser.parse_args()
    if min(args.teacher_epochs, args.student_epochs, args.cpu_threads) < 1:
        parser.error("Epoch and thread counts must be positive")
    if len(set(args.subsets)) != len(args.subsets) or len(set(args.seeds)) != len(args.seeds):
        parser.error("Subsets and seeds must be unique")
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    run(args)


if __name__ == "__main__":
    main()

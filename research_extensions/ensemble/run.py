"""Fit ensemble weights on validation predictions, then score frozen test predictions."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from spikedistill.evaluation.metrics import regression_metrics
from spikedistill.training.baselines import save_result, write_json
from spikedistill.training.pilot import digest


def load_member(folder, split):
    report = json.loads((folder / f"{split}_metrics.json").read_text())
    config = report["config"]
    if report["split"] != split or report["prediction_units"] != "cycles":
        raise ValueError(f"Wrong split or units in {folder}")
    path = folder / f"{split}_predictions.csv"
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = sorted(csv.DictReader(stream), key=lambda row: int(row["engine_id"]))
    data = dict(engine_id=np.array([int(r["engine_id"]) for r in rows]),
                end_cycle=np.array([int(r["end_cycle"]) for r in rows]),
                y=np.array([float(r["target_capped_cycles"]) for r in rows]),
                y_raw=np.array([float(r["target_raw_cycles"]) for r in rows]))
    prediction = np.array([float(r["prediction_cycles"]) for r in rows])
    if len(np.unique(data["engine_id"])) != len(rows) or not np.isfinite(prediction).all():
        raise ValueError(f"Duplicate engines or invalid predictions in {folder}")
    if not np.array_equal(data["y"], np.minimum(data["y_raw"], config["cap"])):
        raise ValueError(f"Targets disagree with cap in {folder}")
    for label, target in (("raw", data["y_raw"]), ("capped", data["y"])):
        scores = regression_metrics(target, prediction)
        for name, value in scores.items():
            expected = report[label][name]
            if value != expected and (value is None or expected is None
                                       or not np.isclose(value, expected, rtol=1e-7, atol=1e-7)):
                raise ValueError(f"Saved {label} {name} differs from predictions in {folder}")
    hashes = config.get("source_sha256", {})
    identity = dict(subset=config["subset"], cap=config["cap"],
                    train=hashes.get("train.npz", config.get("train_sha256")),
                    validation=hashes.get("validation.npz", config.get("validation_sha256")))
    if not identity["train"] or not identity["validation"]:
        raise ValueError(f"Missing data provenance in {folder}")
    return report, data, prediction, identity


def aligned(members, split):
    loaded = [load_member(folder, split) for folder in members]
    _, reference, _, identity = loaded[0]
    for _, data, _, current in loaded[1:]:
        if current != identity or any(not np.array_equal(data[key], value) for key, value in reference.items()):
            raise ValueError("Member data fingerprints, targets or endpoints differ")
    return loaded


def make_plan(members):
    loaded = aligned(members, "validation")
    errors = np.array([r["capped"]["MSE"] for r, _, _, _ in loaded])
    inverse = 1 / np.maximum(errors, 1e-12)
    identity = loaded[0][-1]
    entries = []
    for folder, (report, _, _, _) in zip(members, loaded):
        entries.append(dict(model=report["config"]["model"], seed=report["config"]["seed"],
                            selected_epoch=report["selected_epoch"], parameters=report["parameters"],
                            validation_predictions_sha256=digest(folder / "validation_predictions.csv"),
                            validation_report_sha256=digest(folder / "validation_metrics.json"),
                            config=report["config"]))
    identifiers = [(e["model"], e["seed"]) for e in entries]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Members must have different model/seed identities")
    return dict(identity=identity, members=entries, selected_by="capped validation MSE",
                equal_weights=[1 / len(members)] * len(members),
                inverse_validation_mse_weights=(inverse/inverse.sum()).tolist(),
                validation_best_index=int(errors.argmin()))


def run(args):
    members = [path.resolve() for path in args.members]
    if len(set(members)) != len(members) or any(path == args.output.resolve() for path in members):
        raise ValueError("Use distinct member folders and a separate output folder")
    plan = make_plan(members)
    plan_path = args.output / "ensemble_plan.json"
    if args.phase == "fit":
        if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
            raise ValueError("Existing ensemble has different members or validation files")
        write_json(plan_path, plan)
        split = "validation"
    else:
        if not plan_path.exists() or json.loads(plan_path.read_text()) != plan:
            raise ValueError("Run the matching validation-only fit first")
        split = "test"
    loaded = aligned(members, split)
    for entry, (report, _, _, identity) in zip(plan["members"], loaded):
        if (report["config"] != entry["config"] or report["selected_epoch"] != entry["selected_epoch"]
                or report["parameters"] != entry["parameters"] or identity != plan["identity"]):
            raise ValueError("Test report does not match the frozen validation member")
    data = loaded[0][1]
    predictions = np.stack([row[2] for row in loaded])
    best = plan["validation_best_index"]
    approaches = {
        "equal_ensemble": np.array(plan["equal_weights"]),
        "validation_weighted_ensemble": np.array(plan["inverse_validation_mse_weights"]),
        "validation_best_single": np.eye(len(members))[best],
    }
    summary = []
    for name, weights in approaches.items():
        config = dict(model=name, subset=plan["identity"]["subset"], cap=plan["identity"]["cap"],
                      members=plan["members"], weights=weights.tolist(),
                      selection="validation only", ensemble_plan_sha256=digest(plan_path))
        count = sum(entry["parameters"] for entry, weight in zip(plan["members"], weights) if weight > 0)
        report = save_result(args.output / name, config, split, data,
                             weights @ predictions, None, count)
        summary.append(report)
        print(f"{split} {name}: raw RMSE {report['raw']['RMSE']:.4f}, "
              f"NASA score {report['raw']['cmapss_score']:.2f}", flush=True)
    write_json(args.output / f"{split}_summary.json", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--members", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("fit", "test"), default="fit")
    args = parser.parse_args()
    if len(args.members) < 2:
        parser.error("Supply at least two member folders")
    run(args)

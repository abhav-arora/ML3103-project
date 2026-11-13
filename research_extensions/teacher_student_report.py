"""Report paired RUL metrics using aligned per-engine predictions."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from research_extensions.teacher_student import GUIDANCE, SUBSETS, TEACHERS
from spikedistill.evaluation.metrics import regression_metrics


ENDPOINTS = ["engine_id", "end_cycle", "target_capped_cycles", "target_raw_cycles"]
METRICS = ["RMSE", "MAE", "MSE", "RAE_pct", "R2", "cmapss_score", "bias_cycles"]


def tables(root, subsets, seeds, originals, teachers=TEACHERS, guidance=GUIDANCE):
    rows, pairs = [], []
    original = pd.read_csv(originals)
    original = original[original.target.eq("raw")]
    for subset in subsets:
        for seed in seeds:
            reference = original[(original.subset == subset) & (original.seed == seed)]
            count, = reference.engines.unique()
            for split in ("validation", "test"):
                predictions = {}
                expected = None
                for name in (*teachers, *guidance):
                    path = root / subset / f"seed_{seed}" / name / f"{split}_predictions.csv"
                    frame = pd.read_csv(path).sort_values("engine_id").reset_index(drop=True)
                    if frame.engine_id.duplicated().any() or (split == "test" and len(frame) != count):
                        raise ValueError(f"Unexpected engine coverage in {path}")
                    if expected is not None:
                        pd.testing.assert_frame_equal(frame[ENDPOINTS], expected)
                    expected = frame[ENDPOINTS]
                    predictions[name] = frame.prediction_cycles.to_numpy()
                predictions["teacher_average"] = (predictions[teachers[0]] + predictions[teachers[1]]) / 2
                for target in ("raw", "capped"):
                    y = expected[f"target_{target}_cycles"].to_numpy()
                    scores = {}
                    for name, prediction in predictions.items():
                        metrics = regression_metrics(y, prediction)
                        denominator = np.abs(y - y.mean()).sum()
                        metrics["RAE_pct"] = 100 * np.abs(prediction - y).sum() / denominator if denominator else None
                        metrics["bias_cycles"] = np.mean(prediction - y)
                        scores[name] = metrics["RMSE"]
                        rows.append(dict(subset=subset, seed=seed, split=split, target=target,
                                         model=name, **metrics))
                    for teacher, student in zip((*teachers, "teacher_average"), tuple(guidance)[1:]):
                        pairs.append(dict(subset=subset, seed=seed, split=split, target=target,
                                          teacher=teacher, student=student, teacher_RMSE=scores[teacher],
                                          student_RMSE=scores[student],
                                          student_minus_teacher=scores[student] - scores[teacher],
                                          student_minus_supervised=scores[student] - scores["snn_supervised"]))
    old = original[original.subset.isin(subsets) & original.seed.isin(seeds)].copy()
    return pd.DataFrame(rows), pd.DataFrame(pairs), old


def report(root, subsets, seeds, originals, display_tables=True, teachers=TEACHERS, guidance=GUIDANCE):
    scores, pairs, old = tables(root, subsets, seeds, originals, teachers, guidance)
    for frame, name in ((scores, "research_metrics"), (pairs, "teacher_student_pairs"), (old, "original_raw_reference")):
        frame.to_csv(root / f"{name}.csv", index=False)
    if display_tables:
        from IPython.display import display

        print("Official raw test RMSE for each teacher–student pair")
        display(pairs[(pairs.split == "test") & (pairs.target == "raw")].round(3))
        print("Official raw test metrics for every new model")
        display(scores.loc[(scores.split == "test") & (scores.target == "raw"),
                           ["subset", "seed", "model", *METRICS]].round(3))
        print("Original seed-matched raw test reference")
        display(old.pivot(index=["subset", "seed"], columns="model", values="RMSE").round(3))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--originals", type=Path, default=Path("results/test_metrics_per_seed.csv"))
    parser.add_argument("--subsets", nargs="+", choices=SUBSETS, default=list(SUBSETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    args = parser.parse_args()
    report(args.root, args.subsets, args.seeds, args.originals, display_tables=False)

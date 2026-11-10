"""Reproducible C-MAPSS preparation. No model training or test-set selection."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ['engine_id', 'cycle'] + [f'op_{i}' for i in range(1, 4)] + [f'sensor_{i:02d}' for i in range(1, 22)]
SENSORS = COLUMNS[5:]
SETTINGS = COLUMNS[2:5]


def load_trajectories(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep=r'\s+', header=None)
    if frame.shape[1] != 26:
        raise ValueError(f'{path}: expected 26 columns, got {frame.shape[1]}')
    frame.columns = COLUMNS
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError(f'{path}: missing or non-finite values')
    for col in ['engine_id', 'cycle']:
        if not np.equal(frame[col], np.floor(frame[col])).all() or (frame[col] < 1).any():
            raise ValueError(f'{path}: invalid {col}')
        frame[col] = frame[col].astype(int)
    if frame.duplicated(['engine_id', 'cycle']).any():
        raise ValueError(f'{path}: duplicate engine/cycle pairs')
    for engine, rows in frame.groupby('engine_id', sort=True):
        if not np.array_equal(rows.cycle.to_numpy(), np.arange(1, len(rows) + 1)):
            raise ValueError(f'{path}: unordered or nonconsecutive cycles for engine {engine}')
    return frame


def split_engines(frame, seed=42, validation_fraction=0.2):
    ids = np.sort(frame.engine_id.unique())
    if len(ids) < 2 or not 0 < validation_fraction < 1:
        raise ValueError('Need multiple engines and a validation fraction in (0, 1)')
    n_val = min(len(ids) - 1, max(1, int(round(len(ids) * validation_fraction))))
    shuffled = np.random.default_rng(seed).permutation(ids)
    val_ids, train_ids = np.sort(shuffled[:n_val]), np.sort(shuffled[n_val:])
    return train_ids, val_ids


def assign_regimes(settings, centers):
    return ((settings[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)


def fit_regimes(settings, k, seed):
    """Small NumPy k-means++ implementation; fitted on training engines only."""
    rng = np.random.default_rng(seed)
    centers = [settings[rng.integers(len(settings))].copy()]
    while len(centers) < k:
        d2 = ((settings[:, None] - np.asarray(centers)[None]) ** 2).sum(2).min(1)
        if d2.sum() <= 1e-12:
            raise ValueError('Insufficient distinct operating conditions')
        centers.append(settings[rng.choice(len(settings), p=d2 / d2.sum())].copy())
    centers = np.asarray(centers)
    for _ in range(100):
        labels = assign_regimes(settings, centers)
        updated = centers.copy()
        for j in range(k):
            if np.any(labels == j):
                updated[j] = settings[labels == j].mean(0)
        if np.allclose(updated, centers, atol=1e-7, rtol=0):
            centers = updated
            break
        centers = updated
    return centers


def fit_preprocessor(train, subset, seed=42):
    sensor_std = train[SENSORS].std(ddof=0)
    sensors = sensor_std.index[sensor_std > 1e-8].tolist()
    if not sensors:
        raise ValueError('All sensors are constant')
    op_mean = train[SETTINGS].mean().to_numpy()
    op_scale = train[SETTINGS].std(ddof=0).to_numpy(copy=True)
    op_scale[op_scale < 1e-8] = 1.0
    op = (train[SETTINGS].to_numpy() - op_mean) / op_scale
    k = 6 if subset in ('FD002', 'FD004') else 1
    centers = fit_regimes(op, k, seed) if k > 1 else np.zeros((1, 3))
    labels = assign_regimes(op, centers)
    sensor_values = train[sensors].to_numpy()
    means, scales, counts = [], [], []
    for j in range(k):
        values = sensor_values[labels == j]
        if len(values) == 0:
            raise ValueError('Empty operating-condition cluster')
        means.append(values.mean(0))
        scale = values.std(0)
        scale[scale < 1e-8] = 1.0
        scales.append(scale)
        counts.append(len(values))
    return dict(sensors=sensors, dropped_sensors=[s for s in SENSORS if s not in sensors],
                feature_names=sensors + SETTINGS, op_mean=op_mean.tolist(), op_scale=op_scale.tolist(),
                regime_centers=centers.tolist(), sensor_means=np.asarray(means).tolist(),
                sensor_scales=np.asarray(scales).tolist(), regime_training_rows=counts,
                normalization='condition-aware' if k > 1 else 'global-training-only')


def transform(frame, prep):
    op = (frame[SETTINGS].to_numpy() - prep['op_mean']) / prep['op_scale']
    regimes = assign_regimes(op, np.asarray(prep['regime_centers']))
    sensors = (frame[prep['sensors']].to_numpy() - np.asarray(prep['sensor_means'])[regimes]) / np.asarray(prep['sensor_scales'])[regimes]
    values = np.concatenate([sensors, op], axis=1).astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError('Non-finite normalized features')
    return values


def make_windows(frame, prep, window, cap, mode, endpoints=None, final_rul=None):
    """Full training windows; one masked, left-padded endpoint window for val/test.

    y_raw and y are explicitly separate. Padded rows repeat the earliest observed
    row, but are marked False in mask. Future training must respect that mask.
    """
    if window < 1 or cap <= 0 or mode not in ('train', 'validation', 'test'):
        raise ValueError('Invalid window configuration')
    xs, masks, ys, raw_ys, engines, cycles, lengths = [], [], [], [], [], [], []
    for engine, rows in frame.groupby('engine_id', sort=True):
        features = transform(rows, prep)
        total = len(rows)
        if mode == 'train':
            ends = range(window, total + 1)
        else:
            ends = [int(endpoints[int(engine)]) if endpoints else total]
        for end in ends:
            if not 1 <= end <= total:
                raise ValueError('Endpoint outside trajectory')
            start = max(0, end - window)
            observed = features[start:end]
            valid = len(observed)
            padded = np.concatenate([np.repeat(observed[:1], window-valid, axis=0), observed])
            raw = float(final_rul[int(engine)]) if mode == 'test' else float(total - end)
            xs.append(padded)
            masks.append(np.arange(window) >= window - valid)
            raw_ys.append(raw)
            ys.append(min(cap, raw))
            engines.append(int(engine))
            cycles.append(end)
            lengths.append(valid)
    if not xs:
        raise ValueError('No windows; reduce window length or check trajectories')
    return dict(X=np.stack(xs), mask=np.asarray(masks, dtype=bool),
                y=np.asarray(ys, dtype=np.float32), y_raw=np.asarray(raw_ys, dtype=np.float32),
                engine_id=np.asarray(engines, dtype=np.int32),
                end_cycle=np.asarray(cycles, dtype=np.int32),
                valid_length=np.asarray(lengths, dtype=np.int32))


def validation_endpoints(frame, seed):
    # Fixed incomplete prefixes approximate test usage without selecting from test.
    rng = np.random.default_rng(seed + 1000)
    endpoints = {}
    for engine, rows in frame.groupby('engine_id', sort=True):
        if len(rows) < 2:
            raise ValueError('Validation engine must have at least two cycles')
        endpoints[int(engine)] = max(1, min(len(rows)-1, int(np.floor(len(rows)*rng.uniform(.4, .9)))))
    return endpoints


def audit_frame(frame):
    lengths = frame.groupby('engine_id').size()
    return dict(rows=len(frame), engines=len(lengths), columns=frame.shape[1],
                cycles_min=int(lengths.min()), cycles_median=float(lengths.median()),
                cycles_max=int(lengths.max()), missing_values=int(frame.isna().sum().sum()))


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')


def prepare_subset(data_dir, output_dir, subset, window=30, cap=125., seed=42):
    files = {name: data_dir / f'{name}_{subset}.txt' for name in ['train', 'test', 'RUL']}
    train = load_trajectories(files['train'])
    test = load_trajectories(files['test'])
    rul = np.loadtxt(files['RUL'], ndmin=1)
    test_ids = np.sort(test.engine_id.unique())
    if not np.array_equal(test_ids, np.arange(1, len(test_ids)+1)):
        raise ValueError('Test IDs must be 1..N to map the official RUL file safely')
    if rul.ndim != 1 or len(rul) != len(test_ids) or not np.isfinite(rul).all() or (rul < 0).any():
        raise ValueError('Invalid test RUL labels')
    train_ids, val_ids = split_engines(train, seed)
    train_part = train[train.engine_id.isin(train_ids)]
    val_part = train[train.engine_id.isin(val_ids)]
    prep = fit_preprocessor(train_part, subset, seed)
    endpoints = validation_endpoints(val_part, seed)
    target = output_dir / subset
    target.mkdir(parents=True, exist_ok=True)
    summary = dict(subset=subset, official_train=audit_frame(train), official_test=audit_frame(test),
                   training_engines=len(train_ids), validation_engines=len(val_ids),
                   features=len(prep['feature_names']), dropped_sensors=prep['dropped_sensors'],
                   normalization=prep['normalization'], arrays={})
    for mode, frame in [('train', train_part), ('validation', val_part), ('test', test)]:
        arrays = make_windows(frame, prep, window, cap, mode,
                              endpoints=endpoints if mode == 'validation' else None,
                              final_rul=dict(zip(test_ids, rul)) if mode == 'test' else None)
        assert arrays['X'].shape[:2] == arrays['mask'].shape
        assert np.all(arrays['y'] >= 0) and np.all(arrays['y'] <= cap)
        assert np.array_equal(arrays['y'], np.minimum(arrays['y_raw'], cap))
        expected = train_ids if mode == 'train' else val_ids if mode == 'validation' else test_ids
        assert np.array_equal(np.unique(arrays['engine_id']), expected)
        np.savez_compressed(target / f'{mode}.npz', **arrays)
        summary['arrays'][mode] = dict(shape=list(arrays['X'].shape), padded_windows=int((~arrays['mask'].all(1)).sum()))
    write_json(target / 'preprocessing.json', prep)
    write_json(target / 'split_manifest.json', dict(seed=seed, window=window, rul_cap=cap,
               train_engine_ids=train_ids.tolist(), validation_engine_ids=val_ids.tolist(),
               validation_endpoints=endpoints, validation_protocol='one prefix per held-out engine; fraction U(0.4,0.9), seed+1000',
               test_protocol='one final observed window per official test engine',
               target_protocol='y capped; y_raw uncapped; both retained; evaluation must name protocol',
               sources={name: dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for name, path in files.items()}))
    write_json(target / 'audit.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('artifacts/data'))
    parser.add_argument('--subsets', nargs='+', choices=['FD001','FD002','FD003','FD004'], default=['FD001','FD002','FD003','FD004'])
    parser.add_argument('--window', type=int, default=30)
    parser.add_argument('--cap', type=float, default=125.)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for subset in args.subsets:
        summary = prepare_subset(args.data_dir, args.output_dir, subset, args.window, args.cap, args.seed)
        summaries.append(summary)
        print(f"{subset}: {summary['training_engines']} train engines / {summary['validation_engines']} validation engines; {summary['features']} features; {summary['arrays']}", flush=True)
    write_json(args.output_dir / 'audit_summary.json', summaries)


if __name__ == '__main__':
    main()

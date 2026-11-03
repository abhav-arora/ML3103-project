"""Small FD001 compatibility/recovery trial, not final research results.

No test archive is opened. Resume is automatic from the latest complete checkpoint.
Run one process per run directory; concurrent writers are not supported.
"""
import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from metrics import regression_metrics


class PilotMLP(nn.Module):
    def __init__(self, window, features):
        super().__init__()
        self.layers = nn.Sequential(nn.Flatten(), nn.Linear(window*features, 64),
                                    nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 32),
                                    nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, x, mask):
        # Feed-forward model: zero masked padding before flattening.
        return self.layers(x * mask.unsqueeze(-1)).squeeze(-1)


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_checkpoint(payload, directory):
    """Never overwrite the last complete checkpoint; manifest moves only after readback.

    Immutable epoch files make a partial cloud upload recoverable. Drive is remote:
    successful round-trip checks reduce risk but do not guarantee upload durability.
    """
    directory = Path(directory)
    epoch = payload['epoch']
    path = directory / f'epoch_{epoch:04d}.pt'
    temporary = directory / f'epoch_{epoch:04d}.partial'
    torch.save(payload, temporary)
    check = torch.load(temporary, map_location='cpu', weights_only=True)
    if check['epoch'] != epoch:
        raise RuntimeError('Checkpoint write/read mismatch')
    os.replace(temporary, path)
    index = dict(file=path.name, sha256=digest(path), epoch=epoch)
    index_temp = directory / 'latest.partial.json'
    index_temp.write_text(json.dumps(index, indent=2), encoding='utf-8')
    os.replace(index_temp, directory / 'latest.json')
    return path


def load_latest(directory):
    directory = Path(directory)
    # Try indexed latest first; complete immutable checkpoints remain as fallback.
    paths = []
    index_path = directory / 'latest.json'
    if index_path.exists():
        try:
            index = json.loads(index_path.read_text())
            path = directory / index['file']
            if path.parent != directory or not path.name.startswith('epoch_'):
                raise ValueError('Invalid checkpoint index')
            if digest(path) == index['sha256']:
                paths.append(path)
        except (OSError, ValueError, KeyError):
            pass
    paths += sorted(directory.glob('epoch_*.pt'), reverse=True)
    seen = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        try:
            return torch.load(path, map_location='cpu', weights_only=True), path
        except Exception as error:
            print(f'Cannot read {path.name}: {error}; trying previous complete file', flush=True)
    if paths:
        raise RuntimeError('No readable checkpoint; refusing to silently restart this run')
    return None, None


def rng_state(generator):
    state = np.random.get_state()
    return dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                python=random.getstate(), numpy=[state[0], state[1].tolist(), state[2], state[3], state[4]],
                loader=generator.get_state())


def restore_rng(state, generator):
    torch.set_rng_state(state['torch'])
    generator.set_state(state['loader'])
    random.setstate(state['python'])
    a = state['numpy']
    np.random.set_state((a[0], np.asarray(a[1], dtype=np.uint32), a[2], a[3], a[4]))
    if torch.cuda.is_available() and len(state['cuda']) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all(state['cuda'])


def train(args):
    args.run_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    requested = getattr(args, 'device', 'auto')
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if requested == 'auto' else requested)
    arrays = {}
    for split in ['train', 'validation']:
        with np.load(args.data_dir / f'{split}.npz', allow_pickle=False) as archive:
            arrays[split] = {key: archive[key].copy() for key in ['X', 'mask', 'y']}
    manifest = json.loads((args.data_dir / 'split_manifest.json').read_text())
    cap = float(manifest['rul_cap'])
    x, mask, y = (arrays['train'][key] for key in ['X', 'mask', 'y'])
    selection = np.random.default_rng(args.seed).choice(len(x), min(args.max_examples, len(x)), replace=False)
    x, mask, y = x[selection], mask[selection], y[selection]
    config = dict(seed=args.seed, max_examples=args.max_examples, batch_size=args.batch_size,
                  lr=1e-3, architecture='PilotMLP-64-32-dropout0.1-sigmoid',
                  window=x.shape[1], features=x.shape[2], cap=cap, target='capped',
                  sources={name: digest(args.data_dir/name) for name in ['train.npz', 'validation.npz', 'preprocessing.json', 'split_manifest.json']})
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(mask), torch.from_numpy(y/cap)),
                        batch_size=args.batch_size, shuffle=True, generator=generator, num_workers=0)
    val = arrays['validation']
    vx = torch.from_numpy(val['X']).to(device)
    vm = torch.from_numpy(val['mask']).to(device)
    model = PilotMLP(x.shape[1], x.shape[2]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    checkpoint, path = load_latest(args.run_dir)
    start, history, origin = 0, [], args.runtime_id
    if checkpoint is not None:
        if checkpoint['config'] != config:
            raise ValueError('Run configuration/data changed: use a different run directory')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.eval()
        with torch.no_grad():
            restored = model(vx[:8], vm[:8]).cpu()
        torch.testing.assert_close(restored, checkpoint['probe_prediction'], rtol=1e-4, atol=1e-5)
        # Verify Adam state was restored before taking another optimizer step.
        for key, values in checkpoint['optimizer']['state'].items():
            for name, value in values.items():
                actual = optimizer.state_dict()['state'][key][name]
                if isinstance(value, torch.Tensor):
                    torch.testing.assert_close(actual.cpu(), value.cpu(), rtol=0, atol=0)
        restore_rng(checkpoint['rng'], generator)
        start, history, origin = checkpoint['epoch'], checkpoint['history'], checkpoint['origin_runtime_id']
        print(f'Restored {path.name}: epoch={start}; prediction and optimizer checks passed', flush=True)
    for epoch in range(start+1, args.epochs+1):
        sync(device)
        began = time.perf_counter()
        model.train()
        loss_sum = 0.
        for bx, bm, by in loader:
            bx, bm, by = bx.to(device), bm.to(device), by.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(bx, bm), by)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training loss')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(bx)
        model.eval()
        with torch.no_grad():
            prediction = model(vx, vm).cpu().numpy() * cap
            probe = model(vx[:8], vm[:8]).cpu()
        sync(device)
        row = dict(epoch=epoch, train_normalized_mse=loss_sum/len(x),
                   validation=regression_metrics(val['y'], prediction), seconds=time.perf_counter()-began)
        history.append(row)
        payload = dict(epoch=epoch, model=model.state_dict(), optimizer=optimizer.state_dict(),
                       config=config, history=history, rng=rng_state(generator),
                       probe_prediction=probe, origin_runtime_id=origin,
                       last_runtime_id=args.runtime_id, device=str(device), torch_version=str(torch.__version__))
        saved = save_checkpoint(payload, args.run_dir)
        print(json.dumps(row), flush=True)
        print(f'Checkpoint verified: {saved}', flush=True)
    result = dict(completed_epoch=max(start, args.epochs), history=history, device=str(device),
                  torch_version=str(torch.__version__), parameters=sum(p.numel() for p in model.parameters()),
                  purpose='compatibility pilot, not final benchmark', target='capped',
                  origin_runtime_id=origin, current_runtime_id=args.runtime_id)
    (args.run_dir/'progress.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--max-examples', type=int, default=2048)
    parser.add_argument('--runtime-id', default='local')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.max_examples) < 1:
        parser.error('Epochs, batch size and max examples must be positive')
    train(args)


if __name__ == '__main__':
    main()

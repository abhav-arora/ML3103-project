"""Run in the Colab runtime: python -m unittest -v test_training.

Checks save/resume equivalence using tiny synthetic CPU data, never official test data.
"""
import argparse
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from train_pilot import train, load_latest


class RecoveryTests(unittest.TestCase):
    def test_restart_restores_next_epoch_and_corruption_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root/'data'
            data.mkdir()
            rng = np.random.default_rng(7)
            for split, n in [('train',24), ('validation',6)]:
                x = rng.normal(size=(n,3,2)).astype(np.float32)
                np.savez_compressed(data/f'{split}.npz', X=x, mask=np.ones((n,3),dtype=bool),
                                    y=(50+10*x[:,0,0]).astype(np.float32))
            (data/'preprocessing.json').write_text('{}')
            (data/'split_manifest.json').write_text(json.dumps({'rul_cap':125}))

            def args(name, epochs):
                return argparse.Namespace(data_dir=data, run_dir=root/name, epochs=epochs,
                                          seed=42,batch_size=8,max_examples=24,
                                          runtime_id=f'session-{epochs}',device='cpu')
            train(args('continuous',3))
            train(args('resumed',2))
            train(args('resumed',3))
            continuous,_ = load_latest(root/'continuous')
            resumed,_ = load_latest(root/'resumed')
            self.assertEqual(resumed['epoch'],3)
            for key in continuous['model']:
                torch.testing.assert_close(continuous['model'][key],resumed['model'][key],rtol=0,atol=0)
            # If the last complete file is corrupted, epoch 2 must remain recoverable.
            (root/'resumed'/'epoch_0003.pt').write_bytes(b'interrupted upload')
            previous,_ = load_latest(root/'resumed')
            self.assertEqual(previous['epoch'],2)


if __name__ == '__main__':
    unittest.main()

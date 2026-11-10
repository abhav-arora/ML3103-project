import unittest

import numpy as np
import torch
from torch import nn

from spikedistill.models.baselines import MODELS, ConvertedSNN, calibrate_conversion, make_model
from spikedistill.training.baselines import batches, predict_cycles


class BaselineTests(unittest.TestCase):
    def test_models_ignore_padding_and_train(self):
        torch.manual_seed(42)
        x = torch.randn(4, 6, 3)
        mask = torch.tensor([[False, False, True, True, True, True]] * 4)
        changed = x.clone()
        changed[~mask] = 1000
        for name in MODELS:
            if name in {"mean", "ann_to_snn"}:
                continue
            with self.subTest(model=name):
                model = make_model(name, 6, 3)
                prediction = model(x, mask)
                self.assertEqual(prediction.shape, (4,))
                torch.testing.assert_close(prediction, model(changed, mask))
                prediction.square().mean().backward()
                self.assertTrue(all(p.grad is not None and p.grad.isfinite().all() for p in model.parameters()))

    def test_inverse_scaling_uses_manifest_cap_without_clipping(self):
        class Fixed(nn.Module):
            def forward(self, x, mask):
                return x[:, 0, 0]

        data = dict(X=np.array([[[0.5]], [[1.2]], [[-0.1]]], dtype=np.float32),
                    mask=np.ones((3, 1), dtype=bool), y=np.array([40, 80, 0], dtype=np.float32))
        loader = batches(data, 80, 2)
        np.testing.assert_allclose(predict_cycles(Fixed(), loader, torch.device("cpu"), 80),
                                   [40, 96, -8], atol=1e-5)

    def test_conversion_preserves_constant_readout_and_masks(self):
        model = make_model("deepnn", 2, 1)
        with torch.no_grad():
            model.readout.weight.zero_()
            model.readout.bias.fill_(0.3)
        data = dict(X=np.ones((2, 2, 1), dtype=np.float32), mask=np.ones((2, 2), dtype=bool),
                    y=np.ones(2, dtype=np.float32))
        loader = batches(data, 125, 2)
        thresholds = calibrate_conversion(model, loader, torch.device("cpu"))
        converted = ConvertedSNN(model, thresholds, 4)
        np.testing.assert_allclose(predict_cycles(converted, loader, torch.device("cpu"), 125), [37.5, 37.5], atol=1e-5)

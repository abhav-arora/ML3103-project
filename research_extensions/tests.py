"""Checks for validation-only ensembles and variable-width students."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch
from torch import nn

from research_extensions.ensemble.run import make_plan
from spikedistill.models.distillation import Student
from spikedistill.training.baselines import save_result


class ExtensionTests(unittest.TestCase):
    def test_ensemble_plan_uses_validation_files_only(self):
        data = dict(engine_id=np.array([1, 2]), end_cycle=np.array([30, 40]),
                    y=np.array([10., 20.]), y_raw=np.array([10., 20.]))
        with TemporaryDirectory() as temporary:
            members = []
            for seed, error in ((42, 1.), (43, 2.)):
                directory = Path(temporary) / str(seed)
                config = dict(model="spikedistill", seed=seed, subset="FD001", cap=125,
                              train_sha256="train", validation_sha256="validation")
                save_result(directory, config, "validation", data, data["y"] + error, 1, 3409)
                # Test data are deliberately unreadable. Weight fitting must not inspect them.
                (directory / "test_predictions.csv").write_text("invalid test input")
                members.append(directory)
            plan = make_plan(members)
            np.testing.assert_allclose(plan["inverse_validation_mse_weights"], [0.8, 0.2])
            self.assertEqual(plan["validation_best_index"], 0)
            self.assertEqual(plan["equal_weights"], [0.5, 0.5])

    def test_ensemble_rejects_different_validation_endpoints(self):
        data = dict(engine_id=np.array([1, 2]), end_cycle=np.array([30, 40]),
                    y=np.array([10., 20.]), y_raw=np.array([10., 20.]))
        with TemporaryDirectory() as temporary:
            members = []
            for seed in (42, 43):
                directory = Path(temporary) / str(seed)
                config = dict(model="spikedistill", seed=seed, subset="FD001", cap=125,
                              train_sha256="train", validation_sha256="validation")
                current = dict(data, end_cycle=data["end_cycle"] + (seed - 42))
                save_result(directory, config, "validation", current, data["y"] + 1, 1, 3409)
                members.append(directory)
            with self.assertRaisesRegex(ValueError, "endpoints differ"):
                make_plan(members)

    def test_student_width_preserves_masks_and_trains_projection(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)
        x = torch.randn(3, 5, 18)
        mask = torch.tensor([[False, False, True, True, True]] * 3)
        changed = x.clone()
        changed[~mask] = 500
        for width in (16, 32, 48, 64):
            student = Student(18, width)
            self.assertEqual(sum(p.numel() for p in student.parameters()), width**2 + 23*width + 1)
            torch.testing.assert_close(student(x, mask), student(changed, mask))
            pred, membrane, attention, spikes = student.signals(x, mask)
            projection = nn.Linear(width, 96)
            loss = pred.square().mean() + projection(membrane[mask]).square().mean()
            loss.backward()
            self.assertEqual(spikes.shape, (3, 5, width))
            self.assertTrue(torch.equal(attention[~mask], torch.zeros_like(attention[~mask])))
            self.assertTrue(all(p.grad is not None and p.grad.isfinite().all()
                                for p in [*student.parameters(), *projection.parameters()]))


if __name__ == "__main__":
    unittest.main()

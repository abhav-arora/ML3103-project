"""Checks for leakage-sensitive preprocessing and temporal target correctness."""
import unittest
import numpy as np
import pandas as pd
from prepare_data import COLUMNS, load_trajectories, split_engines, fit_preprocessor, transform, make_windows, validation_endpoints
from io import StringIO


def fixture():
    rows = []
    for engine in range(1, 6):
        for cycle in range(1, 9):
            rows.append([engine, cycle, 0., 0., 0.] + [engine*10. + cycle + j for j in range(21)])
    return pd.DataFrame(rows, columns=COLUMNS)


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.frame = fixture()
        self.prep = fit_preprocessor(self.frame[self.frame.engine_id < 4], 'FD001')

    def test_engine_split_disjoint_complete_reproducible(self):
        train, val = split_engines(self.frame)
        self.assertFalse(set(train) & set(val))
        self.assertEqual(set(train) | set(val), set(range(1,6)))
        np.testing.assert_array_equal(train, split_engines(self.frame)[0])

    def test_scaler_only_sees_training_engines(self):
        train = self.frame[self.frame.engine_id < 4]
        np.testing.assert_allclose(transform(train, self.prep).mean(0), 0, atol=1e-6)
        self.assertGreater(abs(transform(self.frame[self.frame.engine_id == 5], self.prep)[:,0].mean()), 1)

    def test_window_targets_and_no_engine_crossing(self):
        data = make_windows(self.frame, self.prep, 3, 4, 'train')
        np.testing.assert_array_equal(data['end_cycle'][:6], np.arange(3,9))
        np.testing.assert_array_equal(data['y_raw'][:6], [5,4,3,2,1,0])
        np.testing.assert_array_equal(data['y'][:6], [4,4,3,2,1,0])
        np.testing.assert_allclose(data['X'][0], transform(self.frame.iloc[:3], self.prep))
        self.assertTrue(data['mask'].all())

    def test_prefix_padding_and_no_future_features(self):
        frame = self.frame[self.frame.engine_id == 1]
        data = make_windows(frame, self.prep, 4, 125, 'validation', endpoints={1:2})
        np.testing.assert_array_equal(data['mask'][0], [False,False,True,True])
        self.assertEqual(data['y_raw'][0], 6)
        altered = frame.copy()
        altered.loc[altered.cycle > 2, 'sensor_01'] = 1e9
        second = make_windows(altered, self.prep, 4, 125, 'validation', endpoints={1:2})
        np.testing.assert_array_equal(data['X'], second['X'])

    def test_test_labels_are_external_not_observed_end(self):
        data = make_windows(self.frame[self.frame.engine_id == 1], self.prep, 3, 125, 'test', final_rul={1:147})
        self.assertEqual(data['y_raw'][0], 147)
        self.assertEqual(data['y'][0], 125)
        self.assertEqual(data['end_cycle'][0], 8)

    def test_validation_prefixes_fixed_and_before_failure(self):
        points = validation_endpoints(self.frame, 42)
        self.assertEqual(points, validation_endpoints(self.frame,42))
        self.assertTrue(all(1 <= end < 8 for end in points.values()))

    def test_duplicate_and_unordered_cycles_rejected(self):
        duplicate = StringIO(pd.concat([self.frame, self.frame.iloc[:1]]).to_csv(sep=' ', header=False, index=False))
        with self.assertRaises(ValueError):
            load_trajectories(duplicate)
        reversed_rows = StringIO(self.frame.iloc[::-1].to_csv(sep=' ', header=False, index=False))
        with self.assertRaises(ValueError):
            load_trajectories(reversed_rows)


if __name__ == '__main__':
    unittest.main()

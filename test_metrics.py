import unittest
import numpy as np
from metrics import regression_metrics


class MetricTests(unittest.TestCase):
    def test_perfect_predictions(self):
        result = regression_metrics([1,2,3], [1,2,3])
        self.assertEqual(result['RMSE'], 0)
        self.assertEqual(result['R2'], 1)
        self.assertEqual(result['cmapss_score'], 0)

    def test_asymmetric_score_and_units(self):
        result = regression_metrics([10,10], [0,20])
        self.assertEqual(result['MAE'], 10)
        self.assertEqual(result['MSE'], 100)
        self.assertEqual(result['RMSE'], 10)
        self.assertIsNone(result['R2'])
        self.assertAlmostEqual(result['cmapss_score'], np.expm1(10/13)+np.expm1(1))
        self.assertGreater(regression_metrics([10],[20])['cmapss_score'], regression_metrics([10],[0])['cmapss_score'])

    def test_bad_inputs(self):
        for a,b in [([],[]),([1],[1,2]),([np.nan],[1])]:
            with self.assertRaises(ValueError):
                regression_metrics(a,b)


if __name__ == '__main__':
    unittest.main()

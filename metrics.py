"""Regression metrics in cycles. Always declare capped vs raw targets."""
import numpy as np


def regression_metrics(y, prediction):
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if y.shape != prediction.shape or not len(y):
        raise ValueError('Targets/predictions must have matching nonempty shapes')
    if not np.isfinite(y).all() or not np.isfinite(prediction).all():
        raise ValueError('Non-finite targets/predictions')
    error = prediction - y
    mse = float(np.mean(error ** 2))
    total = float(np.sum((y - y.mean()) ** 2))
    score = np.where(error < 0, np.expm1(-error / 13), np.expm1(error / 10))
    return dict(MAE=float(np.mean(abs(error))), MSE=mse, RMSE=float(np.sqrt(mse)),
                R2=float(1 - np.sum(error ** 2) / total) if total > 0 else None,
                cmapss_score=float(score.sum()), n=len(y))

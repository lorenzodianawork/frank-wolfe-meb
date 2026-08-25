"""
Geometric anomaly detection on top of the trained per-class MEB models.

For a test point z of class k with ball center c_hat and radius R, the
anomaly score is ratio(z) = ||z - c_hat|| / R:
    ratio > 1  -> outside the ball (outlier)
    ratio ~ 1  -> on the boundary (borderline)
    ratio << 1 -> deep interior (typical example)
"""

import numpy as np


def compute_anomaly_scores(meb_models, X_test, y_test):
    """Computes distance-to-center and normalized ratio for every test
    point, scored against the MEB of its own (true) class.
    """
    n = X_test.shape[0]
    distances = np.full(n, np.nan)
    ratios = np.full(n, np.nan)

    for label, model in meb_models.items():
        mask = (y_test == label)
        if not np.any(mask):
            continue

        X_c = X_test[mask]
        center, radius = model["center"], model["radius"]

        dist_sq = np.sum(X_c ** 2, axis=1) - 2.0 * (X_c @ center) + center @ center
        dist = np.sqrt(np.maximum(0.0, dist_sq))

        distances[mask] = dist
        ratios[mask] = dist / radius if radius > 0 else np.inf

    return distances, ratios

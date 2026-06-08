"""
Shared evaluation metrics.

Every model outputs p = P(a beats b), so evaluation is identical regardless
of model. This is the single source of truth for scoring. Implement each
function below.
"""
import numpy as np


def accuracy(p, outcome):
    """Fraction of matches where (p > 0.5) agrees with the outcome."""
    return float(np.mean((p > 0.5) == (outcome == 1)))


def log_loss(p, outcome, eps=1e-12):
    """Mean negative log-likelihood. Clip p into [eps, 1-eps] first."""
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(outcome * np.log(p) + (1-outcome) * np.log(1 - p)))


def brier(p, outcome):
    """Mean squared error between predicted prob and outcome."""
    return float(np.mean((p - outcome) ** 2))


def evaluate(p, outcome):
    """Convenience: return a dict of accuracy, log_loss, brier, n."""
    return {
        "accuracy": accuracy(p, outcome),
        "log_loss": log_loss(p, outcome),
        "brier": brier(p, outcome),
        "n": int(len(outcome)),
    }


def calibration_bins(p, outcome, n_bins=10):
    """Return (bin_centers, observed_freq, count) for a reliability diagram."""
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    centers, obs, counts = [], [], []
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() == 0:          # skip empty buckets
            continue
        centers.append(p[mask].mean())        # avg predicted prob in bucket
        obs.append(outcome[mask].mean())       # actual win rate in bucket
        counts.append(int(mask.sum()))         # how many matches landed here
    return np.array(centers), np.array(obs), np.array(counts)


def expected_calibration_error(p, outcome, n_bins=10):
    """Weighted average gap between predicted and observed freq across bins."""
    centers, obs, counts = calibration_bins(p, outcome, n_bins)
    return float(np.sum(counts * np.abs(centers - obs)) / np.sum(counts))

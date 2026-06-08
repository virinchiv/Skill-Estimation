"""
Shared evaluation metrics.

Every model outputs p = P(a beats b), so evaluation is identical regardless
of model. This is the single source of truth for scoring. Implement each
function below.
"""
import numpy as np


def accuracy(p, outcome):
    """Fraction of matches where (p > 0.5) agrees with the outcome."""
    raise NotImplementedError("Phase 2 - P1")


def log_loss(p, outcome, eps=1e-12):
    """Mean negative log-likelihood. Clip p into [eps, 1-eps] first."""
    raise NotImplementedError("Phase 2 - P1")


def brier(p, outcome):
    """Mean squared error between predicted prob and outcome."""
    raise NotImplementedError("Phase 2 - P1")


def evaluate(p, outcome):
    """Convenience: return a dict of accuracy, log_loss, brier, n."""
    raise NotImplementedError("Phase 2 - P1")


def calibration_bins(p, outcome, n_bins=10):
    """Return (bin_centers, observed_freq, count) for a reliability diagram."""
    raise NotImplementedError("Phase 2 - P1")


def expected_calibration_error(p, outcome, n_bins=10):
    """Weighted average gap between predicted and observed freq across bins."""
    raise NotImplementedError("Phase 2 - P1")

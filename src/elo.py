"""
Elo baseline.   [PHASE 3 - owner: P1]

NOT a graphical model -- this is the comparison point that shows whether the
Bayesian model actually buys us anything. Elo gives a point estimate with NO
uncertainty; that gap is exactly what the Bayesian model fills.

Formula:
    prob(a, b) = 1 / (1 + 10**((r[b] - r[a]) / scale))
    on each match: r[a] += k * (outcome - prob);  k ~= 24, base = 1500
"""
import numpy as np


class Elo:
    def __init__(self, n_players, k=24.0, base=1500.0, scale=400.0):
        self.r = np.full(n_players, base)
        self.k, self.scale = k, scale

    def prob(self, a, b):
        """P(a beats b) from current ratings."""
        raise NotImplementedError("Phase 3 - P1")

    def update(self, a, b, outcome):
        """Update both players' ratings after one match."""
        raise NotImplementedError("Phase 3 - P1")

    def fit(self, a, b, outcome):
        """Online pass through training matches (chronological order ideal)."""
        raise NotImplementedError("Phase 3 - P1")

    def predict_proba(self, a, b):
        """Vector of P(a beats b) for arrays of matchups."""
        raise NotImplementedError("Phase 3 - P1")

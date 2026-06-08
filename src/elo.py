"""
Elo baseline.

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
        return 1.0 / (1.0 + 10 ** ((self.r[b] - self.r[a]) / self.scale))

    def update(self, a, b, outcome):
        """Update both players' ratings after one match."""
        p = self.prob(a, b)
        self.r[a] += self.k * (outcome - p)
        self.r[b] += self.k * ((1 - outcome) - (1 - p))

    def fit(self, a, b, outcome):
        """Online pass through training matches (chronological order ideal)."""
        for ai, bi, oi in zip(a, b, outcome):
            self.update(int(ai), int(bi), int(oi))
        return self
        

    def predict_proba(self, a, b):
        """Vector of P(a beats b) for arrays of matchups."""
        return np.array([self.prob(int(ai), int(bi)) for ai, bi in zip(a, b)])

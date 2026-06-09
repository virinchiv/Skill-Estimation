"""
Bayesian Bradley-Terry skill model.

This is the core graphical model that earns the grade.

Graphical model:
    skill_i ~ Normal(0, sigma)            # latent skill, one per player
    P(a beats b) = sigmoid(skill_a - skill_b)
    outcome_m   ~ Bernoulli(P)            # observed match result

Inference:
    - ADVI (fast variational): use for full-data and data-size runs.
    - NUTS  (exact MCMC):       use for small cold-start subsamples where
                                clean posterior bands matter for the figure.

The +race variant adds an antisymmetric race-matchup term to the logit:
    P(a beats b) = sigmoid(skill_a - skill_b + R[race_a, race_b])
    with R[i, j] = -R[j, i].
"""
import numpy as np
import pymc as pm


def fit_bt(a, b, outcome, n_players, sigma=1.0, draws=1000,
           method="advi", seed=0, progress=False):
    """Fit Bayesian Bradley-Terry.

    Returns a dict with per-player posterior 'mean' and 'std', the raw
    'samples' array (players x draws), and the inference 'idata'.
    """
    with pm.Model() as model:
        skill = pm.Normal("skill", mu = 0.0, sigma = sigma, shape = n_players)
        logit = skill[a] - skill[b]
        pm.Bernoulli("obs", logit_p = logit, observed = outcome)
        if method == "advi":
            approx = pm.fit(n=30000, method="advi", random_seed=seed, progressbar=progress)
            idata = approx.sample(draws)
        elif method == "nuts":
            idata = pm.sample(draws, tune=1000, chains=2, cores=1, random_seed=seed, 
                              progressbar=progress, target_accept=0.9)
    
    sk = idata.posterior["skill"].stack(s=("chain", "draw")).values
    return {
        "mean": sk.mean(axis=1),
        "std": sk.std(axis=1),
        "samples": sk,
        "idata": idata,
    }


def fit_bt_race(a, b, ra, rb, outcome, n_players, n_races=4,
                sigma=1.0, draws=1000, method="advi", seed=0, progress=False):
    """Bradley-Terry + learned race-matchup effects.

    Keep the race arrays aligned to the SAME orientation flip used for the
    player pairs, or the signs will be wrong. Returns skill mean/std plus
    race-effect mean/std and the pair-index lookup.
    """
    raise NotImplementedError("Phase 4 - P2 (extension)")


def predict_proba(post, a, b):
    """Posterior predictive P(a beats b) by averaging sigmoid over samples."""
    sk = post["samples"]
    logit = sk[a] - sk[b]
    return (1.0 / (1.0 + np.exp(-logit))).mean(axis=1)
"""Datasets for the Class 11 figures."""
import numpy as np


def inverse_problem(n=600, seed=1, noise=0.05):
    """A forward map that is one-to-one and its inverse, which is not.

    x = y + 0.3 sin(2 pi y) + noise, sampled with y uniform.  Reading the
    pair the other way round gives a target with up to three valid
    answers for one input -- the case a single-valued network cannot fit.
    """
    rng = np.random.default_rng(seed)
    y = rng.uniform(0, 1, n)
    x = y + 0.3 * np.sin(2 * np.pi * y) + noise * rng.normal(size=n)
    return x, y


def heteroscedastic(n=250, seed=3):
    """Noise whose width depends on the input."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n)
    sd = 0.05 + 0.45 * (x + 1) / 2
    return x, np.sin(2.2 * x) + sd * rng.normal(size=n), sd

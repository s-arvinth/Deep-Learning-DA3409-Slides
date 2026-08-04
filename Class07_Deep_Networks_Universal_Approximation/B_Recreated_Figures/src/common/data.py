"""Datasets for the Class 7 figures."""
import numpy as np


def two_moons(n=200, noise=0.10, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, np.pi, n)
    A = np.stack([np.cos(t), np.sin(t)], 1)
    B = np.stack([1 - np.cos(t), 0.4 - np.sin(t)], 1)
    X = np.vstack([A, B]) + noise * rng.normal(size=(2 * n, 2))
    return X, np.hstack([np.zeros(n), np.ones(n)])


def target_2d(X1, X2):
    """The continuous target used in the universal-approximation figure."""
    r2 = X1 ** 2 + X2 ** 2
    return np.exp(-0.7 * r2) * np.sin(2.2 * X1 + 1.1 * X2) + 0.25 * X1

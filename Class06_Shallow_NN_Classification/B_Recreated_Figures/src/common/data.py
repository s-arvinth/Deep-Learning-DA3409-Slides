"""Datasets used across the Class 6 (classification) figures."""
import numpy as np


def two_gaussians(N=120, seed=3, sep=2.4, cov=((1.0, 0.55), (0.55, 0.9))):
    """Two Gaussian classes with a SHARED covariance -- the setting in
    which the posterior is exactly a logistic sigmoid."""
    rng = np.random.default_rng(seed)
    S = np.asarray(cov)
    m1 = np.array([-sep / 2, -0.35])
    m2 = np.array([+sep / 2, +0.35])
    L = np.linalg.cholesky(S)
    X1 = m1 + rng.normal(size=(N, 2)) @ L.T
    X2 = m2 + rng.normal(size=(N, 2)) @ L.T
    X = np.vstack([X1, X2])
    y = np.r_[np.zeros(N), np.ones(N)]
    return X, y, (m1, m2, S)


def three_gaussians(N=90, seed=8):
    rng = np.random.default_rng(seed)
    means = np.array([[-1.7, -1.1], [1.8, -0.9], [0.1, 1.9]])
    S = np.array([[0.85, 0.25], [0.25, 0.70]])
    L = np.linalg.cholesky(S)
    X = np.vstack([m + rng.normal(size=(N, 2)) @ L.T for m in means])
    y = np.repeat(np.arange(3), N)
    return X, y, (means, S)


def separable_1d(N=20, seed=1, outliers=False):
    """One-dimensional two-class data; optionally with a cluster of
    distant but correctly-labelled points."""
    rng = np.random.default_rng(seed)
    x0 = rng.normal(-1.2, 0.55, N)
    x1 = rng.normal(1.2, 0.55, N)
    if outliers:
        x1 = np.r_[x1, rng.normal(6.5, 0.35, N // 2)]
    x = np.r_[x0, x1]
    y = np.r_[np.zeros(len(x0)), np.ones(len(x1))]
    return x, y


def noisy_xor(N=70, seed=11, spread=0.42):
    """Four clusters in XOR arrangement: not linearly separable, but a
    two-unit hidden layer is enough."""
    rng = np.random.default_rng(seed)
    centres = np.array([[-1, -1], [1, 1], [-1, 1], [1, -1]], float)
    labels = np.array([0, 0, 1, 1])
    X = np.vstack([c + spread * rng.normal(size=(N, 2)) for c in centres])
    y = np.repeat(labels, N).astype(float)
    return X, y


def regression_cloud(N=140, seed=4, sd=0.22):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, N)
    mean = 0.35 + 1.25 * x
    return x, mean + sd * rng.normal(size=N), sd

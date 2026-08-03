"""The datasets used across the Class 5 figures."""
import numpy as np


def linear_data(N=18, seed=5, noise=0.16, w0=0.35, w1=1.15):
    """N noisy samples of a line: y_n = w0 + w1 x_n + eps."""
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0, 1, N))
    y = w0 + w1 * x + noise * rng.normal(size=N)
    return x, y


def gaussian_cloud(N=160, seed=2, sd=0.16, w0=0.40, w1=1.00):
    """A dense cloud with constant Gaussian noise about a linear mean."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, N)
    y = w0 + w1 * x + sd * rng.normal(size=N)
    return x, y, (w0, w1, sd)


def target_wave(x):
    """The continuous target used in the approximation figures."""
    return np.sin(3.2 * np.pi * x) * np.exp(-1.2 * x)

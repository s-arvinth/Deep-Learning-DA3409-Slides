"""Datasets and target functions for the Class 9 figures."""
import numpy as np


def targets_1d(x):
    """The four targets used to illustrate universal approximation."""
    return {
        r"$f(x)=x^{2}$":            x ** 2,
        r"$f(x)=\sin(2\pi x)$":     np.sin(2 * np.pi * x),
        r"$f(x)=|x|$":              np.abs(x),
        r"$f(x)=\mathbb{1}[x>0]$":  (x > 0).astype(float),
    }


def smooth_target(x):
    """A single smooth target for the rate experiments."""
    return np.sin(3.0 * np.pi * x) * np.exp(-1.1 * x) + 0.4 * x

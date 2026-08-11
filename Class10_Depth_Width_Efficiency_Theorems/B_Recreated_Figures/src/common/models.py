"""Models and counting formulas for Class 9.

Notation (the course convention, see _shared/NOTATION.md):
    x in R^D          input            D  input dimension
    y, yhat           target, prediction
    z^(l) = h(a^(l))  hidden units     M  width  (units per hidden layer)
    W^(l), b^(l)      layer weights    L  depth  (number of hidden layers)
"""
import numpy as np
from math import comb


def relu(a):
    return np.maximum(a, 0.0)


# ---------------------------------------------------------------------
#  the tent map and its iterates -- the depth-separation witness
# ---------------------------------------------------------------------
def tooth(x):
    """T(x) = 2h(x) - 4h(x - 1/2): a two-unit ReLU network folding [0,1]."""
    return 2.0 * relu(x) - 4.0 * relu(x - 0.5)


def tooth_iter(x, L):
    y = np.asarray(x, float).copy()
    for _ in range(L):
        y = tooth(y)
    return y


# ---------------------------------------------------------------------
#  parameter counts:  D inputs, L hidden layers of width M, 1 output
# ---------------------------------------------------------------------
def n_params(M, L, D=1):
    return M * (D + 1) + (L - 1) * M * (M + 1) + (M + 1)


# ---------------------------------------------------------------------
#  linear-region counts
# ---------------------------------------------------------------------
def regions_shallow(M, D=1):
    """Zaslavsky: M hyperplanes in general position in R^D."""
    return float(sum(comb(int(M), j) for j in range(0, D + 1)))


def regions_deep_lower(M, L, D=1):
    """Montufar et al. (2014) lower bound for width M >= D."""
    if M < D:
        return np.nan
    return float((M // D) ** (D * (L - 1))) * regions_shallow(M, D)


def regions_deep_upper(M, L, D=1):
    """A simple product upper bound (cf. Serra et al., 2018)."""
    return float(regions_shallow(M, D) ** L)


# ---------------------------------------------------------------------
#  approximation rates
# ---------------------------------------------------------------------
def barron_rate(M, C=1.0):
    """Barron (1993): ||f - f_M||_2 <= C / sqrt(M), independent of D."""
    return C / np.sqrt(M)


def sobolev_rate(W, D, s=2.0, C=1.0):
    """DeVore, Howard & Micchelli (1989): W parameters cannot beat
    W^{-s/D} uniformly over the Sobolev ball of smoothness s."""
    return C * np.asarray(W, float) ** (-s / D)


# ---------------------------------------------------------------------
#  VC dimension of a ReLU network (Bartlett et al., 2019): Theta(W L log W)
# ---------------------------------------------------------------------
def vc_dim(W, L):
    W = np.asarray(W, float)
    return W * L * np.log2(np.maximum(W, 2.0))


# ---------------------------------------------------------------------
#  best continuous piecewise-linear approximation with P pieces
# ---------------------------------------------------------------------
def best_pwl(x, y, P):
    """Least-squares fit of a continuous piecewise-linear function with
    P pieces on an even grid -- the best a width-(P-1) shallow ReLU
    network can do with joints in those places."""
    knots = np.linspace(x.min(), x.max(), P + 1)[1:-1]
    A = np.stack([relu(x - c) for c in knots] + [x, np.ones_like(x)], 1)
    w, *_ = np.linalg.lstsq(A, y, rcond=None)
    return A @ w

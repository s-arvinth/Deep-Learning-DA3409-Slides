"""Models for Class 6, written in the course notation.

    x            input          D  = input dimension
    y            target         M  = hidden units
    yhat         prediction     K  = classes / outputs
    a            pre-activation of an OUTPUT unit
    sigma(a)     logistic sigmoid       -> Bernoulli parameter
    softmax(a)   normalised exponential -> categorical parameters
    w^(1), w^(2) all learnable weights
"""
import numpy as np


def sigmoid(a):
    return 0.5 * (1.0 + np.tanh(0.5 * a))        # overflow-safe


def dsigmoid(a):
    s = sigmoid(a)
    return s * (1.0 - s)


def softmax(A, axis=-1):
    A = A - A.max(axis=axis, keepdims=True)      # shift invariance
    E = np.exp(A)
    return E / E.sum(axis=axis, keepdims=True)


def relu(a):
    return np.maximum(a, 0.0)


# ---------------------------------------------------------------------
#  logistic regression, fitted by gradient descent on cross-entropy
# ---------------------------------------------------------------------
def fit_logistic(X, y, steps=4000, lr=0.25):
    Xb = np.c_[np.ones(len(X)), X]
    w = np.zeros(Xb.shape[1])
    for _ in range(steps):
        yhat = sigmoid(Xb @ w)
        w -= lr * (Xb.T @ (yhat - y)) / len(X)   # grad = sum (yhat - y) x
    return w


def fit_least_squares(X, y):
    """Least squares on the SAME 0/1 targets -- the thing we argue against."""
    Xb = np.c_[np.ones(len(X)), X]
    w, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    return w


# ---------------------------------------------------------------------
#  the exact Gaussian-class-conditional posterior (Bishop, 2024, S5.3)
#     shared covariance  =>  p(C1|x) = sigma(w'x + w0)
# ---------------------------------------------------------------------
def gaussian_posterior_weights(m1, m2, S, p1=0.5, p2=0.5):
    Si = np.linalg.inv(S)
    w = Si @ (m2 - m1)
    w0 = (-0.5 * m2 @ Si @ m2 + 0.5 * m1 @ Si @ m1 + np.log(p2 / p1))
    return w, w0


# ---------------------------------------------------------------------
#  a shallow network with M hidden ReLU units and a sigmoid output,
#  trained by full-batch gradient descent on cross-entropy
# ---------------------------------------------------------------------
def fit_shallow_classifier(X, y, M=2, steps=6000, lr=0.35, seed=0):
    rng = np.random.default_rng(seed)
    D = X.shape[1]
    W1 = rng.normal(0, np.sqrt(2.0 / D), (M, D))
    b1 = np.zeros(M)
    W2 = rng.normal(0, np.sqrt(2.0 / M), M)
    b2 = 0.0
    N = len(X)
    for _ in range(steps):
        A1 = X @ W1.T + b1
        Z = relu(A1)
        a = Z @ W2 + b2
        yhat = sigmoid(a)
        d = (yhat - y) / N                        # dE/da  =  yhat - y
        gW2 = Z.T @ d
        gb2 = d.sum()
        dZ = np.outer(d, W2) * (A1 > 0)
        gW1 = dZ.T @ X
        gb1 = dZ.sum(0)
        W1 -= lr * gW1; b1 -= lr * gb1
        W2 -= lr * gW2; b2 -= lr * gb2
    return W1, b1, W2, b2


def shallow_forward(X, W1, b1, W2, b2):
    Z = relu(X @ W1.T + b1)
    return Z, sigmoid(Z @ W2 + b2)


# ---------------------------------------------------------------------
#  fixed Gaussian basis functions (Bishop, 2024, Fig. 5.15)
# ---------------------------------------------------------------------
def gaussian_basis(X, centres, s=0.65):
    d2 = ((X[:, None, :] - centres[None, :, :]) ** 2).sum(-1)
    return np.exp(-d2 / (2 * s ** 2))

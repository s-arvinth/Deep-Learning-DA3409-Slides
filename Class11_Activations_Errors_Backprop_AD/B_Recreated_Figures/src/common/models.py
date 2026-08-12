"""Activations, their derivatives, and a small network we can differentiate.

Notation (the course convention, see _shared/NOTATION.md):
    a^(l)_j            pre-activation of unit j in layer l
    z^(l)_j = h(a)     its activation
    delta^(l)_j        dE/da^(l)_j -- the 'error' backpropagated to that unit
    W^(l), b^(l)       layer weights and biases
"""
import numpy as np


# ---- the activation zoo ---------------------------------------------
def relu(a):        return np.maximum(a, 0.0)
def drelu(a):       return (a > 0).astype(float)

def lrelu(a, s=0.1):  return np.where(a > 0, a, s * a)
def dlrelu(a, s=0.1): return np.where(a > 0, 1.0, s)

def elu(a, al=1.0):   return np.where(a > 0, a, al * (np.exp(np.minimum(a, 0)) - 1))
def delu(a, al=1.0):  return np.where(a > 0, 1.0, al * np.exp(np.minimum(a, 0)))

def sigmoid(a):     return 0.5 * (1.0 + np.tanh(0.5 * a))
def dsigmoid(a):    s = sigmoid(a); return s * (1 - s)

def dtanh(a):       return 1.0 - np.tanh(a) ** 2

def softplus(a):    return np.log1p(np.exp(-np.abs(a))) + np.maximum(a, 0)
def dsoftplus(a):   return sigmoid(a)

def _Phi(a):        return 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) *
                                              (a + 0.044715 * a ** 3)))
def gelu(a):        return a * _Phi(a)
def dgelu(a):
    e = 1e-5
    return (gelu(a + e) - gelu(a - e)) / (2 * e)

def silu(a):        return a * sigmoid(a)
def dsilu(a):       return sigmoid(a) * (1 + a * (1 - sigmoid(a)))


ACTIVATIONS = [
    ("ReLU",        r"$\max(0,a)$",            relu,     drelu),
    ("leaky ReLU",  r"$\max(0.1a,\,a)$",       lrelu,    dlrelu),
    ("ELU",         r"$a$ or $e^{a}-1$",       elu,      delu),
    ("softplus",    r"$\ln(1+e^{a})$",         softplus, dsoftplus),
    ("GELU",        r"$a\,\Phi(a)$",           gelu,     dgelu),
    ("SiLU / swish",r"$a\,\sigma(a)$",         silu,     dsilu),
    ("tanh",        r"$\tanh a$",              np.tanh,  dtanh),
    ("logistic",    r"$1/(1+e^{-a})$",         sigmoid,  dsigmoid),
]


# ---- a deep chain, for measuring how gradients travel ----------------
def layer_gradient_norms(L=20, M=64, gain=1.0, act="tanh", seed=0, N=256):
    """Forward a random input through L layers, backpropagate a unit
    signal, and record the gradient norm arriving at each layer."""
    rng = np.random.default_rng(seed)
    f, df = (np.tanh, dtanh) if act == "tanh" else (relu, drelu)
    Ws = [rng.normal(0, gain / np.sqrt(M), (M, M)) for _ in range(L)]
    Z = [rng.normal(size=(N, M))]
    A = []
    for W in Ws:
        A.append(Z[-1] @ W.T)
        Z.append(f(A[-1]))
    delta = rng.normal(size=(N, M)) / np.sqrt(M)      # unit signal at the top
    norms = []
    for l in range(L - 1, -1, -1):
        delta = delta * df(A[l])
        norms.append(np.sqrt((delta ** 2).sum(1)).mean())
        delta = delta @ Ws[l]
    return np.array(norms[::-1])


# ---- how many ReLU units are dead ------------------------------------
def dead_fraction(bias, M=512, D=64, N=2048, seed=0):
    """Fraction of units whose pre-activation is negative for EVERY input.

    Such a unit outputs zero on all data and has zero gradient, so no
    update can ever revive it.  A large negative bias -- which gradient
    descent can produce -- is what drives units into that state.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N, D))
    W = rng.normal(0, 1.0 / np.sqrt(D), (M, D))
    A = X @ W.T + bias
    return float((A.max(0) <= 0).mean())

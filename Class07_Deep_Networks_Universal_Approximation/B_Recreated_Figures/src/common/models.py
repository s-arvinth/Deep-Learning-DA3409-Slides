"""Models for Class 7, in the course notation.

    x                      input
    z^(l)_j                unit j of hidden layer l
    a^(l)_j                its pre-activation
    w^(l)_{ji}, w^(l)_{j0} weights and bias of layer l
    L                      number of hidden layers (depth)
    M                      units per hidden layer (width)
    h(.)                   activation, ReLU throughout

The two running networks compose to make the deep example:

    f1:  z_1 = h(x + 1),  z_2 = h(x + 0.5),  z_3 = h(x - 0.5)
         a   = -1 + 4 z_1 - 6 z_2 + 6 z_3        (a zigzag, 3 branches)

    f2:  the same shape of network applied to f1's output
"""
import numpy as np


def relu(a):
    return np.maximum(a, 0.0)


# ---- network 1 and network 2 (scalar in, scalar out) -----------------
W1_1 = np.array([[1.0, 1.0], [0.5, 1.0], [-0.5, 1.0]])   # [bias, slope]
W2_1 = np.array([-1.0, 4.0, -6.0, 6.0])                  # [bias, weights]
W1_2 = np.array([[1.0, 1.0], [0.0, 1.0], [-0.5, 1.0]])
W2_2 = np.array([-1.0, 2.0, -3.5, 3.0])


def layer(x, W1, W2):
    """Return (pre-activations a, activations z, output)."""
    a = W1[:, 0][:, None] + W1[:, 1][:, None] * x[None, :]
    z = relu(a)
    return a, z, W2[0] + (W2[1:][:, None] * z).sum(axis=0)


def f1(x):
    return layer(x, W1_1, W2_1)[2]


def f2(x):
    return layer(x, W1_2, W2_2)[2]


# ---- the two-input network used for the 2-D composition --------------
V1 = np.array([[ 1.4, -0.6,  0.15],      # [w_j1, w_j2, w_j0]
               [-0.9, -1.1,  0.35],
               [ 0.3,  1.5,  0.10]])
V2 = np.array([0.0, 1.6, -1.4, 1.2])


def net2d(X1, X2):
    a = (V1[:, 0][:, None, None] * X1 + V1[:, 1][:, None, None] * X2
         + V1[:, 2][:, None, None])
    return a, V2[0] + (V2[1:][:, None, None] * relu(a)).sum(0)


# ---- the sawtooth: the witness for the depth separation --------------
def tooth(x):
    """T(x) = 2 h(x) - 4 h(x - 1/2)  maps [0,1] onto [0,1].

    Exactly a two-unit ReLU network, and T is a single 'tent': it folds
    [0,1] in half.  Composing it L times gives 2^L linear pieces.
    """
    return 2.0 * relu(x) - 4.0 * relu(x - 0.5)


def tooth_iter(x, L):
    y = x.copy()
    for _ in range(L):
        y = tooth(y)
    return y


# ---- parameter and region counts -------------------------------------
def params_shallow(M):
    """1 input, M hidden units, 1 output: 2M weights + M+1 biases/weights."""
    return 3 * M + 1


def params_deep(M, L):
    return 2 * M + (L - 1) * M * (M + 1) + M + 1


def regions_shallow(M):
    return M + 1.0


def regions_deep(M, L):
    return (M + 1.0) ** L

"""The running models for Class 5, in the notation used on the slides.

Notation (Bishop & Bishop, 2024, Ch. 4 and 6):
    x            input          D  = input dimension
    y            target         M  = hidden units
    yhat         prediction     K  = outputs
    a_j          pre-activation of hidden unit j
    z_j = h(a_j) hidden unit (activation)
    w^(1)_{ji}   weight from input i into hidden unit j;  w^(1)_{j0} its bias
    w^(2)_{kj}   weight from hidden unit j into output k; w^(2)_{k0} its bias
    phi_j(x)     a FIXED basis function -- never a parameter
"""
import numpy as np


def relu(a):
    return np.maximum(a, 0.0)


# ---------------------------------------------------------------------
#  the running scalar-input network:  D = 1, M = 3, K = 1
#     a_j    = w1[j,0] + w1[j,1] * x
#     z_j    = ReLU(a_j)
#     yhat   = w2[0] + sum_j w2[j+1] * z_j
# ---------------------------------------------------------------------
W1 = np.array([[ 0.55, 1.0],          # [w^(1)_{10}, w^(1)_{11}]
               [-0.15, 1.0],
               [-0.70, 1.0]])
W2 = np.array([-0.35, 1.6, -3.2, 2.4])   # [w^(2)_0, w^(2)_1, w^(2)_2, w^(2)_3]


def shallow(x, W1=W1, W2=W2):
    """Return (pre-activations a, activations z, output yhat)."""
    a = W1[:, 0][:, None] + W1[:, 1][:, None] * x[None, :]
    z = relu(a)
    return a, z, W2[0] + (W2[1:][:, None] * z).sum(0)


def joints(W1=W1):
    """Where each ReLU switches on: a_j = 0  =>  x = -w_{j0}/w_{j1}."""
    return [-b / w for b, w in W1]


# ---------------------------------------------------------------------
#  the running two-input network:  D = 2, M = 3, K = 1
# ---------------------------------------------------------------------
V1 = np.array([[ 1.6, -0.7,  0.10],   # [w^(1)_{j1}, w^(1)_{j2}, w^(1)_{j0}]
               [-1.0, -1.2,  0.30],
               [ 0.4,  1.7,  0.05]])
V2 = np.array([0.0, 1.5, -1.3, 1.1])


def shallow2d(X1, X2, V1=V1, V2=V2):
    a = (V1[:, 0][:, None, None] * X1 + V1[:, 1][:, None, None] * X2
         + V1[:, 2][:, None, None])
    z = relu(a)
    return a, V2[0] + (V2[1:][:, None, None] * z).sum(0)


# ---------------------------------------------------------------------
#  a ReLU network written as a linear model in learned features, so it
#  can be fitted in closed form for illustration purposes
# ---------------------------------------------------------------------
def relu_design(x, cuts):
    """Design matrix [ReLU(x - c_1) ... ReLU(x - c_M), x, 1]."""
    return np.stack([relu(x - c) for c in cuts] +
                    [x, np.ones_like(x)], axis=1)


def fit_relu(x, y, M):
    """Least-squares fit of an M-unit ReLU network with evenly spaced joints."""
    cuts = np.linspace(x.min(), x.max(), M + 2)[1:-1]
    A = relu_design(x, cuts)
    w, *_ = np.linalg.lstsq(A, y, rcond=None)
    return cuts, w

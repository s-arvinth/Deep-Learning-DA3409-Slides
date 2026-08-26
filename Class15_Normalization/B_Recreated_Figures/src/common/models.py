"""Networks, statistics and normalizers for the Class 15 figures.

Everything is a few lines of numpy so a student can read the rule off
the code and match it to the slide.

Notation is the frozen course convention:

    w           every learnable parameter
    a, z        pre-activation and activation
    E(w)        the error function
    D           fan-in of a layer (Prince writes D_h)
    sigma^2_w   variance of the initialization distribution
    gamma, beta learnable scale and shift of a normalization layer
    kappa       condition number of the Hessian

Prince writes the weights Omega and the pre-activations f; Bishop writes
w and a.  Both are translated here, once, at the source.
"""
import numpy as np


# =====================================================================
#  Where the condition number comes from, for a linear model
# =====================================================================
def linear_hessian(X):
    """Hessian of the least-squares error for a linear model.

    E(w) = sum_n (w^T x_n - y_n)^2  has  H = 2 X^T X, so the curvature
    in direction i is set by the second moment of feature i and the
    off-diagonal terms by the correlations between features.
    """
    return 2.0 * X.T @ X / len(X)


def condition_number(H):
    ev = np.linalg.eigvalsh(H)
    ev = ev[ev > 1e-14]
    return ev.max() / ev.min() if len(ev) else np.inf


def standardize(X, mu=None, sd=None):
    """Bishop Eqs. 7.48--7.50: zero mean and unit variance per feature.

    The statistics are returned so that exactly the same numbers can be
    applied to validation and test data.
    """
    if mu is None:
        mu = X.mean(axis=0)
    if sd is None:
        sd = X.std(axis=0)
    return (X - mu) / sd, mu, sd


def two_feature_data(n=400, scale=(1.0, 1.0), corr=0.0, seed=0):
    """Two features with prescribed scales and correlation."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 2))
    z[:, 1] = corr * z[:, 0] + np.sqrt(max(1.0 - corr ** 2, 0.0)) * z[:, 1]
    return z * np.asarray(scale)


# =====================================================================
#  Variance propagation through a deep ReLU network
# =====================================================================
def relu(x):
    return np.maximum(x, 0.0)


def forward_variances(D=100, K=50, sigma2=0.02, n=256, seed=0):
    """Variance of the pre-activations, layer by layer.

    Each layer computes a = W z with W drawn from N(0, sigma2) and
    z = ReLU(a_prev).  Because ReLU discards half the mass of a
    symmetric distribution, one layer multiplies the variance by
    D sigma2 / 2 -- so over K layers the factor is that number raised
    to the Kth power.
    """
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, D))
    out = [a.var()]
    for _ in range(K):
        W = rng.normal(0.0, np.sqrt(sigma2), size=(D, D))
        a = relu(a) @ W.T
        out.append(a.var())
    return np.array(out)


def backward_variances(D=100, K=50, sigma2=0.02, n=256, seed=0):
    """Variance of the gradient signal, layer by layer, going back.

    The backward pass multiplies by W^T and masks by the ReLU
    indicator, so the same geometric factor governs it.
    """
    rng = np.random.default_rng(seed)
    Ws, masks = [], []
    a = rng.standard_normal((n, D))
    for _ in range(K):
        W = rng.normal(0.0, np.sqrt(sigma2), size=(D, D))
        z = relu(a)
        masks.append((a > 0).astype(float))
        a = z @ W.T
        Ws.append(W)
    g = rng.standard_normal((n, D))
    out = [g.var()]
    for W, m in zip(reversed(Ws), reversed(masks)):
        g = (g @ W) * m
        out.append(g.var())
    return np.array(out[::-1])


def he_variance(fan_in):
    """Bishop Eq. 7.23 / Prince Eq. 7.31:  sigma^2 = 2 / fan_in."""
    return 2.0 / fan_in


def glorot_variance(fan_in, fan_out):
    """Prince Eq. 7.33: the compromise between forward and backward."""
    return 4.0 / (fan_in + fan_out)


def lecun_variance(fan_in):
    """The pre-ReLU rule, without the factor of two."""
    return 1.0 / fan_in


# =====================================================================
#  The three normalizers
# =====================================================================
def batch_norm(A, eps=1e-5, gamma=None, beta=None):
    """Bishop Eqs. 7.52--7.55.

    `A` is (batch, units).  The mean and variance are taken down each
    COLUMN, that is across the mini-batch, separately for each unit.
    """
    mu = A.mean(axis=0, keepdims=True)
    var = A.var(axis=0, keepdims=True)
    Ah = (A - mu) / np.sqrt(var + eps)
    if gamma is None:
        return Ah
    return gamma * Ah + beta


def layer_norm(A, eps=1e-5, gamma=None, beta=None):
    """Bishop Eqs. 7.58--7.60.

    The same expression with the average taken along the other axis:
    across the units, separately for each data point.
    """
    mu = A.mean(axis=1, keepdims=True)
    var = A.var(axis=1, keepdims=True)
    Ah = (A - mu) / np.sqrt(var + eps)
    if gamma is None:
        return Ah
    return gamma * Ah + beta


def group_norm(A, groups=4, eps=1e-5):
    """Layer normalization applied to disjoint blocks of units."""
    n, m = A.shape
    g = m // groups
    out = np.empty_like(A)
    for k in range(groups):
        sl = slice(k * g, (k + 1) * g)
        blk = A[:, sl]
        mu = blk.mean(axis=1, keepdims=True)
        var = blk.var(axis=1, keepdims=True)
        out[:, sl] = (blk - mu) / np.sqrt(var + eps)
    return out


def batch_norm_backward(G, A, eps=1e-5):
    """The exact gradient through `batch_norm`, with no learnable scale.

    With  z = (a - mu)/sigma  computed down each column of `A`, and G
    the gradient arriving at z,

        dE/da = ( B G - sum(G) - z * sum(G z) ) / (B sigma)

    where B is the number of rows and every operation is per column.
    """
    B = A.shape[0]
    mu = A.mean(axis=0, keepdims=True)
    var = A.var(axis=0, keepdims=True)
    sd = np.sqrt(var + eps)
    Z = (A - mu) / sd
    return (B * G - G.sum(axis=0, keepdims=True)
            - Z * (G * Z).sum(axis=0, keepdims=True)) / (B * sd)


def layer_norm_backward(G, A, eps=1e-5):
    """The same expression with the average taken along the rows."""
    M = A.shape[1]
    mu = A.mean(axis=1, keepdims=True)
    var = A.var(axis=1, keepdims=True)
    sd = np.sqrt(var + eps)
    Z = (A - mu) / sd
    return (M * G - G.sum(axis=1, keepdims=True)
            - Z * (G * Z).sum(axis=1, keepdims=True)) / (M * sd)


def running_average(prev, new, alpha=0.9):
    """Bishop Eqs. 7.56--7.57: the statistics kept for inference."""
    return alpha * prev + (1.0 - alpha) * new


# =====================================================================
#  A small trainable network, used by the training-curve figures
# =====================================================================
def make_net(dims, sigma2, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.normal(0.0, np.sqrt(sigma2(dims[i], dims[i + 1])),
                       size=(dims[i + 1], dims[i]))
            for i in range(len(dims) - 1)]


def forward(Ws, X, norm=None):
    """Forward pass, optionally inserting a normalizer per hidden layer.

    Returns the output and the list of pre-activations, so the figures
    can plot what the statistics were doing.
    """
    A = X
    acts = []
    for k, W in enumerate(Ws):
        A = A @ W.T
        if norm is not None and k < len(Ws) - 1:
            A = norm(A)
        acts.append(A)
        if k < len(Ws) - 1:
            A = relu(A)
    return A, acts


def train(Ws, X, Y, eta, steps, norm=None, batch=32, seed=0,
          norm_backward=None):
    """Plain SGD on a least-squares objective for a ReLU stack.

    `norm` is applied to the pre-activations of every hidden layer and
    `norm_backward` is its exact gradient; passing `batch_norm` with
    `batch_norm_backward` gives a correct batch-normalized network.
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    losses = []
    Ws = [W.copy() for W in Ws]
    for t in range(steps):
        idx = rng.choice(n, size=min(batch, n), replace=False)
        xb, yb = X[idx], Y[idx]
        # ---- forward, keeping what the backward pass needs ----------
        A, cache = xb, []
        for k, W in enumerate(Ws):
            pre = A @ W.T
            post = pre
            if norm is not None and k < len(Ws) - 1:
                post = norm(pre)
            cache.append((A, pre, post))
            A = relu(post) if k < len(Ws) - 1 else post
        losses.append(float(((A - yb) ** 2).mean()))
        # ---- backward ------------------------------------------------
        G = 2.0 * (A - yb) / len(xb)
        for k in range(len(Ws) - 1, -1, -1):
            Ain, pre, post = cache[k]
            if k < len(Ws) - 1:
                G = G * (post > 0)
                if norm is not None:
                    bw = norm_backward or batch_norm_backward
                    G = bw(G, pre)
            dW = G.T @ Ain
            G = G @ Ws[k]
            Ws[k] = Ws[k] - eta * dW
    return np.array(losses), Ws

"""Models and optimisers used by the Class 13 figures.

Everything here is deliberately small and explicit so a student can read
the update rule off the code and match it to the slide.

Notation follows the frozen course convention (see _shared/NOTATION.md):

    w       the parameter vector, whatever the book calls it
    E(w)    the error function
    eta     the learning rate
    B       the mini-batch size

Prince writes the parameters as `phi`, the learning rate as `alpha` and
the loss as `L[.]`.  Those are translated here once, at the source, so no
figure and no slide has to carry two notations.
"""
import numpy as np


# =====================================================================
#  A two-parameter quadratic error surface
# =====================================================================
def quadratic_E(W1, W2, lam=(1.0, 8.0), centre=(0.0, 0.0), rot=0.0):
    """E(w) = E* + 1/2 sum_i lambda_i alpha_i^2  (Bishop Eq. 7.11).

    `lam` are the Hessian eigenvalues, `rot` rotates the eigenvectors
    away from the coordinate axes so the picture is not degenerate.
    """
    c, s = np.cos(rot), np.sin(rot)
    d1, d2 = W1 - centre[0], W2 - centre[1]
    a1 = c * d1 + s * d2
    a2 = -s * d1 + c * d2
    return 0.5 * (lam[0] * a1 ** 2 + lam[1] * a2 ** 2)


def quadratic_grad(w, lam=(1.0, 8.0), centre=(0.0, 0.0), rot=0.0):
    """Gradient of `quadratic_E`, i.e. H (w - w*)."""
    c, s = np.cos(rot), np.sin(rot)
    R = np.array([[c, s], [-s, c]])
    H = R.T @ np.diag(lam) @ R
    return H @ (np.asarray(w, float) - np.asarray(centre, float))


def hessian(lam=(1.0, 8.0), rot=0.0):
    c, s = np.cos(rot), np.sin(rot)
    R = np.array([[c, s], [-s, c]])
    return R.T @ np.diag(lam) @ R


# =====================================================================
#  A non-convex two-parameter surface, for the landscape figures
# =====================================================================
def bumpy_E(W1, W2):
    """A smooth non-convex surface with two minima and a saddle.

    Used only to make the geometry of Bishop Fig. 7.1 concrete: one
    local minimum, one global minimum, and a gradient arrow in between.
    """
    return (0.6 * (W1 ** 2 + W2 ** 2)
            - 1.9 * np.exp(-((W1 + 1.15) ** 2 + (W2 + 0.75) ** 2) / 0.55)
            - 2.9 * np.exp(-((W1 - 1.20) ** 2 + (W2 - 0.65) ** 2) / 0.55))


def bumpy_grad(w):
    w1, w2 = float(w[0]), float(w[1])
    g1 = 1.2 * w1
    g2 = 1.2 * w2
    for amp, c1, c2 in ((-1.9, -1.15, -0.75), (-2.9, 1.20, 0.65)):
        e = np.exp(-((w1 - c1) ** 2 + (w2 - c2) ** 2) / 0.55)
        g1 += amp * e * (-2 * (w1 - c1) / 0.55)
        g2 += amp * e * (-2 * (w2 - c2) / 0.55)
    return np.array([g1, g2])


# =====================================================================
#  Prince's Gabor model  (Prince 2023, Eq. 6.8)
# =====================================================================
def gabor(x, w):
    """f(x, w) = sin(w0 + 0.06 w1 x) exp(-(w0 + 0.06 w1 x)^2 / 32)."""
    u = w[0] + 0.06 * w[1] * np.asarray(x, float)
    return np.sin(u) * np.exp(-(u ** 2) / 32.0)


def gabor_data(seed=7, n=28, w_true=(0.0, 16.6), noise=0.06):
    """Prince's training set: 28 points, uniform x, Gabor plus noise."""
    rng = np.random.default_rng(seed)
    x = np.linspace(-15.0, 15.0, n)
    y = gabor(x, np.array(w_true)) + noise * rng.standard_normal(n)
    return x, y


def gabor_loss(w0, w1, x, y):
    """Least-squares loss (Prince Eq. 6.9), vectorised over a w-grid."""
    w0 = np.atleast_1d(w0)
    u = w0[..., None] + 0.06 * np.atleast_1d(w1)[..., None] * x
    pred = np.sin(u) * np.exp(-(u ** 2) / 32.0)
    return ((pred - y) ** 2).sum(axis=-1)


def gabor_grad(w, x, y):
    """Analytic gradient of the Gabor least-squares loss."""
    u = w[0] + 0.06 * w[1] * x
    e = np.exp(-(u ** 2) / 32.0)
    f = np.sin(u) * e
    dfdu = np.cos(u) * e + np.sin(u) * e * (-u / 16.0)
    r = 2.0 * (f - y)
    return np.array([(r * dfdu).sum(), (r * dfdu * 0.06 * x).sum()])


# =====================================================================
#  Optimisers
# =====================================================================
def gradient_descent(grad, w0, eta, steps):
    """Batch gradient descent, Bishop Eq. 7.16:  w <- w - eta grad E."""
    w = np.asarray(w0, float).copy()
    path = [w.copy()]
    for _ in range(steps):
        w = w - eta * grad(w)
        path.append(w.copy())
    return np.array(path)


def sgd(grad_batch, w0, eta, steps, n, batch, seed=0):
    """Mini-batch SGD, Bishop Alg. 7.2 / Prince Eq. 6.10.

    Draws without replacement and reshuffles at the end of each epoch,
    exactly as Algorithm 7.2 specifies.
    """
    rng = np.random.default_rng(seed)
    w = np.asarray(w0, float).copy()
    path = [w.copy()]
    order = rng.permutation(n)
    k = 0
    for _ in range(steps):
        if k + batch > n:
            order = rng.permutation(n)
            k = 0
        idx = order[k:k + batch]
        k += batch
        w = w - eta * grad_batch(w, idx) * (n / len(idx))
        path.append(w.copy())
    return np.array(path)


# =====================================================================
#  The mini-batch gradient as an estimator  (Bishop Section 7.2.4)
# =====================================================================
def batch_gradient_samples(per_point_grads, batch, draws, seed=0):
    """Sample the mini-batch gradient estimate `draws` times.

    `per_point_grads` is the (N, d) array of per-example gradients.  The
    mini-batch estimate is their mean over a random subset of size
    `batch`; its standard error is sigma / sqrt(batch), which is the
    quantity Bishop Section 7.2.4 uses to argue for diminishing returns.
    """
    rng = np.random.default_rng(seed)
    n = len(per_point_grads)
    out = np.empty((draws, per_point_grads.shape[1]))
    for i in range(draws):
        idx = rng.choice(n, size=batch, replace=False)
        out[i] = per_point_grads[idx].mean(axis=0)
    return out

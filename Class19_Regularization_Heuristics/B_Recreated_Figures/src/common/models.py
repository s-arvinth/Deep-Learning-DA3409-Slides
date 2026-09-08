"""The objects Class 19 experiments on.

Everything in this class is a heuristic, so everything here is measured
rather than asserted. Three settings are enough for all of it:

1. A 1-D regression problem fitted in a random-feature basis. This is
   the object early stopping stops, ensembles average, dropout thins and
   noise perturbs.

2. A quadratic error with known Hessian eigenvalues, so the gradient
   descent path and the ridge path can be compared coefficient by
   coefficient. This is where early stopping and weight decay are shown
   to be doing the same thing.

3. A linear model with noisy inputs, where the equivalence between
   training on perturbed inputs and adding a penalty is exact rather
   than approximate.
"""
import numpy as np

SIGMA = 0.22


# ------------------------------------------------------ 1-D regression
def truth(x):
    return np.sin(2.0 * np.pi * x) + 0.4 * np.sin(6.0 * np.pi * x)


def sample(n, rng, sigma=SIGMA):
    x = np.sort(rng.uniform(0.0, 1.0, size=n))
    return x, truth(x) + sigma * rng.normal(size=n)


def features(x, K, rng=None, width=0.075, centres=None):
    """K Gaussian bumps of fixed width, plus a constant column."""
    if centres is None:
        centres = np.linspace(-0.03, 1.03, K)
    Phi = np.exp(-0.5 * ((np.asarray(x)[:, None] - centres[None, :])
                         / width) ** 2)
    return np.hstack([np.ones((len(np.atleast_1d(x)), 1)), Phi]), centres


def sgd_path(Phi, y, steps, lr, rng, drop=0.0, batch=8, record=None):
    """Plain SGD on the squared error, optionally with dropout.

    `drop` is the probability that a feature is zeroed on a given step,
    exactly as dropout zeroes a hidden unit. The bias column is never
    dropped.
    """
    n, m = Phi.shape
    w = np.zeros(m)
    out = {}
    keep = 1.0 - drop
    for t in range(1, steps + 1):
        idx = rng.integers(0, n, size=batch)
        P = Phi[idx]
        if drop > 0.0:
            mask = np.ones(m)
            mask[1:] = (rng.random(m - 1) < keep) / keep
            P = P * mask
        g = 2.0 * P.T @ (P @ w - y[idx]) / batch
        w = w - lr * g
        if record and t in record:
            out[t] = w.copy()
    return w, out


# ------------------------------------------- quadratic: the two paths
def gd_filter(lam, eta, tau):
    """Coefficient of w* reached after tau steps of gradient descent
    from the origin on a quadratic with curvature lam."""
    return 1.0 - (1.0 - eta * np.asarray(lam, float)) ** tau


def ridge_filter(lam, alpha):
    """Coefficient of w* at the minimum of E + (alpha/2)||w||^2."""
    lam = np.asarray(lam, float)
    return lam / (lam + alpha)


def matched_alpha(eta, tau):
    """Bishop's correspondence: the penalty that a run of tau steps at
    step size eta imitates."""
    return 1.0 / (eta * tau)


# -------------------------------------------------- linear + input noise
def linear_problem(n, p, rng, sigma=0.5):
    w = rng.normal(size=p)
    X = rng.normal(size=(n, p))
    return X, X @ w + sigma * rng.normal(size=n), w


def ridge_solution(X, Y, lam):
    n, p = X.shape
    return np.linalg.solve(X.T @ X + lam * n * np.eye(p), X.T @ Y)


def train_with_input_noise(X, Y, var_x, rng, steps=4000, lr=0.02):
    """Full-batch descent where fresh noise is added to the inputs at
    every step. For a linear model this converges to ridge with
    lambda = var_x, exactly."""
    n, p = X.shape
    w = np.zeros(p)
    for _ in range(steps):
        Xe = X + np.sqrt(var_x) * rng.normal(size=X.shape)
        w = w - lr * (2.0 * Xe.T @ (Xe @ w - Y) / n)
    return w


# ------------------------------------------------------------ ensembles
def bagged_ensemble(x, y, K, L, rng, grid, width=0.11, lam=1e-6):
    """L models, each fitted to its own bootstrap resample.

    `lam` is numerical only: a resample repeats points, so the design can
    be rank-deficient, and without it the members would differ because of
    conditioning rather than because of the data.
    """
    F = np.empty((L, len(grid)))
    Pg, centres = features(grid, K, width=width)
    P0 = np.eye(K + 1); P0[0, 0] = 0.0
    for l in range(L):
        idx = rng.integers(0, len(x), size=len(x))
        P, _ = features(x[idx], K, width=width, centres=centres)
        w = np.linalg.solve(P.T @ P + lam * P0, P.T @ y[idx])
        F[l] = Pg @ w
    return F


# ------------------------------------ descent on a quadratic, as a path
def gd_path_quadratic(lam, wstar, eta, tau):
    """Gradient descent from the origin on  1/2 sum lam_i (w_i - w*_i)^2 .

    Returns the whole path, shape (tau + 1, 2).  Each coordinate follows
    w_i^(t) = (1 - (1 - eta lam_i)^t) w*_i, which is what iterating the
    update gives; the function iterates rather than using the formula,
    so the figure and the theorem are independent of each other.
    """
    lam = np.asarray(lam, float); wstar = np.asarray(wstar, float)
    w = np.zeros(2); path = [w.copy()]
    for _ in range(tau):
        w = w - eta * lam * (w - wstar)
        path.append(w.copy())
    return np.array(path)


# --------------------------- the bump model trained on NOISY INPUTS
def sgd_input_noise(x, y, K, sigma_x, rng, steps=6000, lr=0.05, batch=8,
                    width=0.075):
    """SGD on the bump model where fresh noise of standard deviation
    sigma_x is added to the INPUT x at every step, before the features
    are computed.  The model is nonlinear in x, so this is the genuine
    thing Prince (2023), Section 9.3.4 describes, not the linear
    shortcut."""
    _, centres = features(x, K, width=width)
    w = np.zeros(K + 1)
    for _ in range(steps):
        idx = rng.integers(0, len(x), size=batch)
        xe = x[idx] + sigma_x * rng.normal(size=batch)
        P, _ = features(xe, K, width=width, centres=centres)
        g = 2.0 * P.T @ (P @ w - y[idx]) / batch
        w = w - lr * g
    return w, centres


# ---------------------------------------- exact Bayesian linear regression
def bayes_posterior(Phi, y, noise_var, prior_var):
    """Gaussian prior N(0, prior_var I) and Gaussian likelihood of
    variance noise_var give a Gaussian posterior in closed form:
        S = (I/prior_var + Phi^T Phi / noise_var)^{-1},
        m = S Phi^T y / noise_var.
    The bias column is given a broad prior so it is not shrunk."""
    m_ = Phi.shape[1]
    A = np.eye(m_) / prior_var
    A[0, 0] = 1.0 / (100.0 * prior_var)
    S = np.linalg.inv(A + Phi.T @ Phi / noise_var)
    m = S @ Phi.T @ y / noise_var
    return m, S


def bayes_predictive(Pg, m, S, noise_var):
    """Mean and standard deviation of y at the rows of Pg."""
    mu = Pg @ m
    var = noise_var + np.einsum("ij,jk,ik->i", Pg, S, Pg)
    return mu, np.sqrt(var)

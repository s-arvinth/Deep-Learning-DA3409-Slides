"""The three data-generating processes and estimators used in this deck.

Everything here is exact or Monte-Carlo, never hand-drawn, so the numbers
printed on a slide are numbers that were measured.

1. `truth` / `sample`   -- a smooth 1-D regression problem with additive
   Gaussian noise, used for the bias-variance anatomy. This is our own
   example; it plays the role Bishop's sinusoid and Prince's
   quasi-sinusoid play in their books.

2. `piecewise_fit`      -- a least-squares fit in a basis of `K` bumps.
   Increasing `K` increases capacity, exactly as adding hidden units
   does in Prince's simplified model.

3. `minnorm_risk`       -- the minimum-norm ("ridgeless") least-squares
   estimator for an isotropic Gaussian design. This is the object whose
   risk has a known closed form and diverges at the interpolation
   threshold, so it is what lets us DERIVE double descent rather than
   only exhibit it.
"""
import numpy as np

SIGMA = 0.22          # observation noise of the 1-D problem


# ---------------------------------------------------------------- 1-D toy
def truth(x):
    """The regression function h(x) = E[Y | x]."""
    return np.sin(2.0 * np.pi * x) + 0.45 * np.sin(6.0 * np.pi * x)


def sample(n, rng, sigma=SIGMA, sort=True):
    """One data set of size n drawn from h(x) + N(0, sigma^2)."""
    x = rng.uniform(0.0, 1.0, size=n)
    if sort:
        x = np.sort(x)
    return x, truth(x) + sigma * rng.normal(size=n)


def _design(x, K):
    """K equally spaced Gaussian bumps plus a constant."""
    c = np.linspace(0.0, 1.0, K)
    s = 1.0 / max(K - 1, 1) if K > 1 else 0.5
    Phi = np.exp(-0.5 * ((x[:, None] - c[None, :]) / s) ** 2)
    return np.hstack([np.ones((len(x), 1)), Phi])


def piecewise_fit(x, y, K, lam=1e-8, grid=None):
    """Ridge-regularised least squares in the K-bump basis.

    `lam` defaults to a value small enough to be numerically, but not
    statistically, a regulariser: it only keeps the normal equations
    solvable.
    """
    Phi = _design(x, K)
    A = Phi.T @ Phi + lam * np.eye(Phi.shape[1])
    w = np.linalg.solve(A, Phi.T @ y)
    if grid is None:
        return w
    return _design(grid, K) @ w


# ------------------------------------------------- bias / variance sweep
def bias_variance(K_or_lam, grid, n=25, L=200, seed=0, sweep="capacity",
                  K=12, sigma=SIGMA):
    """Monte-Carlo estimate of integrated squared bias and variance.

    Returns (bias2, variance) evaluated on `grid`, following Bishop
    (2024), Eqs. 4.51 and 4.52: L independent data sets of size n, one
    fit per data set, the ensemble average compared with h(x).
    """
    rng = np.random.default_rng(seed)
    F = np.empty((L, len(grid)))
    for l in range(L):
        x, y = sample(n, rng, sigma)
        if sweep == "capacity":
            F[l] = piecewise_fit(x, y, int(K_or_lam), grid=grid)
        else:                                    # sweep the penalty
            F[l] = piecewise_fit(x, y, K, lam=K_or_lam, grid=grid)
    fbar = F.mean(axis=0)
    bias2 = np.mean((fbar - truth(grid)) ** 2)
    var = np.mean(F.var(axis=0))
    return bias2, var


# ------------------------------------------- minimum-norm least squares
def minnorm_risk(n, p, reps, rng, sigma=1.0, beta_norm2=1.0):
    """Monte-Carlo excess risk of the ridgeless least-squares estimator.

    Model: Y = X w + eps, rows of X iid N(0, I_p), eps ~ N(0, sigma^2).
    The estimator is the minimum-l2-norm solution, w_hat = X^+ Y, which
    interpolates the data whenever p >= n.

    For isotropic X the out-of-sample excess risk equals ||w_hat - w||^2,
    which is what we return (averaged over `reps` draws).
    """
    w = np.zeros(p)
    w[0] = np.sqrt(beta_norm2)
    out = np.empty(reps)
    for r in range(reps):
        X = rng.normal(size=(n, p))
        Y = X @ w + sigma * rng.normal(size=n)
        out[r] = np.sum((np.linalg.pinv(X) @ Y - w) ** 2)
    return out.mean()


def minnorm_risk_theory(gamma, sigma=1.0, beta_norm2=1.0):
    """The asymptotic excess risk, Hastie et al. (2022), Thm. 1.

        gamma < 1 :  sigma^2 gamma / (1 - gamma)
        gamma > 1 :  ||w||^2 (1 - 1/gamma) + sigma^2 / (gamma - 1)

    Both branches diverge as gamma -> 1, which is the peak of the
    double-descent curve.
    """
    g = np.asarray(gamma, dtype=float)
    lo = sigma ** 2 * g / (1.0 - g)
    hi = beta_norm2 * (1.0 - 1.0 / g) + sigma ** 2 / (g - 1.0)
    return np.where(g < 1.0, lo, hi)


def gd_path_risk(n, p, steps, eta, rng, sigma=1.0, beta_norm2=1.0):
    """Excess risk along the gradient-descent path for the same model.

    Gradient descent on the least-squares objective from w = 0 converges
    to the minimum-norm solution, and the iterate at step t is a filtered
    version of it. Tracking ||w_t - w||^2 therefore gives the *epoch-wise*
    view of the same phenomenon: effective capacity grows with t.
    """
    w = np.zeros(p)
    w[0] = np.sqrt(beta_norm2)
    X = rng.normal(size=(n, p))
    Y = X @ w + sigma * rng.normal(size=n)
    wt = np.zeros(p)
    H = X.T @ X / n
    g0 = X.T @ Y / n
    out = np.empty(steps + 1)
    out[0] = np.sum((wt - w) ** 2)
    for t in range(steps):
        wt = wt - eta * (H @ wt - g0)
        out[t + 1] = np.sum((wt - w) ** 2)
    return out

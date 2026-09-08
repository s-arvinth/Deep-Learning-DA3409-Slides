"""The objects Class 18 regularises.

1. A 1-D regression problem in a fixed basis, so a penalty can be added
   to a least-squares fit and its effect seen as a curve.

2. A two-dimensional quadratic error, so weight decay can be drawn as a
   competition between two sets of contours, following the geometry of
   Bishop & Bishop (2024), Fig. 9.3.

3. A two-dimensional loss with a curved *valley* of global minima, so
   that gradient descent has something to choose between and its
   implicit preference becomes visible. Every point of the valley has
   the same loss, so nothing in L distinguishes them: only the algorithm
   can. This is the setting of Prince (2023), Fig. 9.3.

   The modified loss of Prince (2023), Eq. 9.8 is implemented with its
   exact gradient, so its prediction can be *checked* rather than
   asserted:

       grad L~  =  grad L  +  (alpha/2) H grad L ,

   which is the gradient of  L + (alpha/4) ||grad L||^2 .
"""
import numpy as np

SIGMA = 0.20


# ------------------------------------------------------ 1-D regression
def truth(x):
    return np.sin(2.0 * np.pi * x) + 0.4 * np.sin(6.0 * np.pi * x)


def sample(n, rng, sigma=SIGMA):
    x = np.sort(rng.uniform(0.0, 1.0, size=n))
    return x, truth(x) + sigma * rng.normal(size=n)


def design(x, K, width=0.09):
    """K Gaussian bumps of FIXED width, plus a constant column."""
    c = np.linspace(-0.03, 1.03, K)
    Phi = np.exp(-0.5 * ((x[:, None] - c[None, :]) / width) ** 2)
    return np.hstack([np.ones((len(x), 1)), Phi])


def ridge_fit(x, y, K, lam, grid):
    """Least squares with an L2 penalty on the weights but not the bias."""
    Phi = design(x, K)
    P = np.eye(Phi.shape[1]); P[0, 0] = 0.0
    w = np.linalg.solve(Phi.T @ Phi + lam * P, Phi.T @ y)
    return design(grid, K) @ w, w


# -------------------------------------------- 2-D quadratic error
def quad_E(W1, W2, lam=(0.30, 3.2), wstar=(1.55, 1.05)):
    """A quadratic error with Hessian eigenvalues `lam`, centred at wstar."""
    return 0.5 * (lam[0] * (W1 - wstar[0]) ** 2
                  + lam[1] * (W2 - wstar[1]) ** 2)


def ridge_solution(wstar, lam, alpha):
    """Minimiser of  1/2 sum lam_i (w_i - w*_i)^2 + (alpha/2)||w||^2.

    Componentwise  w_i = lam_i/(lam_i + alpha) * w*_i, so every
    eigen-direction is shrunk by its own factor and the flat directions
    are shrunk hardest.
    """
    lam = np.asarray(lam, float)
    return lam / (lam + alpha) * np.asarray(wstar, float)


def effective_parameters(lam, alpha):
    """sum_i lam_i / (lam_i + alpha): how many directions survive."""
    lam = np.asarray(lam, float)
    return float(np.sum(lam / (lam + alpha)))


# ---------------------------------------- 2-D loss with a valley of minima
KAP, AMP, FRQ = 4.0, 0.55, 1.7


def valley(x):
    return AMP * np.sin(FRQ * x)


def _vp(x):
    return AMP * FRQ * np.cos(FRQ * x)


def _vpp(x):
    return -AMP * FRQ * FRQ * np.sin(FRQ * x)


def L(W1, W2):
    return 0.5 * KAP * (W2 - valley(W1)) ** 2


def grad_L(w):
    d = w[1] - valley(w[0])
    return np.array([-KAP * d * _vp(w[0]), KAP * d])


def hess_L(w):
    x = w[0]; d = w[1] - valley(x)
    return np.array([[KAP * (_vp(x) ** 2 - d * _vpp(x)), -KAP * _vp(x)],
                     [-KAP * _vp(x), KAP]])


def grad_L_mod(w, alpha):
    """Exact gradient of  L + (alpha/4)||grad L||^2 ."""
    g = grad_L(w)
    return g + 0.5 * alpha * (hess_L(w) @ g)


def discrete_gd(w0, alpha, steps):
    w = np.asarray(w0, float).copy()
    path = [w.copy()]
    for _ in range(steps):
        w = w - alpha * grad_L(w)
        path.append(w.copy())
    return np.array(path)


def flow(w0, gradfun, total, n=20000, keep=False):
    """Gradient flow, integrated with RK4 so the integrator is not the
    thing being measured."""
    w = np.asarray(w0, float).copy()
    h = total / n
    path = [w.copy()]
    for _ in range(n):
        k1 = -gradfun(w)
        k2 = -gradfun(w + 0.5 * h * k1)
        k3 = -gradfun(w + 0.5 * h * k2)
        k4 = -gradfun(w + h * k3)
        w = w + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        if keep:
            path.append(w.copy())
    return np.array(path) if keep else w


# ------------------------------------------------ batch-gradient variance
def batch_grads(w, B=12, spread=0.40):
    """B batch gradients: each batch sees the valley slightly displaced.

    The batches disagree where the valley is steeply curved and agree
    where it is flat, which is the structure the SGD term of
    Prince (2023), Eq. 9.9 measures.
    """
    out = np.empty((B, 2))
    for b in range(B):
        off = spread * np.sin(2.399 * b + 0.7)
        d = w[1] - valley(w[0] + off)
        out[b] = [-KAP * d * _vp(w[0] + off), KAP * d]
    return out


def batch_variance(w, B=12, spread=0.40):
    G = batch_grads(w, B, spread)
    g = G.mean(axis=0)
    return float(np.mean(np.sum((G - g) ** 2, axis=1)))


# ---------------------------------- 2-D quadratic with a ROTATED Hessian
def rotated_hessian(lam, theta):
    """H = R diag(lam) R^T: eigenvalues `lam`, eigenvectors at angle theta."""
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return R @ np.diag(np.asarray(lam, float)) @ R.T, R


def quad_E_rot(W1, W2, H, wstar):
    """1/2 (w - w*)^T H (w - w*) on a grid."""
    D1 = W1 - wstar[0]; D2 = W2 - wstar[1]
    return 0.5 * (H[0, 0] * D1 ** 2 + 2 * H[0, 1] * D1 * D2 + H[1, 1] * D2 ** 2)


def ridge_solution_rot(H, wstar, alpha):
    """Minimiser of 1/2 (w-w*)^T H (w-w*) + (alpha/2)||w||^2 :
    (H + alpha I)^{-1} H w*.  In the eigenbasis this is the shrinkage
    lam_i/(lam_i+alpha) of Theorem 3, and the caller can check that."""
    return np.linalg.solve(H + alpha * np.eye(2), H @ np.asarray(wstar, float))


# ---------------------- a two-parameter model with several local minima
# The model  yhat(x) = sin(w0 x) + 0.5 sin(w1 x)  fitted by least squares
# to data generated from itself at w* = (3, 7).  Fitting a frequency is
# non-convex: aliases and swapped roles of the two terms give the loss
# many local minima, which is the setting of Prince (2023), Fig. 9.1.
MM_WSTAR = np.array([3.0, 7.0])
MM_N, MM_SIGMA, MM_XMAX = 40, 0.15, 1.2


def mm_data(rng=None):
    rng = np.random.default_rng(3) if rng is None else rng
    x = np.sort(rng.uniform(0.0, MM_XMAX, size=MM_N))
    y = mm_model(x, MM_WSTAR) + MM_SIGMA * rng.normal(size=MM_N)
    return x, y


def mm_model(x, w):
    return np.sin(w[0] * x) + 0.5 * np.sin(w[1] * x)


def mm_loss_grid(W0, W1, x, y):
    """Mean squared error on a (W0, W1) grid; vectorised over the data."""
    out = np.zeros_like(W0)
    for xn, yn in zip(x, y):
        out += (np.sin(W0 * xn) + 0.5 * np.sin(W1 * xn) - yn) ** 2
    return out / len(x)


def mm_loss(w, x, y):
    return float(np.mean((mm_model(x, w) - y) ** 2))


def local_minima(F, g0, g1, refine, tol=0.15):
    """Local minima of a function given on a grid.

    A grid cell is a candidate if it is strictly lower than its eight
    neighbours; each candidate is then refined by `refine`, a callable
    doing a local descent from a start point, and near-duplicates are
    merged.  Returns a list of points.
    """
    cand = []
    for i in range(1, F.shape[0] - 1):
        for j in range(1, F.shape[1] - 1):
            nb = F[i-1:i+2, j-1:j+2].copy()
            v = nb[1, 1]
            nb[1, 1] = np.inf
            if v < nb.min():
                cand.append(np.array([g0[j], g1[i]]))
    pts = []
    for c in cand:
        p = refine(c)
        if not any(np.linalg.norm(p - q) < tol for q in pts):
            pts.append(p)
    return pts


def descend(fun, w0, steps=400, h=1e-4, lr=0.02):
    """Plain gradient descent with a central-difference gradient."""
    w = np.asarray(w0, float).copy()
    for _ in range(steps):
        g = np.array([(fun(w + h * e) - fun(w - h * e)) / (2 * h)
                      for e in np.eye(2)])
        step = lr * g
        n = np.linalg.norm(step)
        if n > 0.05:
            step *= 0.05 / n
        w = w - step
    return w

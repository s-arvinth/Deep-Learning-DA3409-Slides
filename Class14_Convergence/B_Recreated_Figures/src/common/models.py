"""Optimisers and test surfaces for the Class 14 figures.

Everything is written out explicitly so a student can read the update
rule off the code and match it line for line to the slide.

Notation is the frozen course convention:

    w        the parameter vector
    E(w)     the error function
    eta      the learning rate
    mu       the momentum parameter        (Bishop's mu, Prince's beta)
    lambda_i eigenvalues of the Hessian
    kappa    condition number, lambda_max / lambda_min

Prince writes the parameters phi, the learning rate alpha and the loss
L[.]; those are translated here, once, at the source.
"""
import numpy as np


# =====================================================================
#  A quadratic bowl with a prescribed condition number
# =====================================================================
def quad_E(W1, W2, lam=(1.0, 20.0), rot=0.0, centre=(0.0, 0.0)):
    """E(w) = 1/2 sum_i lambda_i alpha_i^2 in the eigen-coordinates."""
    c, s = np.cos(rot), np.sin(rot)
    d1, d2 = W1 - centre[0], W2 - centre[1]
    a1 = c * d1 + s * d2
    a2 = -s * d1 + c * d2
    return 0.5 * (lam[0] * a1 ** 2 + lam[1] * a2 ** 2)


def quad_H(lam=(1.0, 20.0), rot=0.0):
    c, s = np.cos(rot), np.sin(rot)
    R = np.array([[c, s], [-s, c]])
    return R.T @ np.diag(lam) @ R


def quad_grad(w, lam=(1.0, 20.0), rot=0.0, centre=(0.0, 0.0)):
    return quad_H(lam, rot) @ (np.asarray(w, float)
                               - np.asarray(centre, float))


def kappa(lam):
    """Condition number of the Hessian: the ratio of extreme curvatures."""
    return max(lam) / min(lam)


# =====================================================================
#  The optimisers, each in the notation of the slide that introduces it
# =====================================================================
def gd(grad, w0, eta, steps, noise=0.0, seed=0):
    """w <- w - eta grad E."""
    rng = np.random.default_rng(seed)
    w = np.asarray(w0, float).copy()
    path = [w.copy()]
    for _ in range(steps):
        g = grad(w) + noise * rng.standard_normal(w.shape)
        w = w - eta * g
        path.append(w.copy())
    return np.array(path)


def momentum(grad, w0, eta, mu, steps, noise=0.0, seed=0):
    """Bishop Eq. 7.31:  dw <- -eta grad E + mu dw;  w <- w + dw.

    Prince Eq. 6.11 writes the same idea as an exponential average,
    m <- beta m + (1-beta) g, w <- w - eta m, which rescales eta by
    (1-beta) but is otherwise identical.
    """
    rng = np.random.default_rng(seed)
    w = np.asarray(w0, float).copy()
    dw = np.zeros_like(w)
    path = [w.copy()]
    for _ in range(steps):
        g = grad(w) + noise * rng.standard_normal(w.shape)
        dw = -eta * g + mu * dw
        w = w + dw
        path.append(w.copy())
    return np.array(path)


def nesterov(grad, w0, eta, mu, steps, noise=0.0, seed=0):
    """Bishop Eq. 7.34: the gradient is taken at the *predicted* point.

        dw <- -eta grad E(w + mu dw_prev) + mu dw_prev
    """
    rng = np.random.default_rng(seed)
    w = np.asarray(w0, float).copy()
    dw = np.zeros_like(w)
    path = [w.copy()]
    for _ in range(steps):
        look = w + mu * dw
        g = grad(look) + noise * rng.standard_normal(w.shape)
        dw = -eta * g + mu * dw
        w = w + dw
        path.append(w.copy())
    return np.array(path)


def adagrad(grad, w0, eta, steps, eps=1e-8, noise=0.0, seed=0):
    """Bishop Eqs. 7.39--7.40: divide by the root of the running SUM."""
    rng = np.random.default_rng(seed)
    w = np.asarray(w0, float).copy()
    r = np.zeros_like(w)
    path = [w.copy()]
    for _ in range(steps):
        g = grad(w) + noise * rng.standard_normal(w.shape)
        r = r + g ** 2
        w = w - eta * g / (np.sqrt(r) + eps)
        path.append(w.copy())
    return np.array(path)


def rmsprop(grad, w0, eta, beta, steps, eps=1e-8, noise=0.0, seed=0):
    """Bishop Eqs. 7.41--7.42: the same, with an exponential average."""
    rng = np.random.default_rng(seed)
    w = np.asarray(w0, float).copy()
    r = np.zeros_like(w)
    path = [w.copy()]
    for _ in range(steps):
        g = grad(w) + noise * rng.standard_normal(w.shape)
        r = beta * r + (1.0 - beta) * g ** 2
        w = w - eta * g / (np.sqrt(r) + eps)
        path.append(w.copy())
    return np.array(path)


def signsgd(grad, w0, eta, steps, eps=1e-8, noise=0.0, seed=0):
    """Prince Eqs. 6.13--6.14: keep only the sign of each component."""
    rng = np.random.default_rng(seed)
    w = np.asarray(w0, float).copy()
    path = [w.copy()]
    for _ in range(steps):
        g = grad(w) + noise * rng.standard_normal(w.shape)
        w = w - eta * g / (np.sqrt(g ** 2) + eps)
        path.append(w.copy())
    return np.array(path)


def adam(grad, w0, eta, b1, b2, steps, eps=1e-8, noise=0.0, seed=0,
         correct=True):
    """Bishop Eqs. 7.43--7.47 / Prince Eqs. 6.15--6.17.

    `correct=False` drops the bias correction, which is what the
    bias-correction figure needs in order to show what it fixes.
    """
    rng = np.random.default_rng(seed)
    w = np.asarray(w0, float).copy()
    s = np.zeros_like(w)
    r = np.zeros_like(w)
    path = [w.copy()]
    for t in range(1, steps + 1):
        g = grad(w) + noise * rng.standard_normal(w.shape)
        s = b1 * s + (1.0 - b1) * g
        r = b2 * r + (1.0 - b2) * g ** 2
        if correct:
            sh = s / (1.0 - b1 ** t)
            rh = r / (1.0 - b2 ** t)
        else:
            sh, rh = s, r
        w = w - eta * sh / (np.sqrt(rh) + eps)
        path.append(w.copy())
    return np.array(path)


# =====================================================================
#  Learning-rate schedules  (Bishop Eqs. 7.36--7.38, plus two in use)
# =====================================================================
def sched_linear(t, eta0, etaK, K):
    """Bishop Eq. 7.36: linear decay over K steps, then held."""
    t = np.minimum(t, K)
    return (1.0 - t / K) * eta0 + (t / K) * etaK


def sched_power(t, eta0, s, c):
    """Bishop Eq. 7.37: eta0 (1 + t/s)^(-c)."""
    return eta0 * (1.0 + t / s) ** (-c)


def sched_exponential(t, eta0, s, c):
    """Bishop Eq. 7.38: eta0 c^(t/s)."""
    return eta0 * c ** (t / s)


def sched_cosine(t, eta0, T, etamin=0.0):
    """Cosine annealing (Loshchilov & Hutter, 2017)."""
    t = np.minimum(t, T)
    return etamin + 0.5 * (eta0 - etamin) * (1.0 + np.cos(np.pi * t / T))


def sched_warmup_cosine(t, eta0, T, warm, etamin=0.0):
    """A linear warm-up followed by cosine annealing."""
    t = np.asarray(t, float)
    out = np.where(t < warm,
                   eta0 * t / max(warm, 1),
                   sched_cosine(np.maximum(t - warm, 0), eta0,
                                max(T - warm, 1), etamin))
    return out


# =====================================================================
#  Optimal constants for a quadratic, used by the comparison figures
# =====================================================================
def gd_best(lam):
    """The eta that minimises the contraction factor of plain descent.

    eta = 2/(lambda_min + lambda_max), giving a factor
    (kappa - 1)/(kappa + 1) per step.
    """
    lo, hi = min(lam), max(lam)
    eta = 2.0 / (lo + hi)
    k = hi / lo
    return eta, (k - 1.0) / (k + 1.0)


def heavyball_best(lam):
    """Polyak's optimal heavy-ball constants for a quadratic.

    mu  = ((sqrt(kappa) - 1)/(sqrt(kappa) + 1))^2
    eta = 4 / (sqrt(lambda_max) + sqrt(lambda_min))^2

    The contraction factor becomes (sqrt(kappa)-1)/(sqrt(kappa)+1):
    the square root of the condition number replaces the condition
    number itself.
    """
    lo, hi = min(lam), max(lam)
    sk = np.sqrt(hi / lo)
    mu = ((sk - 1.0) / (sk + 1.0)) ** 2
    eta = 4.0 / (np.sqrt(hi) + np.sqrt(lo)) ** 2
    return eta, mu, (sk - 1.0) / (sk + 1.0)


def nesterov_best(lam):
    """Nesterov's constants for a smooth strongly convex quadratic."""
    lo, hi = min(lam), max(lam)
    sk = np.sqrt(hi / lo)
    return 1.0 / hi, (sk - 1.0) / (sk + 1.0)

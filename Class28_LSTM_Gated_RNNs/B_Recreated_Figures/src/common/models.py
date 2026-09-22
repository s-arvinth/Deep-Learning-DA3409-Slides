"""The objects Class 28 computes with.

Three recurrent cells on the tape of Class 27, in the notation of
Jurafsky & Martin (2026) Section 14.5 with Goodfellow's letters noted:

`RNNCell`    h_t = tanh(U h_{t-1} + W x_t + b)
`LSTMCell`   Jurafsky Eqs. 14.20-14.27 / Goodfellow Eqs. 10.40-10.44,
             forget-gate bias initialised to 1 (Gers et al. 2000;
             Jozefowicz et al. 2015)
`GRUCell`    Goodfellow Eqs. 10.45-10.47

`Recaller`   a model that reads a sequence and, at the last step, must
             emit the FIRST symbol: the whole task is carrying one fact
             across the sequence, which is what a gate is for
`train`, `accuracy_by_length`, `grad_by_lag`
             the experiments; `grad_by_lag` reads the gradient of the
             final loss with respect to every hidden state off the tape

`power_iteration`, `leaky_unit`, `cliff_loss`
             the three closed-form pictures of Goodfellow Sections 10.7,
             10.9.2 and 10.11.1, in NumPy
"""
import numpy as np
from . import tape as tp

V = 9                       # symbols 1-8, 0 unused


def _orth(rng, d):
    q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    return q


class RNNCell:
    kind = "RNN"

    def __init__(self, d_in, d, rng):
        s = 1.0 / np.sqrt(d)
        self.W = tp.T(rng.normal(size=(d_in, d)) * s); self.U = tp.T(_orth(rng, d))
        self.b = tp.T(np.zeros(d)); self.d = d
        self.params = [self.W, self.U, self.b]

    def init_state(self, B):
        return tp.T(np.zeros((B, self.d)))

    def step(self, x, h):
        return tp.tanh(x @ self.W + h @ self.U + self.b), {}


class LSTMCell:
    kind = "LSTM"

    def __init__(self, d_in, d, rng, forget_bias=1.0):
        s = 1.0 / np.sqrt(d)
        self.Wf = tp.T(rng.normal(size=(d_in, d)) * s); self.Uf = tp.T(_orth(rng, d)); self.bf = tp.T(np.full(d, forget_bias))
        self.Wi = tp.T(rng.normal(size=(d_in, d)) * s); self.Ui = tp.T(_orth(rng, d)); self.bi = tp.T(np.zeros(d))
        self.Wg = tp.T(rng.normal(size=(d_in, d)) * s); self.Ug = tp.T(_orth(rng, d)); self.bg = tp.T(np.zeros(d))
        self.Wo = tp.T(rng.normal(size=(d_in, d)) * s); self.Uo = tp.T(_orth(rng, d)); self.bo = tp.T(np.zeros(d))
        self.d = d
        self.params = [self.Wf, self.Uf, self.bf, self.Wi, self.Ui, self.bi,
                       self.Wg, self.Ug, self.bg, self.Wo, self.Uo, self.bo]

    def init_state(self, B):
        return (tp.T(np.zeros((B, self.d))), tp.T(np.zeros((B, self.d))))

    def step(self, x, state):
        h, c = state
        f = tp.sigmoid(h @ self.Uf + x @ self.Wf + self.bf)      # forget gate, Eq. 14.20
        i = tp.sigmoid(h @ self.Ui + x @ self.Wi + self.bi)      # add gate,    Eq. 14.23
        g = tp.tanh(h @ self.Ug + x @ self.Wg + self.bg)         # candidate,   Eq. 14.22
        o = tp.sigmoid(h @ self.Uo + x @ self.Wo + self.bo)      # output gate, Eq. 14.26
        c2 = c * f + g * i                                       # Eqs. 14.21, 14.24, 14.25
        h2 = o * tp.tanh(c2)                                     # Eq. 14.27
        return (h2, c2), dict(f=f.v, i=i.v, o=o.v)


class GRUCell:
    kind = "GRU"

    def __init__(self, d_in, d, rng):
        s = 1.0 / np.sqrt(d)
        self.Wu = tp.T(rng.normal(size=(d_in, d)) * s); self.Uu = tp.T(_orth(rng, d)); self.bu = tp.T(np.zeros(d))
        self.Wr = tp.T(rng.normal(size=(d_in, d)) * s); self.Ur = tp.T(_orth(rng, d)); self.br = tp.T(np.zeros(d))
        self.Wh = tp.T(rng.normal(size=(d_in, d)) * s); self.Uh = tp.T(_orth(rng, d)); self.bh = tp.T(np.zeros(d))
        self.d = d
        self.params = [self.Wu, self.Uu, self.bu, self.Wr, self.Ur, self.br, self.Wh, self.Uh, self.bh]

    def init_state(self, B):
        return tp.T(np.zeros((B, self.d)))

    def step(self, x, h):
        u = tp.sigmoid(h @ self.Uu + x @ self.Wu + self.bu)      # update gate, Eq. 10.46
        r = tp.sigmoid(h @ self.Ur + x @ self.Wr + self.br)      # reset gate,  Eq. 10.47
        cand = tp.tanh((r * h) @ self.Uh + x @ self.Wh + self.bh)
        h2 = u * h + (1.0 - u) * cand                            # Eq. 10.45
        return h2, dict(u=u.v, r=r.v)


class Recaller:
    """Embed the symbols, run the cell, read the last state through a
    softmax over the symbols; the target is the first symbol."""

    def __init__(self, cell_cls, d=32, seed=0, **kw):
        rng = np.random.default_rng(seed)
        self.E = tp.T(rng.normal(size=(V, d)) * 0.3)
        self.cell = cell_cls(d, d, rng, **kw)
        self.Vo = tp.T(rng.normal(size=(d, V)) / np.sqrt(d)); self.bo = tp.T(np.zeros(V))
        self.params = [self.E, self.Vo, self.bo] + self.cell.params

    def run(self, X):
        """Returns (list of hidden states, list of gate dicts, logits)."""
        B, T = X.shape
        state = self.cell.init_state(B)
        hs = []; gates = []
        for t in range(T):
            state, gt = self.cell.step(self.E[X[:, t]], state)
            h = state[0] if isinstance(state, tuple) else state
            hs.append(h); gates.append(gt)
        return hs, gates, h @ self.Vo + self.bo

    def loss(self, X, y):
        hs, _, logits = self.run(X)
        return tp.cross_entropy(logits, y), hs


def recall_batch(rng, T, B):
    X = rng.integers(1, V, size=(B, T))
    return X, X[:, 0].copy()


def train(model, steps, rng, lens=(4, 24), B=32, lr=3e-3, clip=5.0, log_every=500):
    opt = tp.Adam(model.params, lr=lr, clip=clip)
    hist = []
    for s in range(1, steps + 1):
        T = int(rng.integers(lens[0], lens[1] + 1))
        X, y = recall_batch(rng, T, B)
        L, _ = model.loss(X, y)
        opt.zero(); L.backward(); opt.step(); hist.append(float(L.v))
        if log_every and s % log_every == 0:
            print("  %s step %5d loss %.3f" % (model.cell.kind, s, np.mean(hist[-log_every:])))
    return hist


def accuracy_by_length(model, rng, lengths, B=200):
    out = {}
    for T in lengths:
        X, y = recall_batch(rng, T, B)
        _, _, logits = model.run(X)
        out[T] = float(np.mean(np.argmax(logits.v, axis=1) == y))
    return out


def grad_by_lag(model, rng, T=40, B=16):
    """||dL_T / dh_k|| averaged over the batch, as a function of the lag
    T - k, read from the tape after one backward pass."""
    X, y = recall_batch(rng, T, B)
    L, hs = model.loss(X, y)
    L.backward()
    norms = np.array([np.linalg.norm(h.g, axis=1).mean() for h in hs])
    lags = np.arange(T - 1, -1, -1)
    return lags, norms


# ------------------------------------------------ closed-form pictures
def power_iteration(rho, T, d=20, seed=0):
    """||W^t h_0|| for a random W scaled to spectral radius rho."""
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(d, d)); W *= rho / np.max(np.abs(np.linalg.eigvals(W)))
    h = rng.normal(size=d); h /= np.linalg.norm(h)
    out = [np.linalg.norm(h)]
    for _ in range(T):
        h = W @ h; out.append(np.linalg.norm(h))
    return np.array(out)


def linear_tanh_composition(depth, d=100, seed=0, n=400):
    """Goodfellow Fig. 10.15: a random linear-tanh layer composed `depth`
    times, seen along a random line through a d-dimensional state,
    projected to one output."""
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(d, d)) / np.sqrt(d) * 1.4
    v = rng.normal(size=d); v /= np.linalg.norm(v)
    p = rng.normal(size=d); p /= np.linalg.norm(p)
    xs = np.linspace(-60, 60, n)
    H = np.outer(xs, v)
    outs = []
    for k in range(depth):
        H = np.tanh(H @ W.T)
        outs.append(H @ p)
    return xs, outs


def leaky_unit(alpha, T=60):
    """Impulse response of mu_t = alpha mu_{t-1} + (1 - alpha) v_t."""
    mu = np.zeros(T); v = np.zeros(T); v[5] = 1.0
    for t in range(1, T):
        mu[t] = alpha * mu[t - 1] + (1 - alpha) * v[t]
    return mu


def cliff_loss(w, b, T=50, target=0.5):
    """A one-parameter-pair recurrent net h_t = w h_{t-1} + b for T steps
    from h_0 = 0, with squared error to `target` at the end; the cliff of
    Goodfellow Fig. 10.17."""
    h = 0.0
    for _ in range(T):
        h = w * h + b
    return 0.5 * (h - target) ** 2


def cliff_grad(w, b, T=50, target=0.5, eps=1e-6):
    gw = (cliff_loss(w + eps, b, T, target) - cliff_loss(w - eps, b, T, target)) / (2 * eps)
    gb = (cliff_loss(w, b + eps, T, target) - cliff_loss(w, b - eps, T, target)) / (2 * eps)
    return np.array([gw, gb])

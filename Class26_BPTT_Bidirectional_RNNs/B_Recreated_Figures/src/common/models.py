"""The objects Class 26 computes with.

`SimpleRNN`   the Elman network of Jurafsky & Martin (2026) Eqs. 14.1-14.3
              / Goodfellow et al. (2016) Eqs. 10.8-10.11, in Jurafsky's
              letters: h_t = tanh(U h_{t-1} + W x_t + b), o_t = V h_t + c,
              y_t = softmax(o_t), with a cross-entropy loss at every step
              or at the last step only.  Runs both on the tape of Class 27
              (so Adam and truncation are available) and in plain NumPy.

`bptt`        Goodfellow Eqs. 10.17-10.28 written out by hand in NumPy:
              the backward recursion for dL/dh_t and the parameter
              gradients as sums over time, returning the per-step terms
              too.  This is the theorem of the class; the tape and finite
              differences are the two independent checks on it.

`finite_difference`
              central differences on every parameter entry.

`Bidirectional`
              two SimpleRNNs read in opposite directions, states
              concatenated (Jurafsky Eqs. 14.16-14.18, Goodfellow Fig. 10.11).

`marker_batch`, `train`, `accuracy`
              the recall task: a marker symbol at the first position,
              fillers after it, the label is the marker; trained with
              full or truncated back-propagation through time.
"""
import time
import numpy as np
from . import tape as tp

NSYM = 12           # symbols 1..K are markers, K+1..NSYM-1 fillers
K = 4


def _orth(rng, d):
    q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    return q


class SimpleRNN:
    def __init__(self, d_in, d, d_out, seed=0, scale=1.0, tie_markers=False):
        rng = np.random.default_rng(seed)
        self.E = tp.T(rng.normal(size=(d_in, d)) * 0.5)          # embedding rows: one per symbol
        if tie_markers:                                          # markers start indistinguishable
            self.E.v[1:K + 1] = self.E.v[1]
        self.W = tp.T(rng.normal(size=(d, d)) / np.sqrt(d))      # input-to-hidden   (Jurafsky W)
        self.U = tp.T(_orth(rng, d) * scale)                     # hidden-to-hidden  (Jurafsky U)
        self.b = tp.T(np.zeros(d))
        self.V = tp.T(rng.normal(size=(d, d_out)) / np.sqrt(d))  # hidden-to-output  (Jurafsky V)
        self.c = tp.T(np.zeros(d_out))
        self.d = d
        self.params = [self.E, self.W, self.U, self.b, self.V, self.c]
        self.names = ["E", "W", "U", "b", "V", "c"]

    # ---- on the tape --------------------------------------------------
    def forward(self, X, truncate=None):
        """X: (B, T) symbols.  Returns hidden states and logits per step.
        truncate=k: the state entering step T-k is detached, so the
        backward pass stops there (truncated BPTT)."""
        B, T = X.shape
        h = tp.T(np.zeros((B, self.d)))
        hs, os = [], []
        for t in range(T):
            if truncate is not None and t == max(T - truncate, 0) and t > 0:
                h = tp.T(h.v.copy())                             # cut the graph here
            h = tp.tanh(self.E[X[:, t]] @ self.W + h @ self.U + self.b)
            hs.append(h); os.append(h @ self.V + self.c)
        return hs, os

    def loss(self, X, Y, every_step=True, truncate=None):
        hs, os = self.forward(X, truncate)
        if every_step:
            Z = tp.concat(os, axis=0); L = tp.cross_entropy(Z, Y.T.reshape(-1))
        else:
            L = tp.cross_entropy(os[-1], Y)
        return L, hs, os

    # ---- in plain NumPy: forward, then Eqs. 10.17-10.28 ----------------
    def forward_np(self, X):
        B, T = X.shape
        E, W, U, b, V, c = [p.v for p in self.params]
        h = np.zeros((B, self.d)); hs = [h]; os = []; ys = []
        for t in range(T):
            h = np.tanh(E[X[:, t]] @ W + h @ U + b)
            o = h @ V + c; z = o - o.max(axis=1, keepdims=True); y = np.exp(z); y /= y.sum(axis=1, keepdims=True)
            hs.append(h); os.append(o); ys.append(y)
        return hs, os, ys


def bptt(model, X, Y, every_step=True):
    """Back-propagation through time by hand, Goodfellow Eqs. 10.17-10.28
    in Jurafsky's letters, for the mean cross-entropy over the batch (and
    over the steps, if every_step).  Returns the gradients in the order
    of model.params and the per-step terms of the sums over t."""
    B, T = X.shape
    E, W, U, b, V, c = [p.v for p in model.params]
    hs, os, ys = model.forward_np(X)
    n = B * T if every_step else B
    # Eq. 10.18: grad on the outputs, y_hat - 1_{y}
    dO = []
    for t in range(T):
        if every_step:
            g = ys[t].copy(); g[np.arange(B), Y[:, t]] -= 1.0; dO.append(g / n)
        else:
            g = np.zeros_like(ys[t])
            if t == T - 1:
                g = ys[t].copy(); g[np.arange(B), Y] -= 1.0; g /= n
            dO.append(g)
    # Eqs. 10.19-10.21: backward recursion for grad on the hidden states
    dH = [None] * T
    dH[T - 1] = dO[T - 1] @ V.T
    for t in range(T - 2, -1, -1):
        dH[t] = (dH[t + 1] * (1 - hs[t + 2] ** 2)) @ U.T + dO[t] @ V.T
    # Eqs. 10.22-10.28: parameter gradients as sums over time
    per_step = dict(U=[], W=[], V=[], E=[])
    gE = np.zeros_like(E); gW = np.zeros_like(W); gU = np.zeros_like(U); gb = np.zeros_like(b)
    gV = np.zeros_like(V); gc = np.zeros_like(c)
    for t in range(T):
        dA = dH[t] * (1 - hs[t + 1] ** 2)              # diag(1 - h_t^2) grad_{h_t} L
        gc += dO[t].sum(axis=0)
        tV = hs[t + 1].T @ dO[t]; gV += tV; per_step["V"].append(np.linalg.norm(tV))
        gb += dA.sum(axis=0)
        tU = hs[t].T @ dA; gU += tU; per_step["U"].append(np.linalg.norm(tU))      # Eq. 10.26
        xe = E[X[:, t]]
        tW = xe.T @ dA; gW += tW; per_step["W"].append(np.linalg.norm(tW))         # Eq. 10.28
        tE = np.zeros_like(E); np.add.at(tE, X[:, t], dA @ W.T); gE += tE; per_step["E"].append(np.linalg.norm(tE))
    return [gE, gW, gU, gb, gV, gc], per_step, dH


def finite_difference(model, X, Y, every_step=True, eps=1e-5, max_entries=None, rng=None):
    """Central differences on (a sample of) every parameter entry.
    Returns, per parameter, the list of (index, fd value)."""
    out = []
    for p in model.params:
        idx = list(np.ndindex(p.v.shape))
        if max_entries is not None and len(idx) > max_entries:
            rng = rng or np.random.default_rng(0)
            idx = [idx[i] for i in rng.choice(len(idx), max_entries, replace=False)]
        vals = []
        for ij in idx:
            old = p.v[ij]
            p.v[ij] = old + eps; Lp = float(model.loss(X, Y, every_step)[0].v)
            p.v[ij] = old - eps; Lm = float(model.loss(X, Y, every_step)[0].v)
            p.v[ij] = old
            vals.append((ij, (Lp - Lm) / (2 * eps)))
        out.append(vals)
    return out


def tape_gradients(model, X, Y, every_step=True):
    L, _, _ = model.loss(X, Y, every_step)
    for p in model.params:
        p.g = None
    L.backward()
    return [p.g.copy() for p in model.params]


# ------------------------------------------------------ the recall task
def marker_batch(rng, T, B):
    """Marker 1..K at position 0, fillers K+1..NSYM-1 after it; the label
    is the marker, read at the last step."""
    X = rng.integers(K + 1, NSYM, size=(B, T))
    y = rng.integers(0, K, size=B)
    X[:, 0] = y + 1
    return X, y


def train(model, steps, rng, T, B=32, lr=3e-3, truncate=None, log_every=0):
    opt = tp.Adam(model.params, lr=lr, clip=5.0)
    hist = []
    for s in range(1, steps + 1):
        X, y = marker_batch(rng, T, B)
        L, _, _ = model.loss(X, y, every_step=False, truncate=truncate)
        opt.zero(); L.backward(); opt.step(); hist.append(float(L.v))
        if log_every and s % log_every == 0:
            print("    step %4d loss %.3f" % (s, np.mean(hist[-log_every:])))
    return hist


def accuracy(model, rng, T, B=400):
    X, y = marker_batch(rng, T, B)
    _, os = model.forward(X)
    return float(np.mean(np.argmax(os[-1].v, axis=1) == y))


def marker_gradient(model, rng, T, truncate, B=32):
    """||dL/dE[marker rows]|| with the backward pass truncated to the
    last `truncate` steps: the gradient that reaches the first input."""
    X, y = marker_batch(rng, T, B)
    L, _, _ = model.loss(X, y, every_step=False, truncate=truncate)
    for p in model.params:
        p.g = None
    L.backward()
    g = model.E.g
    return float(np.linalg.norm(g[1:K + 1])) if g is not None else 0.0


# ------------------------------------------------------ cost against tau
def timing(d=32, B=16, taus=(10, 20, 40, 80, 160, 320), repeats=3, seed=0):
    """Wall-clock seconds of the forward and backward passes against the
    sequence length, and the number of stored states."""
    model = SimpleRNN(NSYM, d, K, seed=seed)
    rng = np.random.default_rng(seed)
    out = []
    for T in taus:
        X, _ = marker_batch(rng, T, B); y = rng.integers(0, K, size=(B, T))
        tf = tb = 0.0
        for _ in range(repeats):
            t0 = time.perf_counter(); L, hs, _ = model.loss(X, y, every_step=True); t1 = time.perf_counter()
            for p in model.params:
                p.g = None
            L.backward(); t2 = time.perf_counter()
            tf += (t1 - t0) / repeats; tb += (t2 - t1) / repeats
        out.append(dict(T=T, forward=tf, backward=tb, stored=len(hs)))
    return out


# ------------------------------------------------------ bidirectional
class Bidirectional:
    """Two SimpleRNNs, one read forward and one backward, states
    concatenated at every step and read through one output matrix."""

    def __init__(self, d_in, d, d_out, seed=0):
        rng = np.random.default_rng(seed)
        self.fwd = SimpleRNN(d_in, d, d_out, seed=seed)
        self.bwd = SimpleRNN(d_in, d, d_out, seed=seed + 1)
        self.V = tp.T(rng.normal(size=(2 * d, d_out)) / np.sqrt(2 * d)); self.c = tp.T(np.zeros(d_out))

    def forward(self, X):
        B, T = X.shape
        hf = []; h = tp.T(np.zeros((B, self.fwd.d)))
        for t in range(T):
            h = tp.tanh(self.fwd.E[X[:, t]] @ self.fwd.W + h @ self.fwd.U + self.fwd.b); hf.append(h)
        hb = [None] * T; h = tp.T(np.zeros((B, self.bwd.d)))
        for t in range(T - 1, -1, -1):
            h = tp.tanh(self.bwd.E[X[:, t]] @ self.bwd.W + h @ self.bwd.U + self.bwd.b); hb[t] = h
        hs = [tp.concat([f, b], axis=1) for f, b in zip(hf, hb)]
        return hs, [h @ self.V + self.c for h in hs]


def influence_map(model, X, embed_of):
    """||d o_t / d e_s|| for every output step t and input step s, where
    e_s is the embedding fed at step s: the causal structure of the model."""
    B, T = X.shape
    M = np.zeros((T, T))
    for t in range(T):
        # rebuild the graph with the input embeddings as leaves
        leaves = [tp.T(embed_of(X[:, s])) for s in range(T)]
        os = model.forward_from_embeddings(leaves)
        s_ = os[t].sum()
        for lf in leaves:
            lf.g = None
        s_.backward()
        for s in range(T):
            M[t, s] = np.linalg.norm(leaves[s].g) if leaves[s].g is not None else 0.0
    return M


def _uni_from_embeddings(self, leaves):
    B = leaves[0].v.shape[0]; h = tp.T(np.zeros((B, self.d))); os = []
    for e in leaves:
        h = tp.tanh(e @ self.W + h @ self.U + self.b); os.append(h @ self.V + self.c)
    return os


def _bi_from_embeddings(self, leaves):
    B = leaves[0].v.shape[0]; T = len(leaves)
    hf = []; h = tp.T(np.zeros((B, self.fwd.d)))
    for e in leaves:
        h = tp.tanh(e @ self.fwd.W + h @ self.fwd.U + self.fwd.b); hf.append(h)
    hb = [None] * T; h = tp.T(np.zeros((B, self.bwd.d)))
    for t in range(T - 1, -1, -1):
        h = tp.tanh(leaves[t] @ self.bwd.W + h @ self.bwd.U + self.bwd.b); hb[t] = h
    return [tp.concat([f, b], axis=1) @ self.V + self.c for f, b in zip(hf, hb)]


SimpleRNN.forward_from_embeddings = _uni_from_embeddings
Bidirectional.forward_from_embeddings = _bi_from_embeddings

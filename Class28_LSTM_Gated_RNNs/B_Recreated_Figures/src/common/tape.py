"""A minimal reverse-mode automatic differentiation tape in NumPy.

The sandbox has no PyTorch, and Class 11 taught backpropagation as a
graph of local Jacobians, so the training runs in this block use a
tape of exactly that kind: every operation records its inputs and a
function that turns the gradient of its output into gradients of its
inputs.  `backward()` walks the tape in reverse.  It is small enough
to read in one sitting, and every experiment in Classes 27-29 rests on
it, so it is checked against finite differences on import.
"""
import numpy as np


class T:
    __slots__ = ("v", "g", "parents", "bw", "name")

    def __init__(self, v, parents=(), bw=None, name=""):
        self.v = np.asarray(v, dtype=float)
        self.g = None
        self.parents = parents
        self.bw = bw
        self.name = name

    @property
    def shape(self):
        return self.v.shape

    # ---- arithmetic
    def __add__(self, o):
        o = _t(o)
        return T(self.v + o.v, (self, o),
                 lambda g: (_unb(g, self.v.shape), _unb(g, o.v.shape)))

    __radd__ = __add__

    def __sub__(self, o):
        o = _t(o)
        return T(self.v - o.v, (self, o),
                 lambda g: (_unb(g, self.v.shape), _unb(-g, o.v.shape)))

    def __rsub__(self, o):
        return _t(o) - self

    def __mul__(self, o):
        o = _t(o)
        return T(self.v * o.v, (self, o),
                 lambda g: (_unb(g * o.v, self.v.shape), _unb(g * self.v, o.v.shape)))

    __rmul__ = __mul__

    def __neg__(self):
        return T(-self.v, (self,), lambda g: (-g,))

    def __matmul__(self, o):
        o = _t(o)
        return T(self.v @ o.v, (self, o),
                 lambda g: (g @ o.v.T if o.v.ndim == 2 else np.outer(g, o.v),
                            self.v.T @ g if self.v.ndim == 2 else np.outer(self.v, g)))

    def __getitem__(self, idx):
        def bw(g):
            out = np.zeros_like(self.v); np.add.at(out, idx, g); return (out,)
        return T(self.v[idx], (self,), bw)

    def sum(self, axis=None):
        def bw(g):
            if axis is None:
                return (np.broadcast_to(g, self.v.shape).copy(),)
            return (np.broadcast_to(np.expand_dims(g, axis), self.v.shape).copy(),)
        return T(self.v.sum(axis=axis), (self,), bw)

    def mean(self, axis=None):
        n = self.v.size if axis is None else self.v.shape[axis]
        return self.sum(axis) * (1.0 / n)

    def T_(self):
        return T(self.v.T, (self,), lambda g: (g.T,))

    # ---- backward
    def backward(self):
        order = []; seen = set()

        def visit(n):
            if id(n) in seen:
                return
            seen.add(id(n))
            for p in n.parents:
                visit(p)
            order.append(n)
        visit(self)
        for n in order:
            n.g = np.zeros_like(n.v)
        self.g = np.ones_like(self.v)
        for n in reversed(order):
            if n.bw is None:
                continue
            for p, pg in zip(n.parents, n.bw(n.g)):
                p.g = p.g + pg


def _t(x):
    return x if isinstance(x, T) else T(x)


def _unb(g, shape):
    """Undo broadcasting: sum `g` down to `shape`."""
    g = np.asarray(g)
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and g.shape[i] != 1:
            g = g.sum(axis=i, keepdims=True)
    return g


# ---- functions
def tanh(x):
    y = np.tanh(x.v)
    return T(y, (x,), lambda g: (g * (1 - y * y),))


def sigmoid(x):
    y = 1 / (1 + np.exp(-x.v))
    return T(y, (x,), lambda g: (g * y * (1 - y),))


def relu(x):
    m = (x.v > 0).astype(float)
    return T(x.v * m, (x,), lambda g: (g * m,))


def concat(xs, axis=0):
    vs = [x.v for x in xs]; out = np.concatenate(vs, axis=axis)
    sizes = [v.shape[axis] for v in vs]

    def bw(g):
        outs = []; k = 0
        for s in sizes:
            sl = [slice(None)] * g.ndim; sl[axis] = slice(k, k + s)
            outs.append(g[tuple(sl)]); k += s
        return tuple(outs)
    return T(out, tuple(xs), bw)


def stack(xs, axis=0):
    out = np.stack([x.v for x in xs], axis=axis)
    return T(out, tuple(xs), lambda g: tuple(np.take(g, i, axis=axis) for i in range(len(xs))))


def softmax(x, axis=-1):
    z = x.v - x.v.max(axis=axis, keepdims=True)
    e = np.exp(z); p = e / e.sum(axis=axis, keepdims=True)
    return T(p, (x,), lambda g: (p * (g - (g * p).sum(axis=axis, keepdims=True)),))


def log_softmax(x, axis=-1):
    z = x.v - x.v.max(axis=axis, keepdims=True)
    lse = np.log(np.exp(z).sum(axis=axis, keepdims=True))
    p = np.exp(z - lse)
    return T(z - lse, (x,), lambda g: (g - p * g.sum(axis=axis, keepdims=True),))


def cross_entropy(logits, target, axis=-1):
    """Mean over rows of -log softmax(logits)[target]; `target` is an
    integer array with one entry per row."""
    ls = log_softmax(logits, axis)
    rows = np.arange(ls.v.shape[0])
    picked = ls[rows, target] if ls.v.ndim == 2 else ls[target]
    return -picked.mean()


# ---- an optimiser
class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, clip=None):
        self.p = params; self.lr = lr; self.b1 = b1; self.b2 = b2; self.eps = eps
        self.m = [np.zeros_like(p.v) for p in params]
        self.s = [np.zeros_like(p.v) for p in params]
        self.t = 0; self.clip = clip

    def step(self):
        self.t += 1
        gs = [p.g for p in self.p]
        if self.clip is not None:
            norm = np.sqrt(sum(float((g * g).sum()) for g in gs))
            if norm > self.clip:
                gs = [g * (self.clip / norm) for g in gs]
        for p, g, m, s in zip(self.p, gs, self.m, self.s):
            m *= self.b1; m += (1 - self.b1) * g
            s *= self.b2; s += (1 - self.b2) * g * g
            mh = m / (1 - self.b1 ** self.t); sh = s / (1 - self.b2 ** self.t)
            p.v -= self.lr * mh / (np.sqrt(sh) + self.eps)

    def zero(self):
        for p in self.p:
            p.g = None


# ---- checked on import: the tape against finite differences
def _check():
    rng = np.random.default_rng(0)
    W = T(rng.normal(size=(4, 3))); U = T(rng.normal(size=(4, 4)))
    x = rng.normal(size=(3,)); h0 = rng.normal(size=(4,))

    def f(W, U):
        h = tanh(W @ T(x) + U @ T(h0))
        h2 = tanh(U @ h + W @ T(x))
        z = concat([h, h2]) * T(np.linspace(0.5, 1.5, 8))
        return cross_entropy(z, 3)
    L = f(W, U); L.backward()
    eps = 1e-6; i, j = 2, 1
    Wp = T(W.v.copy()); Wp.v[i, j] += eps
    Wm = T(W.v.copy()); Wm.v[i, j] -= eps
    fd = (f(Wp, U).v - f(Wm, U).v) / (2 * eps)
    assert abs(fd - W.g[i, j]) < 1e-6, (fd, W.g[i, j])


_check()

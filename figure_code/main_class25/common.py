"""Shared code for the Class 25 figure scripts (Convolution Filters, Learned).

Plot style (Fira fonts from the repository's fonts/ folder, sci_style with
its fallback, the deck palette, the save helper that writes
output/<name>.pdf or --out DIR) and the objects the figures compute with
(2-D convolution, kernel composition, receptive-field counts), ported
unchanged from
Class25_Convolution_Filters_Learned/B_Recreated_Figures/src/common/{style,models}.py.

Needs: numpy, matplotlib  (optional: sci_style)
"""
import numpy as np
# ======================================================================
# Shared plotting style (ported from src/common/style.py)
# ======================================================================
import os, sys, glob, argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator, FixedLocator
from matplotlib.colors import ListedColormap

try:                                   # SciencePlots 'science' preset
    import sci_style                   # noqa: F401
except ImportError:                    # the fallback the figures were built with
    plt.rcParams.update({"axes.grid": False, "figure.facecolor": "white",
                         "axes.facecolor": "white"})

HERE = os.path.dirname(os.path.abspath(__file__))
FONTS = os.path.abspath(os.path.join(HERE, "..", "..", "fonts"))
for _f in sorted(glob.glob(os.path.join(FONTS, "*.ttf"))):
    try:
        font_manager.fontManager.addfont(_f)
    except Exception:
        pass

# ---- one palette, used by every deck ---------------------------------
L1   = "#341651"      # indigo    — series 1 / fixed features / layer 1
L2   = "#1C7293"      # turquoise — series 2 / activations
OUTC = "#800000"      # maroon    — output / emphasis
ACC  = "#B8860B"      # amber     — series 4
GREY = "#6E6E6E"      # neutral   — references, guides, truth curves
BRANCH = [L1, L2, OUTC, ACC]
SEQ    = "viridis"    # sequential colormap for surfaces and densities
CAT    = "plasma"     # categorical ramp for ordered families of curves

REGION_CMAP = ListedColormap([
    "#EFECF3", "#E1EAEF", "#F3E7E7", "#F6F1E5",
    "#E9EDE9", "#EDE6F0", "#E4EEF1", "#F8F4EC",
])

RC = {
    "text.usetex":      False,
    "font.family":      "Fira Sans",
    "mathtext.fontset": "custom",
    "mathtext.rm":      "Fira Sans",
    "mathtext.it":      "Fira Sans:italic",
    "mathtext.bf":      "Fira Sans:bold",
    "mathtext.sf":      "Fira Sans",
    "mathtext.tt":      "Fira Mono",
    "mathtext.cal":     "Fira Sans:italic",
    "mathtext.default": "regular",
    "font.size":        16.0,
    "axes.titlesize":   17.5,
    "axes.labelsize":   17.0,
    "xtick.labelsize":  14.5,
    "ytick.labelsize":  14.5,
    "legend.fontsize":  14.0,
    "axes.titlepad":    9.0,
    "axes.linewidth":   0.9,
    "xtick.major.size": 4.5,
    "ytick.major.size": 4.5,
    "xtick.major.width": 0.9,
    "ytick.major.width": 0.9,
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
    "xtick.major.pad":  4.0,
    "ytick.major.pad":  4.0,
    "legend.frameon":       True,
    "legend.framealpha":    0.92,
    "legend.facecolor":     "white",
    "legend.edgecolor":     "none",
    "legend.borderpad":     0.35,
    "legend.labelspacing":  0.30,
    "legend.handlelength":  1.5,
    "legend.handletextpad": 0.5,
    "legend.columnspacing": 0.9,
    "legend.borderaxespad": 0.4,
    "axes.xmargin":     0.04,
    "axes.ymargin":     0.08,
    "figure.dpi":       150,
    "pdf.fonttype":     42,
    "ps.fonttype":      42,
    "savefig.bbox":     "tight",
}


def use():
    """Apply the shared style. Call once at the top of a figure script."""
    plt.rcParams.update(RC)
    plt.rcParams["axes.prop_cycle"] = plt.cycler(
        color=[L1, L2, OUTC, ACC, GREY])


def thin_ticks(fig, nx=5, ny=5):
    """Cap the number of major ticks on every linear axis of `fig`."""
    for ax in fig.axes:
        if getattr(ax, "name", "") == "3d":
            continue
        if getattr(ax, "_colorbar", None) is not None:
            continue
        locked = getattr(ax, "_idl_locked", set())
        try:
            if ("x" not in locked
                    and ax.get_xscale() == "linear"
                    and not isinstance(ax.xaxis.get_major_locator(),
                                       FixedLocator)):
                ax.xaxis.set_major_locator(MaxNLocator(nbins=nx, prune=None))
            if ("y" not in locked
                    and ax.get_yscale() == "linear"
                    and not isinstance(ax.yaxis.get_major_locator(),
                                       FixedLocator)):
                ax.yaxis.set_major_locator(MaxNLocator(nbins=ny, prune=None))
        except Exception:
            pass


def out_dir():
    """output/ next to the scripts, or the directory given by --out DIR."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--out", default=os.path.join(HERE, "output"))
    args, _ = p.parse_known_args(sys.argv[1:])
    d = os.path.abspath(args.out)
    os.makedirs(d, exist_ok=True)
    return d


def save(fig, name, nx=5, ny=5):
    """Write `fig` to <out>/<name>.pdf as vector output (ticks thinned
    first, exactly as the original build did)."""
    thin_ticks(fig, nx, ny)
    path = os.path.join(out_dir(), name + ".pdf")
    fig.savefig(path, bbox_inches="tight", dpi=400,
                metadata={"CreationDate": None})
    plt.close(fig)
    print("wrote", path)
    return path


def square(ax):
    """Force a square data box — the preferred aspect for these decks."""
    ax.set_box_aspect(1.0)


def integer_ticks(ax, axis="x", nbins=6):
    """Whole-number ticks; marks the axis so save() leaves it alone."""
    loc = MaxNLocator(nbins=nbins, integer=True)
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_locator(loc)
    locked = getattr(ax, "_idl_locked", set())
    locked.add(axis)
    ax._idl_locked = locked


def tidy3d(ax, ticks=(0.0, 0.5, 1.0), labelsize=11):
    """Three ticks per axis on a 3-D panel, which is all that fits."""
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.set_ticks(list(ticks))
    ax.tick_params(labelsize=labelsize, pad=0.5)


# ======================================================================
# Models (ported from src/common/models.py)
# ======================================================================
# The objects Class 25 computes with.
#
# Every figure in the deck is an exact computation or a small experiment
# run by this code; nothing is traced from a book.
#
# `conv2d`, `relu`                 the layer, in two dimensions
# `stack_params`, `stack_rf`       parameters and receptive field of a
#                                  stack of k x k layers (Bishop Section
#                                  10.2.7)
# `compose`                        the single kernel equal to two kernels
#                                  in succession, when nothing nonlinear
#                                  sits between them
# `VGG16`, `vgg16_walk`            Bishop's Fig. 10.10 architecture with
#                                  exact parameters and connections
# `gabor`                          Bishop Eqs. 10.6-10.8
# `ToyNet`, `grad_cam`             a two-layer filter bank with a class
#                                  score, and Grad-CAM (Bishop Eqs.
#                                  10.9-10.10) computed by hand
# `make_shapes`, `train_logistic`,
# `fgsm`                           a two-class image problem, a linear
#                                  classifier, and the fast gradient sign
#                                  attack of Bishop Eq. 10.11
# `style_matrix`                   Bishop Eq. 10.15
# `synthetic_image`                the scene of Class 23
# ------------------------------------------------------------ the layer
def conv2d(X, W, same=True):
    """Cross-correlation of a 2-D array with a k x k kernel, zero-padded
    to the same size when `same`."""
    k = W.shape[0]; p = k // 2
    Xp = np.pad(X, p) if same else X
    n0 = Xp.shape[0] - k + 1; n1 = Xp.shape[1] - k + 1
    out = np.zeros((n0, n1))
    for a in range(k):
        for b in range(k):
            out += W[a, b] * Xp[a:a + n0, b:b + n1]
    return out


def relu(x):
    return np.maximum(x, 0.0)


# ----------------------------------------------- stacks of small kernels
def stack_params(k, depth, cin, cout):
    """Parameters of `depth` layers of k x k kernels, cin -> cout channels
    then cout -> cout: (k^2 c_in + 1) c_out per layer (Bishop 10.2.7)."""
    total = (k * k * cin + 1) * cout
    for _ in range(depth - 1):
        total += (k * k * cout + 1) * cout
    return total


def stack_rf(k, depth):
    """Receptive field of `depth` stride-one layers of k x k kernels."""
    return 1 + depth * (k - 1)


def compose(W1, W2):
    """The kernel of the composition: applying W1 then W2 (both
    cross-correlations, no nonlinearity) equals one cross-correlation
    with W = corr(W2, W1) of size k1 + k2 - 1."""
    k1, k2 = W1.shape[0], W2.shape[0]
    k = k1 + k2 - 1
    W = np.zeros((k, k))
    for a in range(k2):
        for b in range(k2):
            W[a:a + k1, b:b + k1] += W2[a, b] * W1
    return W


# --------------------------------------------------------- VGG-16
# Bishop Fig. 10.10 / Simonyan & Zisserman configuration D
VGG16 = ([("conv", 64)] * 2 + [("pool", None)]
         + [("conv", 128)] * 2 + [("pool", None)]
         + [("conv", 256)] * 3 + [("pool", None)]
         + [("conv", 512)] * 3 + [("pool", None)]
         + [("conv", 512)] * 3 + [("pool", None)]
         + [("fc", 4096), ("fc", 4096), ("fc", 1000)])


def vgg16_walk(side=224, cin=3, k=3):
    """Per learnable layer: kind, spatial side, channels, parameters
    (weights + biases) and connections (multiply-adds in one forward
    pass)."""
    out = []
    c, s = cin, side
    for kind, ch in VGG16:
        if kind == "conv":
            params = (k * k * c + 1) * ch
            conns = s * s * ch * k * k * c
            c = ch
            out.append(dict(kind=kind, side=s, channels=c, params=params,
                            conns=conns))
        elif kind == "pool":
            s //= 2
        else:
            n_in = s * s * c
            params = (n_in + 1) * ch
            conns = n_in * ch
            c, s = ch, 1
            out.append(dict(kind=kind, side=s, channels=c, params=params,
                            conns=conns))
    return out


# --------------------------------------------------------- Gabor filters
def gabor(n, theta, omega, phi=0.0, alpha=0.5, beta=0.5, A=1.0):
    """Bishop Eqs. 10.6-10.8 on an n x n grid centred at the origin,
    coordinates in [-1, 1]."""
    y, x = np.mgrid[-1:1:n * 1j, -1:1:n * 1j]
    xt = x * np.cos(theta) + y * np.sin(theta)
    yt = -x * np.sin(theta) + y * np.cos(theta)
    return A * np.exp(-alpha * xt ** 2 - beta * yt ** 2) * np.sin(omega * xt + phi)


# ---------------------------------------------------- a toy two-layer net
KERNELS = {
    "vertical edge":   np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float) / 4,
    "horizontal edge": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], float) / 4,
    "blur":            np.ones((3, 3)) / 9,
    "sharpen":         np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], float),
}


class ToyNet:
    """Layer 1: the four fixed kernels + ReLU.  Layer 2: one 3 x 3 kernel
    per input channel, summed, + ReLU, giving a single 'class' map A.
    Class score a = sum over positions of A.  Small enough that the
    gradient of a with respect to the layer-1 maps is written by hand,
    so Grad-CAM can be computed exactly."""

    def __init__(self, seed=25):
        rng = np.random.default_rng(seed)
        self.W1 = list(KERNELS.values())
        self.W2 = [rng.normal(size=(3, 3)) * 0.5 for _ in self.W1]

    def forward(self, X):
        A1 = [relu(conv2d(X, W)) for W in self.W1]              # layer-1 maps
        pre2 = sum(conv2d(a, W) for a, W in zip(A1, self.W2))
        A2 = relu(pre2)
        return A1, pre2, A2, float(A2.sum())

    def grad_wrt_layer1(self, X):
        """d a / d A1[k], with a = sum(A2): the ReLU mask of layer 2,
        cross-correlated back through W2[k] (the transpose operation)."""
        A1, pre2, A2, a = self.forward(X)
        mask = (pre2 > 0).astype(float)
        grads = []
        for W in self.W2:
            grads.append(conv2d(mask, W[::-1, ::-1]))          # transpose conv
        return A1, grads


def grad_cam(net, X):
    """Bishop Eqs. 10.9-10.10 at layer 1: alpha_k = mean gradient over
    the channel, L = sum_k alpha_k A^(k)."""
    A1, grads = net.grad_wrt_layer1(X)
    alphas = [g.mean() for g in grads]
    L = sum(al * A for al, A in zip(alphas, A1))
    return L, alphas


# ------------------------------------------------ a two-class image problem
def make_shapes(n, side=24, rng=None, noise=0.25):
    """Class 0: a filled disc.  Class 1: a filled square.  Near the
    centre, of random size, plus Gaussian pixel noise -- a problem a
    linear classifier on raw pixels can solve, which is the point."""
    rng = np.random.default_rng(0) if rng is None else rng
    X = np.empty((n, side * side)); Y = np.empty(n, int)
    yy, xx = np.mgrid[0:side, 0:side]
    for i in range(n):
        c = rng.integers(0, 2)
        cx, cy = side / 2 + rng.uniform(-1.5, 1.5, size=2); r = rng.uniform(5, 8)
        if c == 0:
            img = (((xx - cx) ** 2 + (yy - cy) ** 2) < r ** 2).astype(float)
        else:
            img = ((np.abs(xx - cx) < r) & (np.abs(yy - cy) < r)).astype(float)
        X[i] = (img + noise * rng.normal(size=img.shape)).ravel()
        Y[i] = c
    return X, Y


def train_logistic(X, Y, steps=3000, lr=0.05, lam=1e-3):
    """Logistic regression by gradient descent; returns (w, b)."""
    n, d = X.shape
    w = np.zeros(d); b = 0.0
    for _ in range(steps):
        p = 1 / (1 + np.exp(-(X @ w + b)))
        g = X.T @ (p - Y) / n + lam * w
        w -= lr * g; b -= lr * float(np.mean(p - Y))
    return w, b


def fgsm(X, Y, w, b, eps):
    """Bishop Eq. 10.11 for the logistic model: E is the negative log
    likelihood, whose gradient with respect to x is (p - t) w."""
    p = 1 / (1 + np.exp(-(X @ w + b)))
    grad = (p - Y)[:, None] * w[None, :]
    return X + eps * np.sign(grad)


def accuracy(X, Y, w, b):
    return float(np.mean(((X @ w + b) > 0).astype(int) == Y))


# ----------------------------------------------------------- style matrix
def style_matrix(maps):
    """Bishop Eq. 10.15: F_kk' = sum_ij a_ijk a_ijk'."""
    K = len(maps)
    F = np.empty((K, K))
    for k in range(K):
        for kp in range(K):
            F[k, kp] = float(np.sum(maps[k] * maps[kp]))
    return F


# ------------------------------------------------- the scene of Class 23
def synthetic_image(n=64):
    y, x = np.mgrid[0:n, 0:n] / (n - 1)
    img = np.full((n, n), 0.45)
    img[(x - 0.30) ** 2 + (y - 0.35) ** 2 < 0.045] = 0.90
    img[(np.abs(x - 0.72) < 0.14) & (np.abs(y - 0.68) < 0.14)] = 0.10
    img += 0.25 * np.clip((x - 0.55) / 0.45, 0, 1) * (y < 0.30)
    img[np.abs(y - 0.85) < 0.012] = 0.95
    return np.clip(img, 0, 1)


# ------------------------------------------------- checked on import
_rng = np.random.default_rng(1)
_X = _rng.normal(size=(12, 12)); _W1 = _rng.normal(size=(3, 3)); _W2 = _rng.normal(size=(3, 3))
_two = conv2d(conv2d(_X, _W1, same=False), _W2, same=False)
_one = conv2d(_X, compose(_W1, _W2), same=False)
assert np.allclose(_two, _one), "composition kernel is wrong"

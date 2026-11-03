"""Shared code for the Class 23 figure scripts (Convolution Filters).

Plot style (Fira fonts from the repository's fonts/ folder, sci_style with
its fallback, the deck palette, the save helper that writes
output/<name>.pdf or --out DIR) and the convolution operations the figures
compute with, ported unchanged from
Class23_Convolution_Filters/B_Recreated_Figures/src/common/{style,models}.py.

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
# The convolution operation (ported from src/common/models.py)
# ======================================================================
# The convolution operation, implemented once and used everywhere.
#
# Nothing in this class is statistical, so nothing here is random: every
# figure in the deck is an exact computation on an explicit array, and the
# numbers printed on the slides come from these functions.
#
# `conv1d`      the 1-D operation with stride, dilation and padding
# `conv_matrix` the same layer written out as a matrix, so it can be
#               compared with a fully connected one
# `out_size`    the output-length formula, checked against `conv1d`
# `rf_size`     the receptive-field recursion, checked against a measured
#               propagation of a single delta
# `conv2d`      the 2-D operation, for the kernel gallery
# `maxpool1d`, `upsample1d`  the two changes of resolution


# ------------------------------------------------------------ 1-D
def out_size(n, k, s=1, d=1, p=0):
    """Length of the output of a 1-D convolution.

        floor( (n + 2p - d(k-1) - 1) / s ) + 1
    """
    return (n + 2 * p - d * (k - 1) - 1) // s + 1


def conv1d(x, w, s=1, d=1, p=0):
    """Cross-correlation, which is what 'convolution' means in this field.

    Zero padding of width p, stride s, dilation d.
    """
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    k = len(w)
    xp = np.concatenate([np.zeros(p), x, np.zeros(p)])
    n_out = out_size(len(x), k, s, d, p)
    out = np.empty(n_out)
    for i in range(n_out):
        seg = xp[i * s: i * s + d * (k - 1) + 1: d]
        out[i] = float(seg @ w)
    return out


def conv_matrix(n, w, s=1, d=1, p=0):
    """The same layer as an explicit matrix, so that the convolution can
    be seen as a fully connected layer with tied and mostly-zero weights."""
    k = len(w)
    n_out = out_size(n, k, s, d, p)
    M = np.zeros((n_out, n))
    for i in range(n_out):
        for j in range(k):
            col = i * s + j * d - p
            if 0 <= col < n:
                M[i, col] = w[j]
    return M


# --------------------------------------------------- receptive fields
def rf_size(kernels, strides=None):
    """Receptive field of one output unit, layer by layer.

        r_0 = 1,   r_l = r_{l-1} + (k_l - 1) * prod_{j<l} s_j
    """
    L = len(kernels)
    strides = [1] * L if strides is None else list(strides)
    r, jump, out = 1, 1, []
    for k, s in zip(kernels, strides):
        r = r + (k - 1) * jump
        jump = jump * s
        out.append(r)
    return out


def measured_rf(kernels, strides=None, n=4001):
    """The receptive field, measured rather than computed.

    Push a single one through the stack with all-ones kernels and count
    how many input positions can reach the centre output unit.
    """
    L = len(kernels)
    strides = [1] * L if strides is None else list(strides)
    out = []
    for l in range(L):
        # a delta at the centre of the l-th output, traced back to the input
        support = np.zeros(n)
        support[n // 2] = 1.0
        for kk, ss in zip(reversed(kernels[:l + 1]), reversed(strides[:l + 1])):
            up = np.zeros(len(support) * ss)
            up[::ss] = support
            support = np.convolve(up, np.ones(kk), mode="full")
        out.append(int(np.count_nonzero(support)))
    return out


# ------------------------------------------------------------ 2-D
def conv2d(X, W, p=None):
    """2-D cross-correlation with 'same' zero padding by default."""
    X = np.asarray(X, float)
    W = np.asarray(W, float)
    kh, kw = W.shape
    if p is None:
        ph, pw = kh // 2, kw // 2
    else:
        ph, pw = p, p
    Xp = np.pad(X, ((ph, ph), (pw, pw)))
    H = Xp.shape[0] - kh + 1
    Wd = Xp.shape[1] - kw + 1
    out = np.empty((H, Wd))
    for i in range(H):
        for j in range(Wd):
            out[i, j] = float(np.sum(Xp[i:i + kh, j:j + kw] * W))
    return out


def synthetic_image(n=64):
    """A small greyscale scene: a disc, a bar, and a graded background.

    Built rather than photographed, so the deck owns it and every pixel
    is reproducible.
    """
    y, x = np.mgrid[0:n, 0:n] / (n - 1.0)
    img = 0.25 + 0.35 * x                       # a smooth gradient
    img += 0.30 * ((x - 0.30) ** 2 + (y - 0.32) ** 2 < 0.028)   # a disc
    img += 0.35 * ((np.abs(x - 0.72) < 0.07) & (y > 0.25) & (y < 0.85))
    img += 0.20 * (y > 0.86)                    # a bright strip
    return np.clip(img, 0.0, 1.0)


KERNELS = {
    "identity":  np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], float),
    "blur":      np.ones((3, 3)) / 9.0,
    "vertical edge":   np.array([[-1, 0, 1],
                                 [-2, 0, 2],
                                 [-1, 0, 1]], float),
    "horizontal edge": np.array([[-1, -2, -1],
                                 [0, 0, 0],
                                 [1, 2, 1]], float),
    "sharpen":   np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], float),
}


# ------------------------------------------------- changing resolution
def maxpool1d(x, k=2, s=None):
    s = k if s is None else s
    n_out = (len(x) - k) // s + 1
    return np.array([np.max(x[i * s: i * s + k]) for i in range(n_out)])


def upsample1d(x, factor=2, mode="duplicate"):
    if mode == "duplicate":
        return np.repeat(x, factor)
    if mode == "zeros":
        out = np.zeros(len(x) * factor)
        out[::factor] = x
        return out
    if mode == "linear":
        idx = np.arange(len(x) * factor) / factor
        return np.interp(idx, np.arange(len(x)), x)
    raise ValueError(mode)


# --------------------------------------------------- parameter counts
def fc_params(side, channels_in, hidden):
    """A fully connected layer from one image to `hidden` units."""
    return side * side * channels_in * hidden + hidden


def conv_params(k, channels_in, channels_out):
    """One convolutional layer, independent of the image size."""
    return k * k * channels_in * channels_out + channels_out

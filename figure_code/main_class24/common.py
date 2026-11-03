"""Shared code for the Class 24 figure scripts (Applications: Detection, Segmentation).

Plot style (Fira fonts from the repository's fonts/ folder, sci_style with
its fallback, the deck palette, the save helper that writes
output/<name>.pdf or --out DIR), the objects the figures compute with
(boxes, IoU, pooling/unpooling, transpose convolution) and the Bishop-style
drawing helpers, ported unchanged from
Class24_Applications_Detection_Segmentation/B_Recreated_Figures/src/common/
{style,models,bishopdraw}.py.

Needs: numpy, matplotlib  (optional: sci_style)
"""
import numpy as np
from matplotlib.patches import Polygon, Rectangle
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
# The objects Class 24 computes with.
#
# Nothing in this class is statistical, so nothing here is random: every
# figure is an exact computation, and every number on a slide comes from
# these functions.
#
# `iou`, `nms`                  the box metric and the suppression rule
#                               of Bishop & Bishop (2024), Sections 10.4.2
#                               and 10.4.5
# `window_cost`                 multiply-adds for the sliding-window
#                               network of Bishop Figs. 10.22-10.23, done
#                               naively and done once as a convolution
# `conv2d_matrix`,
# `transpose_conv`              a strided 2-D convolution as a matrix, and
#                               the up-sampling that is its transpose
# `avgpool`, `maxpool`,
# `unpool_avg`, `unpool_max`    the four resolution changes of Bishop
#                               Figs. 10.28-10.29
# `vgg19_spec`, `layer_params`  the published VGG-19 configuration and
#                               its exact parameter count
# `hourglass_spec`              the encoder-decoder schedule of Noh et al.
#                               (2015) as Prince (2023), Section 10.5.3
#                               describes it
# `synthetic_image`             the constructed scene of Class 23, reused
# ------------------------------------------------------------ boxes
def iou(a, b):
    """Intersection over union of two boxes given as (x, y, w, h) with
    (x, y) the centre, in any units."""
    ax0, ax1 = a[0] - a[2] / 2, a[0] + a[2] / 2
    ay0, ay1 = a[1] - a[3] / 2, a[1] + a[3] / 2
    bx0, bx1 = b[0] - b[2] / 2, b[0] + b[2] / 2
    by0, by1 = b[1] - b[3] / 2, b[1] + b[3] / 2
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def nms(boxes, scores, p_min=0.7, iou_max=0.5):
    """Non-max suppression as Bishop Section 10.4.5 states it, for one
    class: discard boxes below p_min, then repeatedly keep the most
    probable remaining box and discard every box whose IoU with it
    exceeds iou_max.  Returns the indices kept, in the order kept."""
    alive = [i for i in range(len(boxes)) if scores[i] >= p_min]
    kept = []
    while alive:
        best = max(alive, key=lambda i: scores[i])
        kept.append(best)
        alive = [i for i in alive
                 if i != best and iou(boxes[i], boxes[best]) <= iou_max]
    return kept


# ------------------------------------------- the sliding-window network
def window_cost(n_img, n_win=6, k=3, pool=2):
    """Multiply-adds to run Bishop's toy network (k x k convolution,
    non-overlapping pool x pool pooling, one fully connected output) on
    the n_win x n_win windows of an n_img x n_img image.

    Because the pooling is non-overlapping, the enlarged network of
    Fig. 10.23 evaluates windows at a stride of `pool`, so the naive
    count uses the same window positions.

    naive:   one full forward pass per window position
    shared:  one enlarged network over the whole image, whose last layer
             is the fully connected layer applied as a small convolution

    Returns (positions, naive, shared).  Pooling costs no multiply-adds.
    """
    c_win = n_win - k + 1                     # conv map side, one window
    p_win = c_win // pool                     # pooled side, one window
    per_window = c_win ** 2 * k ** 2 + p_win ** 2
    positions = ((n_img - n_win) // pool + 1) ** 2
    c_img = n_img - k + 1                     # conv map side, whole image
    p_img = c_img // pool
    fc_side = p_img - p_win + 1               # the FC layer as a p_win-kernel conv
    assert fc_side ** 2 == positions
    shared = c_img ** 2 * k ** 2 + fc_side ** 2 * p_win ** 2
    return positions, positions * per_window, shared


# --------------------------------------------- transpose convolution
def conv2d_matrix(n, W, s):
    """The n x n -> m x m strided convolution (no padding) as a matrix
    of shape (m*m, n*n)."""
    k = W.shape[0]
    m = (n - k) // s + 1
    M = np.zeros((m * m, n * n))
    for oi in range(m):
        for oj in range(m):
            for a in range(k):
                for b in range(k):
                    M[oi * m + oj, (oi * s + a) * n + (oj * s + b)] = W[a, b]
    return M, m


def transpose_conv(Z, W, s, n):
    """Up-sampling by scattering: every input unit multiplies the kernel
    into a k x k patch of the output, patches stepping by s, and
    overlaps are summed.  Returns the n x n output."""
    k = W.shape[0]
    out = np.zeros((n, n))
    m = Z.shape[0]
    for i in range(m):
        for j in range(m):
            out[i * s:i * s + k, j * s:j * s + k] += Z[i, j] * W
    return out


# --------------------------------------------------------- resolution
def avgpool(X, p=2):
    n = X.shape[0] // p
    return X[:n * p, :n * p].reshape(n, p, n, p).mean(axis=(1, 3))


def maxpool(X, p=2):
    n = X.shape[0] // p
    B = X[:n * p, :n * p].reshape(n, p, n, p)
    out = B.max(axis=(1, 3))
    where = np.zeros_like(X, bool)
    for i in range(n):
        for j in range(n):
            blk = X[i * p:(i + 1) * p, j * p:(j + 1) * p]
            a, b = np.unravel_index(np.argmax(blk), blk.shape)
            where[i * p + a, j * p + b] = True
    return out, where


def unpool_avg(Z, p=2):
    return np.kron(Z, np.ones((p, p)))


def unpool_max(Z, p=2, where=None):
    """Max-unpooling: the value goes to the first cell of its block, or
    to the remembered position if `where` is given."""
    n = Z.shape[0]
    out = np.zeros((n * p, n * p))
    for i in range(n):
        for j in range(n):
            if where is None:
                out[i * p, j * p] = Z[i, j]
            else:
                blk = where[i * p:(i + 1) * p, j * p:(j + 1) * p]
                a, b = np.argwhere(blk)[0]
                out[i * p + a, j * p + b] = Z[i, j]
    return out


def bilinear_up(X, f=2):
    """Bilinear up-sampling by a factor f, edges clamped."""
    n = X.shape[0]
    g = (np.arange(n * f) + 0.5) / f - 0.5
    g = np.clip(g, 0, n - 1)
    i0 = np.floor(g).astype(int); i1 = np.minimum(i0 + 1, n - 1); t = g - i0
    R = X[i0, :] * (1 - t)[:, None] + X[i1, :] * t[:, None]
    C = R[:, i0] * (1 - t)[None, :] + R[:, i1] * t[None, :]
    return C


# -------------------------------------------- published architectures
# VGG-19, Simonyan & Zisserman (2015), configuration E: (kind, out
# channels).  Kernel 3 x 3 everywhere; pooling halves the side.
VGG19 = ([("conv", 64)] * 2 + [("pool", None)]
         + [("conv", 128)] * 2 + [("pool", None)]
         + [("conv", 256)] * 4 + [("pool", None)]
         + [("conv", 512)] * 4 + [("pool", None)]
         + [("conv", 512)] * 4 + [("pool", None)]
         + [("fc", 4096), ("fc", 4096), ("fc", 1000)])


def vgg19_spec(side=224, cin=3):
    """Walk the configuration: returns a list of dicts with the layer
    kind, spatial side, channels and exact parameter count."""
    out = []
    c, s = cin, side
    for kind, ch in VGG19:
        if kind == "conv":
            params = 3 * 3 * c * ch + ch
            c = ch
        elif kind == "pool":
            s = s // 2
            params = 0
        else:
            n_in = s * s * c if not out or out[-1]["kind"] != "fc" else c
            params = n_in * ch + ch
            c, s = ch, 1
        out.append(dict(kind=kind, side=s, channels=c, params=params))
    return out


def hourglass_spec():
    """The encoder-decoder of Noh et al. (2015) as Prince Section 10.5.3
    gives it: a VGG encoder to 7 x 7, two fully connected layers of
    4096, a fully connected layer back to 7 x 7 x 512, then a mirror of
    the encoder up to 224 x 224 x 21.  Returns (labels, sides, chans)."""
    enc = [(224, 3), (224, 64), (112, 128), (56, 256), (28, 512), (14, 512),
           (7, 512)]
    mid = [(1, 4096), (1, 4096)]
    dec = [(7, 512), (14, 512), (28, 512), (56, 256), (112, 128), (224, 64),
           (224, 21)]
    seq = enc + mid + dec
    return [s for s, _ in seq], [c for _, c in seq], len(enc), len(mid)


# ------------------------------------------------- the scene of Class 23
def synthetic_image(n=64):
    """A constructed scene: a bright disc, a dark square, a soft ramp and
    a thin line, on a mid-grey ground."""
    y, x = np.mgrid[0:n, 0:n] / (n - 1)
    img = np.full((n, n), 0.45)
    img[(x - 0.30) ** 2 + (y - 0.35) ** 2 < 0.045] = 0.90
    img[(np.abs(x - 0.72) < 0.14) & (np.abs(y - 0.68) < 0.14)] = 0.10
    img += 0.25 * np.clip((x - 0.55) / 0.45, 0, 1) * (y < 0.30)
    img[np.abs(y - 0.85) < 0.012] = 0.95
    return np.clip(img, 0, 1)


# ------------------------------------------------- checked on import
_Z = np.array([[1.0, 2.0], [3.0, 4.0]])
assert np.allclose(avgpool(unpool_avg(_Z)), _Z)
assert np.allclose(maxpool(unpool_max(_Z))[0], _Z)


# ======================================================================
# Bishop-style drawing helpers (ported from src/common/bishopdraw.py)
# ======================================================================
# Drawing helpers in the style of Bishop & Bishop (2024), Figs. 10.22,
# 10.23, 10.28-10.30: sheared grids seen in perspective, and flat grids
# of numbered cells, in the book's four colours.
from matplotlib.patches import Polygon, Rectangle

RED, GREEN, BLUE, YELLOW, PURPLE = "#FF8080", "#80FF80", "#8080FF", "#FFFF80", "#C080C0"
WHITE = "white"
A, B, H = 0.40, 0.26, 0.62          # column step, rise per column, row height


def cell_corners(x0, y0, r, c):
    """Corners of cell (row r from the top, column c) of a sheared grid
    whose top-left corner is (x0, y0)."""
    x = x0 + c * A; y = y0 - r * H + c * B
    return [(x, y), (x + A, y + B), (x + A, y + B - H), (x, y - H)]


def draw_grid(ax, x0, y0, n, fills, default=YELLOW, lw=0.6, edge="black"):
    for r in range(n):
        for c in range(n):
            ax.add_patch(Polygon(cell_corners(x0, y0, r, c), closed=True,
                                 fc=fills.get((r, c), default), ec=edge, lw=lw,
                                 zorder=2))
    outer = [(x0, y0), (x0 + n * A, y0 + n * B), (x0 + n * A, y0 + n * B - n * H),
             (x0, y0 - n * H)]
    ax.add_patch(Polygon(outer, closed=True, fc="none", ec="black", lw=1.6, zorder=3))


def block_outline(ax, x0, y0, r0, c0, k, lw=1.6):
    """Heavy outline around a k x k block starting at (r0, c0); returns
    its four corners (tl, tr, br, bl)."""
    tl = cell_corners(x0, y0, r0, c0)[0]
    tr = cell_corners(x0, y0, r0, c0 + k - 1)[1]
    br = cell_corners(x0, y0, r0 + k - 1, c0 + k - 1)[2]
    bl = cell_corners(x0, y0, r0 + k - 1, c0)[3]
    ax.add_patch(Polygon([tl, tr, br, bl], closed=True, fc="none", ec="black",
                         lw=lw, zorder=4))
    return tl, tr, br, bl


def connect(ax, src, dst, color="black", lw=0.6):
    for (xa, ya), (xb, yb) in zip(src, dst):
        ax.plot([xa, xb], [ya, yb], color=color, lw=lw, zorder=1)


def flat_grid(ax, x0, y0, vals, colors, cell=1.0, fs=12, lw=1.0):
    """A flat grid of numbered square cells, top-left at (x0, y0)."""
    n0, n1 = vals.shape
    for i in range(n0):
        for j in range(n1):
            ax.add_patch(Rectangle((x0 + j * cell, y0 - (i + 1) * cell), cell, cell,
                                   fc=colors[i][j], ec="black", lw=lw, zorder=2))
            ax.text(x0 + (j + 0.5) * cell, y0 - (i + 0.5) * cell, "%d" % vals[i, j],
                    ha="center", va="center", fontsize=fs, zorder=3)
    ax.add_patch(Rectangle((x0, y0 - n0 * cell), n1 * cell, n0 * cell, fc="none",
                           ec="black", lw=2.0, zorder=3))
    return x0 + n1 * cell


def arrow(ax, x0, x1, y, lw=1.4):
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=lw))

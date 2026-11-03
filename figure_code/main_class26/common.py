"""Shared code for the Class 26 figure scripts (BPTT, Bidirectional RNNs).

Plot style (Fira fonts from the repository's fonts/ folder, sci_style with
its fallback, the deck palette, the save helper that writes
output/<name>.pdf or --out DIR) and the drawing helpers for unrolled
networks, ported unchanged from
Class26_BPTT_Bidirectional_RNNs/B_Recreated_Figures/src/common/{style,draw}.py.

The autodiff tape, the models and the cached experiments are in the
helper modules tape.py, models.py and experiment.py next to this file.

Needs: numpy, matplotlib  (optional: sci_style)
"""
import numpy as np
from matplotlib.patches import Rectangle, FancyBboxPatch
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
# Drawing helpers (ported from src/common/draw.py)
# ======================================================================
# Drawing helpers for unrolled networks in the layout of Jurafsky &
# Martin's Figs. 14.17-14.19: a row of embedding boxes, a row of hidden
# boxes, a row of softmax boxes, tokens below and outputs above, with
# the encoder in indigo and the decoder in maroon.
ENC, DEC = L1, OUTC
ENC_F, DEC_F = "#E4DDEB", "#F1D6D6"


def column(ax, x, y0, color, fill, rows=("emb", "hid", "soft"), dy=0.9, w=0.6, h=0.5):
    """Draw the boxes of one time step; returns their centres by row."""
    cs = {}
    for k, r in enumerate(rows):
        y = y0 + k * dy
        ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, fc=fill, ec=color, lw=1.2, zorder=3))
        cs[r] = (x, y)
    return cs


def arrow(ax, p, q, color="black", lw=1.0, ls="-"):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls,
                                                    shrinkA=2, shrinkB=2), zorder=4)


def band(ax, x0, x1, y0, y1, color, label=None, alpha=0.10):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0.15,rounding_size=0.25",
                                fc=color, ec="none", alpha=alpha, zorder=1))
    if label:
        ax.text((x0 + x1) / 2, y1 + 0.32, label, ha="center", va="bottom", color=color, fontsize=12)

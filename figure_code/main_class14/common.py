"""Shared plot style and save helper for the Class 14 figure scripts.

Ported from Class14_*/B_Recreated_Figures/src/common/style.py (identical to
_shared/deck_style.py). Registers the Fira faces from the repository's
root fonts/ folder, applies the SciencePlots "science" preset through
sci_style (or the fallback below), fixes the one deck palette, and writes
each figure to output/<name>.pdf, or to DIR with  --out DIR.

Needs: matplotlib  (optional: sci_style)
"""
import os, sys, glob
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MaxNLocator, FixedLocator

try:
    import sci_style                   # SciencePlots 'science' preset
except ImportError:
    # The figures on the Class 14 slides were built WITH the SciencePlots
    # "science" preset (inward ticks on all four sides, mathtext tick labels,
    # 0.05 in save padding); the plain block alone does not reproduce them.
    # So the fallback is the conventions' block plus the rcParams sci_style
    # applies (its font, mathtext-font and usetex keys are left out: RC in
    # use() overrides them anyway).
    from cycler import cycler
    plt.rcParams.update({"axes.grid": False, "figure.facecolor": "white",
                         "axes.facecolor": "white"})
    plt.rcParams.update({
        "axes.prop_cycle": cycler("color", ["#0C5DA5", "#00B945", "#FF9500",
                                            "#FF2C00", "#845B97", "#474747",
                                            "#9e9e9e"]),
        "figure.figsize": (3.5, 2.625),
        "xtick.direction": "in", "xtick.major.size": 3,
        "xtick.major.width": 0.5, "xtick.minor.size": 1.5,
        "xtick.minor.width": 0.5, "xtick.minor.visible": True,
        "xtick.top": True,
        "ytick.direction": "in", "ytick.major.size": 3,
        "ytick.major.width": 0.5, "ytick.minor.size": 1.5,
        "ytick.minor.width": 0.5, "ytick.minor.visible": True,
        "ytick.right": True,
        "axes.linewidth": 0.5, "grid.linewidth": 0.5, "lines.linewidth": 1.0,
        "axes.formatter.use_mathtext": True,
        "legend.frameon": False,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
    })

# ---- paths -------------------------------------------------------------
HERE  = os.path.dirname(os.path.abspath(__file__))
FONTS = os.path.abspath(os.path.join(HERE, "..", "..", "fonts"))

for _f in sorted(glob.glob(os.path.join(FONTS, "*.ttf"))):
    try:
        font_manager.fontManager.addfont(_f)
    except Exception:
        pass


def out_dir():
    """output/ next to this file, or the DIR given as  --out DIR."""
    d = os.path.join(HERE, "output")
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a == "--out" and i + 1 < len(argv):
            d = argv[i + 1]
        elif a.startswith("--out="):
            d = a.split("=", 1)[1]
    d = os.path.abspath(d)
    os.makedirs(d, exist_ok=True)
    return d


# ---- one palette, used by every deck -----------------------------------
L1   = "#341651"      # indigo    - series 1 / fixed features / layer 1
L2   = "#1C7293"      # turquoise - series 2 / activations
OUTC = "#800000"      # maroon    - output / emphasis
ACC  = "#B8860B"      # amber     - series 4
GREY = "#6E6E6E"      # neutral   - references, guides, truth curves
BRANCH = [L1, L2, OUTC, ACC]
SEQ    = "viridis"    # sequential colormap for surfaces and densities
CAT    = "plasma"     # categorical ramp for ordered families of curves

# muted tints for activation-pattern regions (deliberately not a rainbow)
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
    # type sizes: deliberately large, the figures are shrunk on the slide
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
    "pdf.fonttype":     42,      # embed TrueType, keep text selectable
    "ps.fonttype":      42,
    "savefig.bbox":     "tight",
}


def use():
    """Apply the shared style. Call once at the top of build()."""
    plt.rcParams.update(RC)
    plt.rcParams["axes.prop_cycle"] = plt.cycler(
        color=[L1, L2, OUTC, ACC, GREY])


def thin_ticks(fig, nx=5, ny=5):
    """Cap the number of major ticks on every linear axis of `fig`.

    Log axes, colourbars and axes locked by integer_ticks() are left alone.
    """
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


def save(fig, name, nx=5, ny=5):
    """Thin the ticks, then write `fig` to <out>/<name>.pdf (vector, 400 dpi
    for any rasterised layer), with no creation date so reruns are stable."""
    thin_ticks(fig, nx, ny)
    path = os.path.join(out_dir(), name + ".pdf")
    fig.savefig(path, bbox_inches="tight", dpi=400,
                metadata={"CreationDate": None})
    plt.close(fig)
    print("wrote", path)
    return path


def square(ax):
    """Force a square data box - the preferred aspect for these decks."""
    ax.set_box_aspect(1.0)


def integer_ticks(ax, axis="x", nbins=6):
    """Whole-number ticks; marks the axis so thin_ticks() leaves it alone."""
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

"""Shared plotting style for the Fundamentals of Deep Learning decks.

Loads the SciencePlots "science" preset, registers the bundled Fira Sans
faces, and fixes one palette so every figure in every deck matches.

Type sizes are deliberately large: the figures are shrunk when placed on
a slide, so what looks oversized here reads correctly from the back row.
"""
import os, glob
import sci_style                       # SciencePlots 'science' preset
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ---- locate the deck root (…/B_Recreated_Figures) --------------------
COMMON = os.path.dirname(os.path.abspath(__file__))
ROOT   = os.path.abspath(os.path.join(COMMON, "..", ".."))
FIGS   = os.path.join(ROOT, "figs")
FONTS  = os.path.join(ROOT, "fonts")
os.makedirs(FIGS, exist_ok=True)

for _f in glob.glob(os.path.join(FONTS, "*.ttf")):
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

# muted tints for activation-pattern regions (deliberately not a rainbow)
from matplotlib.colors import ListedColormap
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
    # --- type sizes: deliberately large.  Figures are scaled down a lot
    #     when placed on a slide, so what looks oversized in the PDF is
    #     what reads correctly from the back of a lecture theatre.
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
    "figure.dpi":       150,
    "pdf.fonttype":     42,      # embed TrueType, keep text selectable
    "ps.fonttype":      42,
    "savefig.bbox":     "tight",
}


def use():
    """Apply the shared style. Call once at the top of a figure module."""
    plt.rcParams.update(RC)
    plt.rcParams["axes.prop_cycle"] = plt.cycler(
        color=[L1, L2, OUTC, ACC, GREY])


def save(fig, name):
    """Write `fig` to ../figs/<name>.pdf as vector output."""
    path = os.path.join(FIGS, name + ".pdf")
    fig.savefig(path, bbox_inches="tight", dpi=400)
    plt.close(fig)
    print("wrote", os.path.relpath(path, ROOT))
    return path


def square(ax):
    """Force a square data box — the preferred aspect for these decks."""
    ax.set_box_aspect(1.0)

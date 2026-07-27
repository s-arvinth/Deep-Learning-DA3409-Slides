"""Shared figure style for all Fundamentals of Deep Learning decks.

Import AFTER `import sci_style`:

    import sci_style
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "..", "_shared"))
    from deck_style import *

Gives every deck the same Fira typography, palette and figure sizes.
"""
import os, glob
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import ListedColormap

# ---- Fira Sans everywhere -------------------------------------------
def use_fira(font_dir=None):
    """Register the bundled Fira faces and make matplotlib use them."""
    if font_dir is None:
        font_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "fonts")
    for f in glob.glob(os.path.join(font_dir, "*.ttf")):
        try:
            font_manager.fontManager.addfont(f)
        except Exception:
            pass
    plt.rcParams.update({
        "text.usetex":      False,
        "font.family":      "Fira Sans",
        "mathtext.fontset": "custom",
        "mathtext.rm":      "Fira Sans",
        "mathtext.it":      "Fira Sans:italic",
        "mathtext.bf":      "Fira Sans:bold",
        "mathtext.sf":      "Fira Sans",
        "mathtext.tt":      "Fira Mono",
        "mathtext.default": "regular",
        "font.size":        11.0,
        "axes.titlesize":   11.5,
        "axes.labelsize":   11.0,
        "xtick.labelsize":  9.5,
        "ytick.labelsize":  9.5,
        "legend.fontsize":  9.5,
        "figure.dpi":       150,
    })

# ---- one palette for every deck -------------------------------------
L1   = "#341651"      # layer 1 / network 1 / class A
L2   = "#1C7293"      # layer 2 / network 2 / activations
OUTC = "#800000"      # composition / deep result / class B
ACC  = "#B8860B"      # fourth series
GREY = "#6E6E6E"
BRANCH = [L1, L2, OUTC, ACC]

REGION_CMAP = ListedColormap([
    "#EFECF3", "#E1EAEF", "#F3E7E7", "#F6F1E5",
    "#E9EDE9", "#EDE6F0", "#E4EEF1", "#F8F4EC",
])

def apply_palette():
    plt.rcParams["axes.prop_cycle"] = plt.cycler(
        color=[L1, L2, OUTC, ACC, GREY])

def save(fig, name, outdir):
    os.makedirs(outdir, exist_ok=True)
    p = os.path.join(outdir, name + ".png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.basename(p))

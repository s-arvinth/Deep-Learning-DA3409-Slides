"""logic_2d — AND, OR and XOR in the 2-D input square, with separating lines

Used in: Class 2 (B), slide(s): "The same gates, drawn in input space"
Needs:   numpy, matplotlib  (optional: sci_style)
Run:     python logic_2d.py            -> output/logic_2d.pdf
"""
import numpy as np
import matplotlib.pyplot as plt

from common import L1, L2, OUTC, GREY, save

NAME = "logic_2d"


def build():
    pts = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    tables = {
        "AND": np.array([0, 0, 0, 1]),
        "OR":  np.array([0, 1, 1, 1]),
        "XOR": np.array([0, 1, 1, 0]),
    }
    # thresholds that realise AND and OR with unit weights
    lines = {"AND": 1.5, "OR": 0.5, "XOR": None}

    fig, ax = plt.subplots(1, 3, figsize=(9.6, 3.1))
    for a, (name, y) in zip(ax, tables.items()):
        a.scatter(*pts[y == 0].T, s=90, marker="o", facecolor="white",
                  edgecolor=L1, lw=1.6, zorder=3, label="output 0")
        a.scatter(*pts[y == 1].T, s=110, marker="^", facecolor=OUTC,
                  edgecolor="white", lw=1.0, zorder=3, label="output 1")
        th = lines[name]
        if th is not None:
            xs = np.linspace(-0.45, 1.45, 10)
            a.plot(xs, th - xs, color=L2, lw=1.6,
                   label=rf"$x_1+x_2=\theta$, $\theta={th}$")
        else:
            xs = np.linspace(-0.45, 1.45, 10)
            for th_bad in (0.5, 1.5):
                a.plot(xs, th_bad - xs, color=GREY, lw=0.9, ls="--",
                       alpha=.7)
            a.text(0.5, -0.33, "no single line separates",
                   color=GREY, fontsize=8.5, ha="center")
        a.set(xlim=(-0.45, 1.45), ylim=(-0.45, 1.45),
              xlabel=r"$x_1$", ylabel=r"$x_2$",
              title=f"({'abc'[list(tables).index(name)]})  {name}")
        a.set_xticks([0, 1]); a.set_yticks([0, 1])
        a.set_aspect("equal", "box")
    ax[0].legend(loc="upper right", fontsize=7.5)
    fig.tight_layout(w_pad=1.6)
    return save(fig, NAME)


if __name__ == "__main__":
    build()

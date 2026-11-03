"""xor — the four XOR points; no line separates the classes

Used in: Class 3 (B), slide(s): "The limit: XOR and non-separable data"
Needs:   numpy, matplotlib  (optional: sci_style)
Run:     python xor.py            -> output/xor.pdf
"""
import numpy as np
import matplotlib.pyplot as plt

from common import L1, OUTC, GREY, save

NAME = "xor"


def build():
    pts = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.scatter(*pts[y == 1].T, s=150, marker="^", facecolor=OUTC,
               edgecolor="white", lw=1.1, zorder=4, label="XOR $=1$")
    ax.scatter(*pts[y == 0].T, s=130, marker="o", facecolor="white",
               edgecolor=L1, lw=1.9, zorder=4, label="XOR $=0$")
    xs = np.linspace(-0.5, 1.5, 10)
    for th in (0.5, 1.5):
        ax.plot(xs, th - xs, color=GREY, lw=1.0, ls="--", alpha=.75)
    ax.set(xlim=(-0.5, 1.5), ylim=(-0.5, 1.5), xlabel=r"$x_1$",
           ylabel=r"$x_2$", xticks=[0, 1], yticks=[0, 1],
           title="no line separates the classes")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_aspect("equal", "box")
    return save(fig, NAME)


if __name__ == "__main__":
    build()

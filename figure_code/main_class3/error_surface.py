"""error_surface — AND error surface over (w1,w2): 3-D, from above, and the PLA path

Used in: Class 3 (B), slide(s): "The error surface"
Needs:   numpy, matplotlib  (optional: sci_style)
Run:     python error_surface.py            -> output/error_surface.pdf
"""
import numpy as np
import matplotlib.pyplot as plt

from common import L1, L2, OUTC, AND_X, AND_Y, and_error, pla_fixed_threshold, save

NAME = "error_surface"


def build():
    g = np.linspace(-0.5, 1.5, 220)
    W1, W2 = np.meshgrid(g, g)
    E = and_error(W1, W2)

    fig = plt.figure(figsize=(11.6, 3.3))

    # (a) the surface itself, as a landscape
    ax0 = fig.add_subplot(1, 3, 1, projection="3d")
    surf = ax0.plot_surface(W1, W2, E, cmap="plasma", linewidth=0,
                            antialiased=True, alpha=.95)
    surf.set_rasterized(True)
    ax0.set(xlabel=r"$w_1$", ylabel=r"$w_2$")
    ax0.set_zlabel("errors", fontsize=9.5)
    ax0.tick_params(labelsize=8)
    for a_ in (ax0.xaxis, ax0.yaxis, ax0.zaxis):
        a_.label.set_size(9.5)
    ax0.set_title("(a)  the error surface", fontsize=11, pad=-2)
    ax0.view_init(elev=32, azim=-128)
    ax0.set_box_aspect((1, 1, 0.62), zoom=1.02)

    # (b) the same surface from above, solution region outlined
    ax1 = fig.add_subplot(1, 3, 2)
    im = ax1.pcolormesh(W1, W2, E, cmap="plasma", shading="auto")
    im.set_rasterized(True)
    ax1.contour(W1, W2, E, levels=[0.5], colors="white", linewidths=1.8)
    cb = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.03)
    cb.set_label("misclassified patterns", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    ax1.text(1.15, 1.15, "solution\nregion", color="white", fontsize=9,
             ha="center", va="center")
    ax1.set(xlabel=r"$w_1$", ylabel=r"$w_2$",
            title="(b)  seen from above")
    ax1.set_aspect("equal", "box")

    # (c) the path the algorithm takes across that same surface
    P = AND_X[AND_Y == 1]; N = AND_X[AND_Y == 0]
    hist = pla_fixed_threshold(P, N, np.array([-0.3, 1.35]))
    H = np.array(hist)
    im2 = ax1_2 = fig.add_subplot(1, 3, 3)
    m = ax1_2.pcolormesh(W1, W2, E, cmap="plasma", shading="auto", alpha=.55)
    m.set_rasterized(True)
    ax1_2.contour(W1, W2, E, levels=[0.5], colors="white", linewidths=1.8)
    ax1_2.plot(H[:, 0], H[:, 1], color="white", lw=2.6, zorder=3)
    ax1_2.plot(H[:, 0], H[:, 1], color=L2, lw=1.5, marker="o", ms=4.5,
               mec="white", mew=0.7, zorder=4, label="PLA path")
    ax1_2.scatter([H[0, 0]], [H[0, 1]], s=75, marker="s", color=L1,
                  edgecolor="white", zorder=5, label="start")
    ax1_2.scatter([H[-1, 0]], [H[-1, 1]], s=110, marker="*", color=OUTC,
                  edgecolor="white", zorder=5, label="solution")
    ax1_2.legend(loc="lower left", fontsize=8, framealpha=.9,
                 facecolor="white", edgecolor="none")
    ax1_2.set(xlabel=r"$w_1$", ylabel=r"$w_2$",
              title="(c)  descending it")
    ax1_2.set_aspect("equal", "box")
    fig.tight_layout(w_pad=1.4)
    return save(fig, NAME)


if __name__ == "__main__":
    build()

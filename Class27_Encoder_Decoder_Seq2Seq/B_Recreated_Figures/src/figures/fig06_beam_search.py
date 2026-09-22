"""Beam search against greedy decoding, drawn twice.

`beam_toy`     the search tree of Jurafsky & Martin (2026), Fig. 13.7,
               in our style: a toy vocabulary {ok, yes, EOS} with the
               book's illustrative probabilities; greedy follows the
               best single step and ends at a worse sequence than a
               beam of width 2 finds.
`beam_search`  the same tree on the model trained here, for the source
               3 5 6 2 6 7 1 6 4 (to be reversed): from the shared
               prefix, every branch carries the model's probability and
               every leaf the running log-probability.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
from common import style as S
from common.experiment import run
from common import models as M

NAME = "beam_toy"
NAMES = ("beam_toy", "beam_search")
SRC = np.array([3, 5, 6, 2, 6, 7, 1, 6, 4])
K = 2
FROM = 5
DEPTH = 3

# Jurafsky Fig. 13.7's tree: (parent path) -> [(token, p)]
TOY = {(): [("ok", 0.4), ("yes", 0.5), ("EOS", 0.1)],
       ("ok",): [("ok", 0.7), ("yes", 0.2), ("EOS", 0.1)],
       ("yes",): [("ok", 0.3), ("yes", 0.4), ("EOS", 0.3)],
       ("ok", "ok"): [("EOS", 1.0)], ("ok", "yes"): [("EOS", 1.0)],
       ("yes", "ok"): [("EOS", 1.0)], ("yes", "yes"): [("EOS", 1.0)]}


def node(ax, x, y, label, color, r=0.42, fs=11, lw=1.6, fill="white"):
    ax.add_patch(Circle((x, y), r, fc=fill, ec=color, lw=lw, zorder=3))
    ax.text(x, y, label, ha="center", va="center", fontsize=fs, zorder=4)


def draw_tree(ax, children, root_label, greedy_path, beam_paths, dx=3.4, fmt="%.1f", root_note="", leaf_total=None, tok=lambda t: t):
    """Generic tree layout: children[path] -> [(token, p)]; paths are tuples."""
    # leaves count for vertical spacing
    def count(path):
        kids = children.get(path, [])
        return 1 if not kids else sum(count(path + (t,)) for t, _ in kids)
    ypos = {}
    def place(path, lo, hi):
        ypos[path] = (lo + hi) / 2
        kids = children.get(path, [])
        if kids:
            total = count(path); acc = lo
            for t, _ in kids:
                c = count(path + (t,)); place(path + (t,), acc, acc + (hi - lo) * c / total); acc += (hi - lo) * c / total
    n = count(()); place((), 0, 1.2 * n)
    node(ax, 0, ypos[()], root_label, "black", fs=10)
    if root_note:
        ax.text(0, ypos[()] + 0.75, root_note, ha="center", fontsize=10, color=S.GREY)
    for path, kids in children.items():
        d = len(path); x0, y0 = dx * d, ypos[path]
        for t, p in kids:
            c = path + (t,); x1, y1 = dx * (d + 1), ypos[c]
            on_g = c == tuple(greedy_path[:len(c)]); on_b = c in beam_paths
            col = S.OUTC if on_g else (S.L2 if on_b else S.GREY); lw = 2.4 if (on_g or on_b) else 1.0
            ax.plot([x0 + 0.42, x1 - 0.42], [y0, y1], color=col, lw=lw, zorder=2)
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.28, fmt % p, ha="center", fontsize=10, color=col, bbox=dict(fc="white", ec="none", pad=0.5), zorder=5)
            node(ax, x1, y1, tok(t), col, lw=lw)
            if leaf_total is not None and c not in children:
                ax.text(x1 + 0.6, y1, leaf_total(c), ha="left", va="center", fontsize=10, color=col)
    return ypos, n


def build_toy():
    S.use()
    fig, ax = plt.subplots(figsize=(12.5, 6.6))
    def prob(path):
        p = 1.0
        for k in range(len(path)):
            p *= dict(TOY[path[:k]])[path[k]]
        return p
    greedy = ("yes", "yes", "EOS"); beam_best = ("ok", "ok", "EOS")
    beam_paths = {("ok",), ("yes",), ("ok", "ok"), ("yes", "yes"), ("ok", "ok", "EOS"), ("yes", "yes", "EOS")}
    ypos, n = draw_tree(ax, TOY, "start", greedy, beam_paths, dx=3.4,
                        leaf_total=lambda c: "$p = %.2f$" % prob(c), tok=lambda t: "$\\langle/s\\rangle$" if t == "EOS" else t)
    for d, lab in ((1, "$t_1$"), (2, "$t_2$"), (3, "$t_3$")):
        ax.text(3.4 * d, 1.2 * n + 0.5, lab, ha="center", fontsize=12, color=S.GREY)
    ax.text(1.7, 1.2 * n + 0.5, "$p(t_1 \\mid \\mathrm{start})$", ha="center", fontsize=10, color=S.GREY)
    ax.text(5.1, 1.2 * n + 0.5, "$p(t_2 \\mid t_1)$", ha="center", fontsize=10, color=S.GREY)
    ax.text(8.5, 1.2 * n + 0.5, "$p(t_3 \\mid t_1, t_2)$", ha="center", fontsize=10, color=S.GREY)
    ax.plot([], [], color=S.OUTC, lw=2.4, label="greedy: yes yes $\\langle/s\\rangle$, $p = 0.20$ --- the best first step, a worse sentence")
    ax.plot([], [], color=S.L2, lw=2.4, label="beam, $k = 2$: keeps ok and yes, then ok ok and yes yes; returns ok ok $\\langle/s\\rangle$, $p = 0.28$")
    ax.legend(loc="lower center", fontsize=10.5, framealpha=0.95, bbox_to_anchor=(0.5, -0.02))
    ax.set_xlim(-1.0, 13.6); ax.set_ylim(-1.9, 1.2 * n + 1.1); ax.set_aspect("equal"); ax.axis("off")
    S.save(fig, "beam_toy")


def build_model():
    S.use()
    m = run()["fixed"]["model"]
    greedy, _ = m.greedy_decode(SRC[None, :]); greedy = [int(y) for y in greedy[0]]
    beam, lp_beam, _ = m.beam_decode(SRC, k=K)
    lp_greedy = m.sequence_logprob(SRC, greedy)
    tgt = list(SRC[::-1]) + [M.EOS]
    print("greedy", greedy, "log p = %.2f, correct: %s" % (lp_greedy, greedy == tgt))
    print("beam  ", beam, "log p = %.2f, correct: %s" % (lp_beam, beam == tgt))
    # replay the shared prefix, then expand the tree by the model's top-K at each node
    H = m.encode(SRC[None, :]); h = H[-1]; y_prev = M.BOS; base = 0.0
    def logp_row(hh, yp):
        h2, logits, _ = m.step(H, hh, np.array([yp])); z = logits.v[0]; return h2, z - z.max() - np.log(np.exp(z - z.max()).sum())
    for t in range(FROM):
        h, lp = logp_row(h, y_prev); y_prev = greedy[t]; base += lp[y_prev]
    children = {}; states = {(): (h, y_prev, base)}
    frontier = [()]
    for d in range(DEPTH):
        nxt = []
        for path in frontier:
            hh, yp, lp0 = states[path]
            h2, lp = logp_row(hh, yp)
            top = [int(y) for y in np.argsort(-lp)[:K]]
            children[path] = [(y, float(np.exp(lp[y]))) for y in top]
            for y in top:
                states[path + (y,)] = (h2, y, lp0 + float(lp[y])); nxt.append(path + (y,))
        # the beam keeps the K best of the frontier
        nxt.sort(key=lambda p: -states[p][2]); frontier = nxt[:K]
    beam_paths = set()
    for d in range(1, DEPTH + 1):
        beam_paths.add(tuple(beam[FROM:FROM + d]))
    tok = lambda y: "$\\langle/s\\rangle$" if y == M.EOS else str(y)
    fig, ax = plt.subplots(figsize=(12.5, 6.6))
    ypos, n = draw_tree(ax, children, "$\\cdots$", tuple(greedy[FROM:]), beam_paths, dx=3.4, fmt="%.2f",
                        root_note="after %s  ($\\log p = %.2f$)" % (" ".join(map(str, greedy[:FROM])), base),
                        leaf_total=lambda c: "$\\log p = %.2f$" % states[c][2], tok=tok)
    for d in range(1, DEPTH + 1):
        ax.text(3.4 * d, 1.2 * n + 0.5, "$y_{%d}$" % (FROM + d), ha="center", fontsize=12, color=S.GREY)
    ax.text(-1.2, -0.75, "the model's $p(y_t \\mid y_{<t}, \\mathbf{x})$ on each branch; the running $\\log p$ at each leaf", ha="left", fontsize=10, color=S.GREY)
    ax.plot([], [], color=S.OUTC, lw=2.4, label="greedy: %s, $\\log p = %.2f$ (wrong)" % (" ".join(tok(y) for y in greedy), lp_greedy))
    ax.plot([], [], color=S.L2, lw=2.4, label="beam, $k = %d$: %s, $\\log p = %.2f$ (the reversal)" % (K, " ".join(tok(y) for y in beam), lp_beam))
    ax.legend(loc="lower center", fontsize=10.5, framealpha=0.95, bbox_to_anchor=(0.5, -0.02))
    ax.set_xlim(-1.4, 14.2); ax.set_ylim(-1.9, 1.2 * n + 1.1); ax.set_aspect("equal"); ax.axis("off")
    S.save(fig, "beam_search")


def build():
    build_toy(); build_model()


if __name__ == "__main__":
    build()

"""cube_majority — the 3-input cube: parallel planes for OR and majority

Used in: Class 2 (B), slide(s): "Parallel planes, different thresholds"
Needs:   numpy, matplotlib  (optional: sci_style)
Run:     python cube_majority.py            -> output/cube_majority.pdf
"""
from common import L2, ACC, fig_cube

NAME = "cube_majority"


def build():
    return fig_cube(
        NAME,
        [(0.5, L2, r"$x_1{+}x_2{+}x_3=\frac{1}{2}$  (OR)"),
         (1.5, ACC, r"$x_1{+}x_2{+}x_3=\frac{3}{2}$  (majority)")],
        lambda p: int(p.sum() >= 2),
        "the two planes are parallel: only the offset changes")


if __name__ == "__main__":
    build()

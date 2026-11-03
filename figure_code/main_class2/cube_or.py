"""cube_or — the 3-input cube: one threshold plane realises OR

Used in: Class 2 (B), slide(s): "A unit splits input space into half-spaces"
Needs:   numpy, matplotlib  (optional: sci_style)
Run:     python cube_or.py            -> output/cube_or.pdf
"""
from common import L2, fig_cube

NAME = "cube_or"


def build():
    return fig_cube(
        NAME,
        [(0.5, L2, r"$x_1{+}x_2{+}x_3 = \frac{1}{2}$  (OR, $\theta{=}1$)")],
        lambda p: int(p.sum() >= 1),
        "one plane separates the 7 firing corners from the 1 silent corner")


if __name__ == "__main__":
    build()

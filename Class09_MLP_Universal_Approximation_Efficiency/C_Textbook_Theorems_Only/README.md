# Class 9 — variant C: textbook results only

Same lecture as `B_Recreated_Figures`, restricted to results that are
**stated or cited in Bishop & Bishop (2024) or Prince (2023)**. It uses
the same recreated figures as variant B.

## Kept

| result | where the books have it |
|---|---|
| the grid blows up (`S^D` cells) | Bishop §6.1.1, the curse of dimensionality |
| universal approximation (Cybenko, Hornik, Leshno) | both books |
| width-bounded universality (Lu et al. 2017) | Prince, Ch. 4 notes |
| depth separation (Telgarsky 2016) | Prince, Ch. 4 notes |
| radial separation (Eldan & Shamir 2016) | Prince, Ch. 4 notes; cited in Bishop |
| linear regions, lower bound (Montúfar et al. 2014) | Prince, Ch. 4 notes |
| linear regions, upper bound (Serra et al. 2018) | Prince, Ch. 4 notes |
| VC dimension (Bartlett et al. 2019) | Prince, Ch. 8 notes |

## Removed — in neither textbook

| result | why it is gone |
|---|---|
| **Barron (1993)** and the **Barron norm** | Prince's only "Barron" is J. T. Barron (2019), an unrelated robust loss |
| **DeVore, Howard & Micchelli (1989)** and the **Sobolev ball** | absent from both books |
| **Yarotsky (2017)** | absent from both books |
| **width lower bound** (Hanin & Sellke 2017, Johnson 2019) | absent from both books |

Seven frames were dropped with them, including the slide that defined
the Barron norm and the smoothness ball. Use variant A or B for the
full quantitative treatment.

```bash
xelatex main.tex && xelatex main.tex     # twice
```

Figures come from `../B_Recreated_Figures/src/`; only the five this
variant still uses are copied into `figs/`.

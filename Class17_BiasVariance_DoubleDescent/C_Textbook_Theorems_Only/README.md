# Class 17 — C variant (textbook theorems only)

This is the B deck with every result that appears in neither Bishop &
Bishop (2024) nor Prince (2023) removed, for use when the course wants to
stay strictly inside the two set texts.

Removed, and why:

| Slide | Rests on |
|---|---|
| The risk, in closed form | Hastie, Montanari, Rosset & Tibshirani (2022), Thm. 1 |
| Reading the two branches | the same theorem |
| Theory against measurement | the same theorem, plus the norm-blow-up mechanism |
| More data can make a model worse | the closed form, from which the curve is drawn |

Kept, because both books state them: the bias–variance–noise decomposition
(Bishop Eqs. 4.45–4.49, Prince Eq. 8.7); the interpolation threshold and
effective model complexity, both of which Bishop §9.3.2 defines and
attributes to Nakkiran et al. (2019); the smoothness argument for the
over-parameterised regime (Prince §8.4.1); and the observation that
double descent also appears epoch-wise, sample-wise and against the
inverse penalty (Bishop §9.3.2).

38 pages against the B variant's 46. The epoch-wise slide now uses Bishop's Fig. 9.10 in every variant, so it is kept here. Compile with `xelatex main.tex`,
run twice.

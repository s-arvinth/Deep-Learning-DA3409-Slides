# Class 23 — C variant (textbook theorems only)

The B deck with the two results that appear in neither set text removed.

| Result | Rests on | What C does instead |
|---|---|---|
| The **converse** of the equivariance theorem — every translation-equivariant linear map is a convolution | Cohen & Welling (2016) | states the forward direction only, as a Proposition, and proves it in five steps |
| The closed-form receptive-field recursion $r_l = r_{l-1} + (k_l-1)\prod_{j<l}s_j$ | standard, but Prince builds it figure by figure and never writes it down | gives a Definition plus the worked table $3,5,7,9$ against $3,5,9,17$, and keeps the measured figure |

Kept, because both books state them: the parameter argument (Bishop §10.2);
invariance and equivariance (Prince Eqs. 10.1–10.2); the output-length
formula, which is exactly what padding, stride and dilation do in
Prince §10.2.2–10.2.3; the matched-filter result (Bishop Exercise 10.1);
channels (Prince §10.2.5); the 2-D operation (Prince §10.3); and pooling
and upsampling (Prince §10.4, Bishop §10.2.6).

43 pages, the same as B: the two removed slides are replaced rather than
deleted, since both results have textbook statements that survive.
Compile with `xelatex main.tex`, twice.

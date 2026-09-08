# Class 18 — C variant (textbook theorems only)

The B deck with the one result that appears in neither set text removed.

Removed:

| Slide | Rests on |
|---|---|
| What weight decay does to each direction | the shrinkage factor $\lambda_i/(\lambda_i+\alpha)$ and the effective-parameter count, which Bishop (2024) §9.2 describes only qualitatively and attributes to Bishop (2006) and Hastie et al. (2009) |
| Proof, and what it means | the same theorem |
| The flat directions go first | the shrinkage-factor curves $\lambda_i/(\lambda_i+\alpha)$ of the same theorem |

The contour figure in Bishop's Fig. 9.3 layout survives, with a caption
confined to the geometry that figure actually shows.

Kept, because both books state them: the MAP reading of a penalty
(Prince §9.1.1); consistent regularizers (Bishop Eqs. 9.6–9.15); the lasso
and its sparsity (Bishop §9.2.2, Fig. 9.6); the implicit regularization of
gradient descent and of SGD (Prince Eqs. 9.8–9.9, which Prince derives in
his end-of-chapter notes); invariance and equivariance (Bishop §9.1.3–9.1.4);
parameter sharing (Bishop §9.4); and the expanded form of a residual network
(Bishop Eq. 9.40).

45 pages against the B variant's 52. Compile with `xelatex main.tex`, twice.

# Class 19 — C variant (textbook theorems only)

The B deck with the explicit *filter* form of the early-stopping /
weight-decay correspondence removed.

Removed:

| Slide | Rests on |
|---|---|
| Where that comes from | the derivation Bishop (2024) leaves to Exercise 9.6 and Bishop (1995a) |
| The two filters, measured | the same algebra, plus the ridge filter $\lambda_i/(\lambda_i+\alpha)$ already removed from the Class 18 C variant; its first panel, the descent path in Bishop Fig. 9.8's layout, survives on its own slide *Descent on a quadratic, drawn* |

Rewritten: *Stopping early is weight decay in disguise* now states the
correspondence in the form Bishop's §9.3.1 actually gives it — descent
reaches stiff directions first and flat ones last, which is the same
selection a penalty makes, and $\eta\tau$ acts as the reciprocal of the
regularization coefficient — as a Proposition rather than a Theorem.

Kept, because both books state them: the committee-error results
(Bishop Eqs. 9.44–9.50 and Exercise 9.15); the weight-scaling inference
rule (Bishop Eq. 9.51, Prince §9.3.3); training with input noise as a
penalty on $\partial\hat y/\partial\mathbf{x}$ (Prince §9.3.4); and the
Bayesian prediction integral (Prince Eqs. 9.11–9.12).

40 pages against the B variant's 46. Compile with `xelatex main.tex`, twice.

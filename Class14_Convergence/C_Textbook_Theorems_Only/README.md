# Class 14 — textbook-theorems-only variant

This is the B deck with every result that appears in **neither**
Bishop & Bishop (2024) **nor** Prince (2023) removed. It uses the same
purpose-built figures as the B variant.

Five results are dropped, together with the slides that state them:

| Removed | Where it comes from |
|---|---|
| the optimal fixed learning rate $\eta^\star = 2/(\lambda_{\min}+\lambda_{\max})$ and the factor $(\kappa-1)/(\kappa+1)$ | standard convex-optimization result |
| the step count $\tfrac{\kappa+1}{2}\ln(1/\epsilon)$ | follows from the above |
| the heavy-ball rate $(\sqrt{\kappa}-1)/(\sqrt{\kappa}+1)$ | Polyak (1964) |
| Nesterov's $O(1/T^2)$, and $1 - 1/\sqrt{\kappa}$ | Nesterov (1983) |
| the Robbins–Monro conditions $\sum\eta = \infty$, $\sum\eta^2 < \infty$ | Robbins & Monro (1951) |

Everything that remains is stated in one of the two books:

* the decoupled contraction $\alpha_i^{(T)} = (1-\eta\lambda_i)^T
  \alpha_i^{(0)}$ and the condition $\eta < 2/\lambda_{\max}$
  — Bishop Eqs. 7.24–7.29
* the rate at the stability limit, $1 - 2\lambda_{\min}/\lambda_{\max}$
  — Bishop Eq. 7.30
* the effective learning rate $\eta/(1-\mu)$ on a consistent slope
  — Bishop Eqs. 7.32–7.33
* the bias correction $1/(1-\beta^{\tau})$ — Bishop Eqs. 7.45–7.46,
  Prince Eq. 6.16
* Algorithms 7.3 and 7.4 — Bishop

The consequence is that the ill-conditioning and momentum sections
become qualitative where the B variant is quantitative: they say that
momentum helps and why, without stating by how much.

# Class 15 — textbook-theorems-only variant

This is the B deck with every result that appears in **neither**
Bishop & Bishop (2024) **nor** Prince (2023) removed. It uses the same
purpose-built figures as the B variant.

Four results are dropped, together with the slides that state them:

| Removed | Where it comes from |
|---|---|
| the feature-scale law, `kappa = (sigma_2/sigma_1)^2` | standard, stated in neither book |
| batch normalization's scale invariance, and the `1/a` gradient law | standard, stated in neither book |
| the smoothness explanation as a formal statement | Santurkar et al. (2018) |
| the pre-norm / post-norm placement | Xiong et al. (2020) |

Bishop does mention Santurkar et al.'s finding in prose, so the
qualitative statement survives in this variant; what is removed is the
formal proposition and the measurement that goes with it.

Everything that remains is stated in one of the two books:

* data normalization — Bishop Eqs. 7.48–7.50
* the variance across one ReLU layer, and `sigma_w^2 = 2/D` — Prince
  Eqs. 7.28–7.31, Bishop Eqs. 7.21–7.23
* the forward/backward compromise `4/(D + D')` — Prince Eq. 7.33
* the Jacobian product behind vanishing and exploding gradients —
  Bishop Eq. 7.51
* symmetry breaking — Bishop §7.2.5
* batch normalization and its learnable scale and shift — Bishop
  Eqs. 7.52–7.55
* the running averages used at inference — Bishop Eqs. 7.56–7.57
* layer normalization — Bishop Eqs. 7.58–7.60

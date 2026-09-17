# Class 25 — C variant (textbook theorems only)

The B deck with the three results that appear in neither set text removed.

Removed:

| Slide | Rests on |
|---|---|
| Proposition *Composition of kernels* and its proof | the exact statement that two linear kernels equal one of size k₁+k₂−1, which Bishop §10.2.8 gives only in words |
| Proposition *Why a tiny ε is enough* | the ε‖w‖₁ arithmetic of Goodfellow et al. (2014), which Bishop reports as a finding rather than a result |
| Proposition *The style matrix is translation-invariant* and its proof | Bishop §10.6 says the style error averages over locations; the invariance is not stated |

Rewritten: *Small filters, composed* keeps Bishop's own remarks (reach from
depth, fewer parameters, the inductive bias of factored filters); the
fast-gradient-sign slide keeps Bishop's remarks on visibility, confidence
and linear models; *Why the style matrix forgets position* keeps Bishop's
account of content against style.

Kept, because the books state them: the layer parameter count (Bishop
§10.2.7); the 1×1 convolution (Prince Fig. 10.14); the Gabor filter
(Bishop Eqs. 10.6–10.8); maximising the pre-activation rather than the
softmax (Bishop §10.3.2); Grad-CAM (Bishop Eqs. 10.9–10.10); the fast
gradient sign method (Bishop Eq. 10.11); DeepDream (Bishop Eq. 10.12); the
content and style errors (Bishop Eqs. 10.13–10.17).

43 pages against the B variant's 47. Compile with `xelatex main.tex`, twice.

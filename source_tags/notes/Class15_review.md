# Class 15 — Normalization: independent review of source tags

Tags checked: A 62 (52 \src + 10 \nosrc), B 66 (56 + 10), plus 7 \srcdef per deck (keys consistent with keys.md).
Checks: check_sources 0 errors / 0 warnings; verify_book_numbers 0 not found; build + compare_pdfs IDENTICAL
(A 45 pp, B 45 pp). The 5 commented-out B frames carry no tags (correct).

## Verified against the sources
- Bishop 2024 §7.2.5 (symmetry breaking, Eqs 7.19-7.23, He init), §7.4.1 (height/platelet example,
  Eqs 7.48-7.50, Fig 7.7), §7.4.2 (Eq 7.51 chain rule, 7.52-7.55, Fig 7.8, 7.56-7.57, Santurkar remark),
  §7.4.3 (small batches, GPUs, RNNs; Eqs 7.58-7.60) — booktext.
- Prince 2023 §7.5 (vanishing/exploding), §7.5.1 Eqs 7.27-7.31 (7.31 He), §7.5.2 Eq 7.32, §7.5.3 Eq 7.33,
  Fig 7.7 (50 layers, sigma^2 = 2/Dh = 0.02); Notes p.113 (Glorot = He without the factor of two).

## Changes
None. By kind: wrong place 0, missing 0, less specific 0, over-tagged 0, srcdef 0.

## Still unverified
- Tagger listed none. Paper tags not re-opened by the reviewer (tagger's verification accepted): he2015 Eqs 9, 10
  / §2.2, glorot2010 Eqs 10-12 / §4.2.1, ioffe2015 Algorithm 1 / §3 / §3.3, ba2016 Eq 3 / §3, wu2018 Fig 2 / §3.1,
  santurkar2018 §2.1 / §3.1, xiong2020 Fig 1 / §3.2 / §3.3.
- Open point carried over: the original source of the kappa = (sigma2/sigma1)^2 feature-scale law is not
  named in the deck; tagged Bishop §7.4.1 only.

# Class 17 — Bias, Variance and Double Descent: independent review of source tags

Tags checked: A 51 (44 \src + 7 \nosrc after review; 50 before), B 57 (50 + 7; 56 before), plus 4 \srcdef per deck.
Checks after edits: check_sources 0 errors / 0 warnings; verify_book_numbers 0 not found; build + compare_pdfs
IDENTICAL (A 44 pp, B 42 pp).

## Verified against the sources
- Bishop 2024 §4.2 Eq 4.39; §4.3 Eqs 4.41-4.49, Figs 4.7, 4.8; §9.3.2 Figs 9.9-9.11 (classical belief,
  Belkin threshold, Nakkiran EMC, epochs, 1/lambda with Yilmaz & Heckel, more data hurts) — booktext.
- Prince 2023 Fig 8.5; §8.2.2 Eqs 8.1-8.7; §8.3.1-8.3.3; Fig 8.9; §8.4, Fig 8.10; §8.4.1 (10^40 bins,
  inductive bias); Fig 8.11; §8.5 (validation set; discrete and conditional hyperparameters).
- Nakkiran et al. (arXiv v1): §2 "Our results" with Definition 1 (EMC), §5 model-wise, §6 epoch-wise,
  §7 sample-wise non-monotonicity.
- Hastie et al. (arXiv v5): §3.2 "Limiting risk", Theorem 1, Eqs (6)-(8).

## Changes (frame: old -> new, reason)
1. A and B "What reduces each term": added \src{bishop2024}{section}{4.3} after the three Prince tags
   (missing: the table row "variance — a penalty on w — more bias" is Bishop §4.3's regularization-parameter
   trade-off (Figs 4.7-4.8); Prince §8.3.1-8.3.3 covers data and capacity only).

By kind: wrong place 0, missing 2 (one per deck), less specific 0, over-tagged 0, srcdef 0.

## Still unverified
- Tagger listed none. Not re-opened by the reviewer (tagger's verification accepted): belkin2019 Fig 1 / §1 / §2,
  yilmaz2022 §5.

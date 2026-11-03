# Class 18 review (independent check of the source tags)

## Checked
- A: 35 frames, 66 \src + 9 \nosrc (before review), 8 \srcdef. B: 37 frames, 71 \src + 9 \nosrc, 9 \srcdef.
- Every Bishop 2024 / Prince 2023 number read against the local book text (caption, equation, section text).
- Papers opened: barrett2021 (arXiv 2009.11162: Thm 3.1 in Sec. 3, proof in App. A.1), smith2021 (arXiv 2101.12176:
  Eq. 2 in Sec. 1, derivation Sec. 2.2), balduzzi2017 (arXiv 1702.08591: Sec. 3.2 "Residual networks"),
  tibshirani1996 (JRSS-B copy: Sec. 2.3 "Geometry of Lasso", p. 270), simard1992 (NeurIPS PDF pp. 895-903: Sec. 2
  "Implementation", regularizer Eqs 1-2). he2016 Sec. 3.1 accepted from the tagger's check.
- \srcdef texts identical to keys.md; all 5 checks: after edits check_sources 0 errors / 0 warnings, verify_book_numbers
  0 not found, both PDFs IDENTICAL to baseline (A 50 pages, B 52 pages).

## Changes (A and B, same frames)
1. "What the step size is really minimising": \src{prince2023}{equation}{9.8} -> \src{prince2023}{equation}{9.6-9.8}
   (less specific: the theorem states the gradient flow, Eq. 9.6, and the discrete step, Eq. 9.7, as well as the
   modified loss 9.8). A and B.
2. "Where the extra term comes from": added \src{prince2023}{equation}{9.14-9.20} as the first tag (missing: the
   slide's five steps -- second-order expansion, correction term, H grad L, modified loss L + alpha/4 |grad L|^2 -- are
   the derivation printed in Prince's chapter-9 Notes, Eqs 9.14-9.20; barrett2021 A.1 kept second). A and B.
Totals: less specific 2, missing 2, wrong place 0, over-tagged 0, srcdef 0.

## Looked at and left unchanged
- "What weight decay does to each direction": Bishop 2024 Fig. 9.3/§9.2 first although the formula comes from PRML
  §3.5.3; judgement call, srcline order kept.
- "Why residual connections smooth the surface" (A): Figs 9.12/9.15 are not on the A slide, but the proposition is
  Eq. 9.40 "depicted in Fig. 9.15" and the shattered-gradient bullet is Fig. 9.12; kept, same as B.
- srcdefs of conference papers (barrett2021, smith2021, balduzzi2017, simard1992, he2016) carry venue + year but no
  arXiv id / pages; they match keys.md and are shared with other decks, so not changed here (coordinator may want to
  extend keys.md globally).

## Still unverified
1. "Learning from data alone is not possible" -- \src{wolpert1996}{page}{1341-1390}: whole-article range; no open copy
   could be fetched (timeouts), page of the theorem not found.
2. "Parameter sharing: a hard constraint" -- \src{nowlan1992}{page}{473-493}: whole-article range; author copy
   (cs.toronto.edu/~hinton/absps/sunspots.pdf) is a scanned PDF, no text; MIT Press page timed out.
3. bishop2006 §3.5.3: only the PRML table of contents was seen (title "Effective number of parameters"), not the
   section text / equation number of the shrinkage factor.

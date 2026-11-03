# Class 14 — Convergence: source tags

| variant | frames | \src | \nosrc |
|---|---|---|---|
| A_Textbook_Figures | 43 | 71 | 12 |
| B_Recreated_Figures | 50 | 88 | 14 |

Checks: build + compare_pdfs IDENTICAL for A (59 pp) and B (58 pp).
`tools/check_sources.py` reports 24 ERRORs per deck, all "malformed \src (needs three arguments)":
a false positive of the linter — its regex `\\src(?!\{)` matches the deck's existing `\srcline{...}`
macro (every frame that has a `\srcline`). With that one regex changed to `\\src(?![{a-z])`
(patched copy in /tmp, repository tool not modified) both decks give 0 errors, 0 warnings.
The linter needs that fix before Classes 14/15/17 (and any deck using `\srcline`) can pass.

All tags are the first lines of the frame body (no frame body starts with `{`).

## \srcdef keys
polyak1964, nesterov2004, robbins1951, goyal2018 (new, appended to keys.md), loshchilov2017,
duchi2011, hinton2012 (new, appended to keys.md: Hinton/Srivastava/Swersky Coursera Lecture 6 slides),
kingma2015, reddi2018, loshchilov2019. Fixed keys used: bishop2024, prince2023.

## Places checked
- Bishop 2024: §7.1.1 Eqs 7.3-7.14, Fig 7.2; §7.3 Eqs 7.24-7.30 (condition number named on p.220), Figs 7.3-7.6;
  §7.3.1 Eqs 7.31-7.34, Algorithm 7.3; §7.3.2 Eqs 7.35-7.38; §7.3.3 Eqs 7.39-7.47, Algorithm 7.4.
- Prince 2023: §6.2.2, §6.3 Eq 6.11, Fig 6.7; §6.3.1 Eq 6.12, Fig 6.8; §6.4 Eqs 6.13-6.18, Fig 6.9; §6.5;
  chapter-6 Notes printed pp. 93 (warm-up, AdaGrad/RMSProp/Adam/AMSGrad) and 94 (SGD vs Adam, AdamW).
- Robbins & Monro 1951 (JSTOR scan via columbia.edu): §3 "Convergence theorems", conditions (6) and (26).
- Duchi et al. 2011 (JMLR PDF): §1.1, diagonal update Eq (1), p. 2123.
- Kingma & Ba 2015 (arXiv v9): Algorithm 1 (p.2), §2, §3 Eqs (1)-(4).
- Loshchilov & Hutter 2017 (arXiv): §3, cosine annealing Eq (5).
- Goyal et al. 2018 (arXiv v2): §2.2 "Warmup".
- Reddi et al. 2018 (arXiv): §3 non-convergence (Theorem 1), §4 AMSGrad (Algorithm 2).
- Loshchilov & Hutter 2019 (arXiv): §2, Algorithm 2 (AdamW).
- Hinton et al. 2012 lecture 6 slides: "rmsprop: A mini-batch version of rprop" (PDF page 29, counted from the fetched text).
- Nesterov 2004: chapter 2 "Smooth Convex Optimization", pp. 51-110 (Springer table of contents).
- Polyak 1964: pages 1-17 (Crossref metadata only).

## Judgement calls
- "What it means for an iteration to converge", "An iteration to converge", "Two rates, drawn": standard definition of linear/sublinear rate, no source cited on the slide -> \nosrc.
- "The best case, stated" (A), "When every direction curves alike" (B): instructor's consequence of Eq 7.27 with equal eigenvalues; Bishop does not state it -> \nosrc.
- "The best a fixed step size can do" (A) / "Optimal learning rate for a quadratic" (B): the eta* = 2/(lmin+lmax) result is not in Bishop or Prince and the slide cites nothing -> \nosrc.
- "Where the condition number comes from" / "Possible Origin of Condition Number": the course's own list -> \nosrc.
- "A ball with inertia": own TikZ drawing of the mechanism Bishop states in §7.3.1 (accumulate on low curvature, cancel when oscillating) -> section 7.3.1. "The analogy, term by term": analogy table -> \nosrc.
- "The two regimes" (A crop, B redraw "following the geometry of Figs 7.4 and 7.5") -> figures 7.4, 7.5.
- "The same two regimes, measured" (B): measured experiment of Eqs 7.32-7.33 -> those equations.
- "Nesterov on the same valley" (B): course simulation comparing (7.31) with Nesterov (7.34) -> Eq 7.34, §7.3.1.
- "What that costs in iterations" (B): plots the heavy-ball factor of Polyak (1964) -> polyak1964.
- "Nesterov momentum, one step drawn" (B): TikZ redraw of Prince Fig 6.8 (same points 1-5) -> figure 6.8.
- "The guarantee behind the name": slide cites Nesterov (1983, 2004). Only Nesterov 2004 chapter 2 could be verified (TOC); Nesterov 1983 not tagged (no verifiable page). Bishop §7.3.1 added for the batch-vs-stochastic remark.
- "The noise floor" (B): measured experiment of the claim of "Why a fixed step size cannot finish the job" -> same tags as that frame.
- "The schedules in use" (B): plots Eqs 7.36-7.38 plus cosine annealing (Loshchilov 2017 Eq 5) and warm-up (Goyal 2018 §2.2).
- "Warm-up, and why the start is different": main source is Prince's Notes p.93 (the adaptive-statistics argument and "first few thousand iterations"), then Goyal §2.2, then SGDR Eq 5 for cosine annealing.
- "Adam" / "AdaGrad" / "RMSProp": equations on the slides are in Bishop's notation and typical values (beta2 = 0.99) are Bishop's -> Bishop first, original paper second.
- "Why the bias correction is there/required?": statement in Bishop's notation (Eqs 7.45-7.46); the proof steps follow Kingma & Ba §3 Eqs (1)-(4); Prince Eq 6.16.
- "The correction, drawn" (B): plot of the bias correction -> Bishop 7.45-7.46, Kingma §3.
- "Five update rules, by family" (B): course's own comparison experiment, no single result tested -> \nosrc.
- "Four honest caveats": Prince Notes pp. 93-94 main; first bullet (axis-aligned curvature) is Bishop §7.3.3; Reddi §3/§4 and Loshchilov 2019 Algorithm 2/§2 for the stated results.
- B deck's last frame "The problem, and three responses" sits after the Questions frame (still before \end{document}); recreation of Prince Fig 6.9 -> figure 6.9.

## Unverified
1. "What momentum buys, precisely" (A), "Momentum: " (B), "What that costs in iterations" (B): \src{polyak1964}{page}{1-17} — the paper is paywalled; only the article's page range (Crossref) was checked, not the page of the heavy-ball theorem.
2. "The guarantee behind the name" (A, B): \src{nesterov2004}{chapter}{2} — chapter title and pages seen on Springer's TOC; the theorem itself and its section inside chapter 2 not seen. Nesterov 1983 left untagged.
3. "RMSProp" (A, B): \src{hinton2012}{page}{29} — PDF page counted from the text extraction of the slide file (could not download the PDF to run pdfinfo).

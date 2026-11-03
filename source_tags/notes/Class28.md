# Class 28 — LSTM and Gated RNNs: source tags

| variant | frames | \src | \nosrc | check_sources | PDF vs baseline |
|---|---|---|---|---|---|
| A | 30 | 65 | 6 | 0 errors with the \srcline fix; official linter: 18 errors, all "malformed \src" false positives from `\srcline` | IDENTICAL (46 pages) |
| B | 33 | 71 | 6 | same: 0 with fix; 18 false positives with the official linter | IDENTICAL (49 pages) |

Linter false positive and bare `\newcommand` lines: as in Class27.md (the tool was not edited; checked with
`../slide-sources-work/tools_c27_29/check_sources_srcline.py`, which differs only in also stripping `\srcline`).

\srcdef keys (both variants): cho2014, chung2014, graves2014, greff2017, hochreiter1997, jozefowicz2015, jurafsky_slp3
(draft of August 19, 2026), mikolov2012, mozer1992, pascanu2013. All existing keys; no new key.

## Verified places (papers)
- hochreiter1997 §3.2.2 The Constant Error Carousel, §4.1 Memory Cells and Gate Units (published Neural Computation 9, 1735-1780 layout, copy at gwern.net/doc/ai/nn/rnn/1997-hochreiter.pdf)
- pascanu2013 §2.1 The mechanics (sufficient condition λ1 < 1/γ), §2.3 The geometrical interpretation (cliffs), §3.2 Scaling down the gradients (Algorithm 1), §3.3 Vanishing gradient regularization (arXiv 1211.5063)
- mozer1992 §1 Building a reduced description (units with time constants, Eqs. 1-2) (NeurIPS proceedings PDF)
- mikolov2012 §3.3.2 Practical Advices for the Training (gradient truncation to [-15, 15]) (thesis PDF from fit.vutbr.cz)
- cho2014 §2.3 and Eqs. (5)-(8) (GRU) (arXiv 1406.1078v3)
- chung2014 §3.2 Gated Recurrent Unit (arXiv 1412.3555v1)
- greff2017 §V.A Comparison of the Variants (removing the forget gate hurts on all datasets) (arXiv 1503.04069; number written "V.A" as printed: section V, subsection A)
- jozefowicz2015 §2.2 Forget Gates Bias, §4 Results (forget gate most important; bias 1) (PMLR v37 PDF)
- graves2014 §3.1 Reading, §3.2 Writing (arXiv 1410.5401)

## Judgement calls
- "Where we are coming from": recap of Classes 26-27 plus the statement of today's problem; tagged J §14.5 and G §10.7 (the problem as both books state it) rather than \nosrc.
- "The problem": the running sentence is J's numbered example (14.19) in §14.5; (14.19) is an example number, not an equation, so the tag is J section 14.5, not equation 14.19.
- "Why the simple cell is not enough": J §14.5 added (its "two tasks simultaneously" paragraph is the slide's last bullet) although the \srcline names only G and Bengio et al.
- "The three paths, drawn" (course figure): caption cites G §10.8, but the panels are skip connections (§10.9.1), leaky units (§10.9.2) and gates (§10.10); §10.8 (echo state networks) is not drawn, so it is not tagged there. It is tagged on "Three ways to a path ..." (the ESN bullet).
- "The LSTM cell, in equations": jozefowicz2015 §2.2 for the "forget bias of 1" remark.
- "Why the cell remembers": the proposition is the course's (plan: "in neither book as a statement"); tagged the CEC it formalises (Hochreiter & Schmidhuber §3.2.2) and G §10.10.1.
- "Clipping at a cliff, drawn": G Fig. 10.17 only; "after Pascanu et al." is the book's attribution, not tagged separately.
- "Explicit memory, explained": Weston et al.'s memory networks are only named; the soft read/write described is the NTM's (Graves et al. §3.1, §3.2), so only graves2014 is tagged.
- "The GRU": the slide's form (U_h(r⊙h), h = u⊙h_{t-1} + (1-u)⊙h̃) is G Eqs. 10.45-10.47 / Chung §3.2; Cho §2.3 Eqs. 5-8 is the original.
- B-only measured frames: power-method result -> G Eqs. 10.36-10.39, §10.7; gradient by lag -> G §10.7, J §14.5; three cells on one task -> J §14.5, G §10.10 (the results they test).
- "The cell, opened" (summary table), Takeaways, References, Questions: \nosrc.

## Unverified (cited on the slides' \srcline, not tagged because no copy could be opened and no section/page could be checked)
1. bengio1994 (IEEE TNN 5(2), paywalled) — "Why the simple cell is not enough", "With the nonlinearity: a bound": content tagged via G §10.7 / Pascanu §2.1 instead.
2. gers2000 (Neural Computation 12(10), paywalled; only the 1999 IDSIA report is open) — "The idea", "What the LSTM has to do", "Why the cell remembers": forget gate tagged via J §14.5 / G §10.10.
3. lin1996 (IEEE TNN 7(6), paywalled) — "Three ways", "The three paths": tagged via G §10.9.1.
4. jaeger2004 (Science 304, paywalled) — "Three ways" (echo state networks): tagged via G §10.8.

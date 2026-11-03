# Class 05 — independent review of source tags

Tags checked: A 34 frames (73 -> 72 \src, 6 \nosrc), B 42 frames (77 -> 77 \src, 11 \nosrc);
\srcdef cybenko1989, hornik1991 (A, B), leshno1993 (B) — text matches keys.md.

## How checked
- Prince (booktext/prince2023.txt): Figs 2.1-2.4, 3.1-3.13 captions; Eqs 2.4, 2.5, 3.1-3.10; sections 2.2.1-2.2.3,
  3.1, 3.1.1, 3.1.2, 3.2, 3.3.1, 3.3.2, 3.4, 3.5; Notes p. 37 (activations, Fig. 3.13) and p. 38 (Cybenko/Hornik,
  Zaslavsky bound). A crops rendered and compared: SupervisedLinear = Fig 2.1 (three lines), ShallowRegions =
  Fig 3.9, ShallowHyperplanes = Fig 3.10 — tags right, visible captions are the other edition (not changed).
- Bishop 2024 (booktext/bishop2024.txt): Eqs 4.3, 4.8-4.16, 4.37, 4.39, 6.7-6.9, 6.12, 6.17 (ReLU), 6.29;
  Figs 4.1, 4.2, 4.3, 4.5, 6.9, 6.10 (in 6.2.2), 6.12 (in 6.2.3), 6.17 (discussed in 6.5.1); sections 4.1.1-4.1.4,
  4.2, 6.2, 6.2.1-6.2.4, 6.4.1, 6.5.1.
- Cybenko 1989 (open scan with journal pagination, web.njit.edu): Theorem 2 in §3 "Application to Artificial
  Neural Networks" (p. 308); existence-only / astronomical number of terms / curse of dimensionality on p. 313,
  inside §5 "Summary". Both tags correct.

## Changes
| deck | frame | old -> new | kind | reason |
|---|---|---|---|---|
| A | Why least squares? The probabilistic view | removed `\src{bishop2024}{section}{4.1.3}` | wrong place / over-tagged | Eqs 4.8-4.11 and the statement "maximizing the likelihood ... equivalent to minimizing the sum-of-squares error (4.11)" are all in §4.1.2; §4.1.3 (solving for w_ML, Eqs 4.12ff) is not on this slide |
| B | Why least squares? The probabilistic view | removed `\src{bishop2024}{section}{4.1.3}` | wrong place / over-tagged | same as A (the proof shown is Eq. 4.10, §4.1.2) |
| B | Linear regression, two views | added `\src{prince2023}{equation}{2.4}` after the figure tag | missing / less specific | slide states the 1-D line model y = w0 + w1 x = Prince Eq. 2.4 (A variant already tags it) |

## Judgement calls accepted (not changed)
- B "It works": own fits tagged with Bishop Fig. 6.17 / §6.5.1 following the A/B rule; arguably an own experiment.
- "How many regions?" (A): figure order 3.10 before 3.9 follows slide layout; acceptable.

## Still unverified
- hornik1991 §2 (Theorem 2) and leshno1993 §4/§6: open copies could not be fetched in this review (fetch timeouts);
  relying on the tagger's and the Class07 tagger's independent reading.
- Zaslavsky 1975 (paywalled): bound tagged via Prince p. 38, as in Class05.md.

## Checks
check_sources 0 errors/0 warnings (A, B); verify_book_numbers 0 not found; build vs baseline IDENTICAL
(A 39 pages, B 54 pages).

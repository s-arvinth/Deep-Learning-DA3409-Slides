# Class 13 — Gradient Descent and Optimization

| variant | frames | \src | \nosrc | check | PDF vs baseline |
|---|---|---|---|---|---|
| A_Textbook_Figures | 45 | 65 | 11 | 17 errors (checker false positive, see below), 0 warnings | IDENTICAL (61 pages) |
| B_Recreated_Figures | 51 | 70 | 12 | 17 errors (same false positive), 0 warnings | IDENTICAL (66 pages) |

Tags are the first lines of every frame body.

## The 17 "malformed \src" errors are a checker bug, not a tagging error
This deck has a pre-existing macro `\srcline{...}` (18 uses, already in base commit 03bf6e7).
`check_sources.py` line 59 tests `re.search(r"\\src(?!\{)", body.replace(r"\srcdef", ""))`, which
matches the `\src` prefix of `\srcline`. Every frame containing `\srcline` is flagged, tagged or not.
I may not edit existing text, so this cannot be fixed in the deck. With the one-line fix
`body.replace(r"\srcdef", "").replace(r"\srcline", "")` (tested on a copy in /tmp, repo tool untouched)
both decks report **0 errors, 0 warnings**. The coordinator should patch the checker (or the regex to
`\\src(?![{a-z])`).

## \srcdef keys used (both variants)
bottou2010, nesterov2004 (both already in keys.md; text copied). Fixed keys: bishop2024, prince2023.

## How the numbers were checked
- Bishop: Fig 7.1 and Eqs 7.1-7.2 in 7.1; 7.1.1 Eqs 7.3-7.14 (7.11 = sum of lambda_i alpha_i^2, 7.12-7.14
  positive definiteness), Fig 7.2; 7.2 Eq 7.15; 7.2.1 Use of gradient information (W(W+3)/2, O(W^3) vs
  O(W^2)); 7.2.2 Eq 7.16; 7.2.3 Eqs 7.17-7.18, Algorithm 7.1, redundancy/duplication argument, escaping
  local minima; 7.2.4 Mini-batches (sigma/sqrt N, factor 100 -> 10, powers of 2, shuffling), Algorithm 7.2;
  7.1 / 6.2.4 for the M! 2^M equivalent minima; chapter 7 intro for "optimization is a means".
- Prince (Oct 23, 2023): Eqs 6.1-6.3 and line search in 6.1; Fig 6.1 in 6.1.1; 6.1.2 Gabor model example
  (convexity / chord definition), Figs 6.2, 6.3; 6.1.3 Local minima and saddle points, Fig 6.4; 6.2, Fig 6.5;
  6.2.1 Eq 6.10, Fig 6.6; 6.2.2 Properties of SGD (the six properties; "does not converge in the
  traditional sense"); chapter-6 Notes "Convexity, minima, and saddle points" on printed page 92;
  Fig 6.10 (line search, bracketing); Fig 6.11 (three 1D loss functions, problem 6.6).
- Bottou 2010 (author's PDF fetched): sec 2.2 Stochastic gradient descent, Eq (4).

## Judgement calls
- A "Convex and non-convex, side by side": the crop TrainConvexProb is Prince **Figure 6.11** (three 1D
  losses, problem 6.6; checked visually); caption only says "Prince §6.1.2". Tagged figure 6.11 + 6.1.2.
- A "Choosing the step length by search": crop TrainLineSearch is Prince **Figure 6.10** (bracketing line
  search; checked visually); caption says "§6.1". Tagged figure 6.10 + section 6.1.
- B recreated figures tagged by the figure they follow (src/README "Follows" column): linreg_descent ->
  Prince Fig 6.1; gabor_landscape -> Figs 6.4, 6.5; sgd_escape -> Figs 6.5, 6.6; quadratic_ellipses ->
  Bishop Fig 7.2; stationary_types / curvature -> Bishop Eq 7.11; batch_variance / shuffling -> 7.2.4;
  gradient_payoff -> 7.2.1; convex_vs_nonconvex -> Prince 6.1.2 (not a redraw of Fig 6.11, so no figure
  tag); steepest_direction -> Bishop Eqs 7.1 / 7.1; descent_lemma and the two step-regime figures ->
  the descent lemma (Nesterov). batch_tradeoff ("Follows: —") -> \nosrc.
- "Four regimes for the step size" (table) and the B step-regime plots: the course's reading of the
  descent lemma; tagged nesterov2004 like the lemma itself.
- "Why the step goes along -grad E": the lemma and Cauchy-Schwarz proof are the course's own, built on
  Bishop Eqs 7.1-7.2 (as its \srcline says).
- "What convexity buys": the course's list; the facts are Prince's Notes p. 92 and 6.1.2.
- "Stationary points": Bishop 7.11-7.14 / 7.1.1, plus Prince Notes p. 92 (Hessian-eigenvalue test).
- "The estimate is unbiased": proposition is the course's derivation; based on Bishop 7.2.4 and Prince
  6.2.1 ("expected gradients ... remain the same"), as the \srcline says.
- "Stochastic gradient descent": Bottou (2010) cited last (sec 2.2), after Bishop Eq 7.18 / Alg 7.1 / 7.2.3.
- \nosrc: Outline, roadmap, Notation, "The hiker in fog" (course analogy), "The bill for losing
  convexity" (summary table), "What the batch size buys" (course discussion; in B also its own plot),
  B "The batch size is a trade-off", "One training run", "What we choose", Takeaways, References,
  Questions.

## Unverified
Nesterov (2004) is a library/purchase book; no copy of the 2004 Kluwer edition could be opened.
The deck's own \srcline names "Lemma 1.2.3"; the section holding it is taken as 1.2.2 "Classes of
differentiable functions" from a table of contents of a 1998 draft (Scribd listing), not the 2004 edition.
1. A "Four regimes for the step size" — \src{nesterov2004}{section}{1.2.2}
2. A "Smoothness: one number for the worst curvature" — \src{nesterov2004}{section}{1.2.2}
3. A "How far a single step may go" — \src{nesterov2004}{section}{1.2.2}
4. B "Smoothness: one number for the worst curvature" — same tag
5. B "How far a single step may go" — same tag
6. B "The guarantee, as a function of the step" — same tag
7. B "Four regimes for the step size" — same tag
8. B "The four regimes on the contour map" — same tag
9. B "The four regimes on the surface" — same tag

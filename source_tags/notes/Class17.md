# Class 17 — Bias, Variance and Double Descent: source tags

| variant | frames | \src | \nosrc |
|---|---|---|---|
| A_Textbook_Figures | 29 | 43 | 7 |
| B_Recreated_Figures | 31 | 49 | 7 |

Checks: build + compare_pdfs IDENTICAL for A (44 pp) and B (42 pp).
`tools/check_sources.py` reports 12 ERRORs per deck, all "malformed \src (needs three arguments)" — the
linter's regex `\\src(?!\{)` matches the deck's existing `\srcline{...}` macro (false positive, see Class14.md).
With that regex changed to `\\src(?![{a-z])` in a /tmp copy both decks give 0 errors, 0 warnings.
All tags are the first lines of the frame body. (B has stray text after \end{document}; untouched.)

## \srcdef keys
belkin2019, nakkiran2020, hastie2022, yilmaz2022 (all already in keys.md). Fixed keys: bishop2024, prince2023.
zhang2017 and bartlett2020 appear only on the References slide and are not tagged anywhere.

## Places checked
- Bishop 2024: §4.2 Eq 4.39; §4.3 Eqs 4.41-4.52, Figs 4.7, 4.8; §9.3.2 Figs 9.9-9.11 (classical belief,
  Belkin threshold, Nakkiran effective model complexity, epochs, 1/lambda with Yilmaz & Heckel, more data hurts).
- Prince 2023: §8.2 Fig 8.5; §8.2.2 Eqs 8.1-8.7; §8.3.1-8.3.3, Fig 8.9; §8.4 Fig 8.10; §8.4.1 (10^40 bins,
  inductive bias), Fig 8.11; §8.5 (validation set, discrete/conditional hyperparameters).
- Belkin et al. 2019 (arXiv): §1 with Fig 1 (interpolation threshold); §2 random-features experiments.
- Nakkiran et al. 2020 (arXiv): §2 Definition 1 (EMC) and the regimes; §5 model-wise, §6 epoch-wise, §7 sample-wise.
- Hastie et al. 2022 (arXiv): §3.2 Theorem 1, Eqs (6)-(8).
- Yilmaz & Heckel 2022 (arXiv): §5 double descent in deep networks vs regularization strength.

## Judgement calls
- "Notation for today": a notation slide (guide: \nosrc), although its \srcline names Bishop 4.41 / Prince 8.1.
- "The best predictor there is": the proposition is Eq 4.39 (in §4.2, Decision theory); the slide also cites 4.42 (§4.3).
  Tagged 4.39, 4.42, §4.2.
- "Proof, in two identical steps": no citation on the slide; the order (noise first, then bias/variance) is
  Prince's derivation Eqs 8.2-8.7 in §8.2.2.
- "What reduces each term": three subsections tagged separately (8.3.1, 8.3.2, 8.3.3).
- "Reading the curve": reading of Bishop Fig 9.9 and §9.3.2 -> figure 9.9, section 9.3.2 (both variants).
- "The interpolation threshold": Belkin Fig 1/§1, Nakkiran §2 (EMC is Definition 1 there; "definition" is not
  an allowed type), plus Bishop §9.3.2 which states both.
- "Reading the two branches": instructor's reading of Hastie Theorem 1 -> Eqs 6-8, §3.2.
- "The same curve along three different axes": Nakkiran §5/§6/§7 for the three named rows, Yilmaz §5 for 1/lambda,
  Bishop §9.3.2 which lists all four.
- "Choosing capacity when the curve has two minima": Prince §8.5 and Nakkiran §2 (critical regime), as the slide cites.
- "What survives and what does not": summary table -> \nosrc.
- B recreations of A crops cite the book figure: "Where the error comes from" (Prince 8.5), "Bias-variance
  Trade-off" (Bishop 4.7), "The trade-off, measured" (Bishop 4.8, Prince 8.9), "The same curve, in the ResNet
  layout" (Bishop 9.9), "Smoothness and Extra Capacity" (Prince 8.11).
- B measured experiments cite the result tested: "What is actually measured" (random features, min-norm) ->
  Bishop §9.3.2, Belkin §2; "The same shape at four noise levels" -> Prince Fig 8.10, §8.4; "Theory against
  measurement" -> Hastie Eq 8, §3.2; "More data can make a model worse" (sample-wise) -> Bishop §9.3.2, Nakkiran §7.

## Unverified
None.

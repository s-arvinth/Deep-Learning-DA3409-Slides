# Class 15 — Normalization: source tags

| variant | frames | \src | \nosrc |
|---|---|---|---|
| A_Textbook_Figures | 31 | 52 | 10 |
| B_Recreated_Figures | 34 | 56 | 10 |

(B has 5 more frames commented out with `%`; they are not frames and carry no tags.)

Checks: build + compare_pdfs IDENTICAL for A (45 pp) and B (45 pp).
`tools/check_sources.py` reports 13 (A) / 12 (B) ERRORs, all "malformed \src (needs three arguments)" —
the linter's regex `\\src(?!\{)` matches the deck's existing `\srcline{...}` macro (false positive, see
Class14.md). With that regex changed to `\\src(?![{a-z])` in a /tmp copy both decks give 0 errors, 0 warnings.
All tags are the first lines of the frame body.

## \srcdef keys
he2015, glorot2010, ioffe2015, ba2016, wu2018, santurkar2018, xiong2020 (all already in keys.md).
Fixed keys used: bishop2024, prince2023.

## Places checked
- Bishop 2024: §7.2.5 (symmetry breaking, He initialization, Eqs 7.19-7.23); §7.4.1 Eqs 7.48-7.50, Fig 7.7
  (height in metres vs platelet count example); §7.4.2 Eqs 7.51-7.57 (covariate shift vs smoothness, Santurkar);
  §7.4.3 Eqs 7.58-7.60 (small batches, split across GPUs, RNNs); Fig 7.8.
- Prince 2023: §7.5 intro; §7.5.1 Eqs 7.27-7.31 (7.31 He); §7.5.2 Eq 7.32; §7.5.3 Eq 7.33; Fig 7.7;
  chapter-7 Notes printed p.113 ("Initialization": Glorot differs by a factor of two).
- He et al. 2015 (arXiv): §2.2, Eqs (6)-(10) forward, (12)-(14) backward.
- Glorot & Bengio 2010 (PMLR PDF): §4.2.1, Eqs (10)-(12), normalized init Eq (16).
- Ioffe & Szegedy 2015 (arXiv): §3 Algorithm 1; §3.3 scale invariance BN(Wu)=BN((aW)u) and the 1/a gradient.
- Ba et al. 2016 (arXiv): §3, Eq (3).
- Wu & He 2018 (arXiv): §3.1 Eqs (1)-(7), Fig 2 (BN/LN/IN/GN).
- Santurkar et al. 2018 (arXiv): §2.1 ("noisy" BatchNorm), §3.1 (smoothing effect).
- Xiong et al. 2020 (arXiv): Fig 1 (Post-LN / Pre-LN), §3.2 (warm-up), §3.3 (gradients at initialization).

## Judgement calls
- "The Hessian of a linear model is a data matrix": instructor's derivation, nothing cited -> \nosrc.
- "The scale of a feature, squared": the kappa = (sigma2/sigma1)^2 proposition is not in either book (the
  course plan lists it as a literature result, without naming the paper); the slide's argument and its
  example (height vs platelet count) are Bishop §7.4.1 -> \src{bishop2024}{section}{7.4.1} only.
  B "Two ways the data alone can wreck the conditioning" (plot of the same claim) gets the same tag.
- "What normalizing the inputs does not fix": course's own list (whitening, log transform) -> \nosrc.
- "One layer is a multiplication, so depth is a power": no citation on the slide; the corollary is He et al.
  Eq (9) (product over layers) and Prince §7.5's exploding/vanishing discussion.
- "What one layer does to the variance": cites what the slide cites (Prince 7.28-7.30, Bishop 7.21-7.22).
- "When the layer is not square": Glorot Eqs 10-12 (§4.2.1) main; Prince 7.33 (§7.5.3) and Prince p.113 for the
  "omits the factor of two" remark.
- "Batch normalization", "Layer normalization": equations are in Bishop's notation -> Bishop first, then the
  original paper (Ioffe Algorithm 1 / Ba Eq 3).
- "A normalized layer does not care how big its weights are" (A): statement is Ioffe & Szegedy §3.3.
- "Why it works is not what was first claimed" (A): Santurkar §2.1 (injected distribution shift) and §3.1
  (smoothness) main, Bishop §7.4.2 second.
- "Where the normalizer sits in a residual block": own TikZ of post-norm vs pre-norm, as in Xiong Fig 1;
  warm-up remark §3.2, gradient argument §3.3.
- "A network that trains", "What each layer of defence buys": course summary diagram/table -> \nosrc.
- B measured-experiment frames cite the result they test: "What that does to the error surface" (redraw of
  Fig 7.7 in panel a) -> Fig 7.7, §7.4.1; "The three rules, and what they do at depth" (He 2/D, Glorot
  4/(D+D'), LeCun 1/D; LeCun not tagged, no source given) -> He Eq 10, §2.2, Glorot Eq 12;
  "Symmetry, drawn" -> Bishop §7.2.5; "The statistics do not stay where they were put" -> Bishop §7.4.2;
  "What it buys" (higher learning rates) -> Ioffe §3.3; "Smoothness, measured" -> Santurkar §3.1;
  "The batch size the statistics depend on" -> Bishop §7.4.3.

## Unverified
None. (Every number was seen in the source PDF. The only open point is the one noted above: the original
source of the feature-scale law for kappa is not named in the deck or the plan, so it is not tagged.)

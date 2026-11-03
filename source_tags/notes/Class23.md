# Class 23 — Convolution Filters

## Counts
- A_Textbook_Figures: 26 frames, 49 \src, 9 \nosrc. PDF IDENTICAL to baseline (38 pages).
- B_Recreated_Figures: 32 frames, 68 \src, 8 \nosrc. PDF IDENTICAL to baseline (40 pages).
- All tags are the first lines of the frame body.

## \srcdef keys (both variants)
cohen2016a, dumoulin2016, luo2016 (existing keys in keys.md, same text).

## Checker note
tools/check_sources.py reports "malformed \src" for every frame containing the deck's own \srcline{...} macro
(regex `\\src(?!\{)` matches the prefix of `\srcline`): 11 such errors in A, 13 in B, nothing else. A copy of the
checker with only that regex excluding \srcline gives 0 errors, 0 warnings for both. The tool was not changed.

## Verification
- Bishop 2024 (local PDF): §10.1.1, §10.2 (the 10^3 x 10^3 / 3 x 10^9 example), §10.2.1 (Eq. 10.1, matched-filter argument),
  §10.2.2 (Figs 10.2, 10.4), §10.2.3, §10.2.4, §10.2.5 (incl. 1x1 convolutions), §10.2.6 (pooling, local translation invariance).
- Prince 2023 (local PDF): §10.1 (Eqs 10.1-10.2, Fig. 10.1), §10.2.1 (Eq. 10.3, Fig. 10.2, cross-correlation footnote),
  §10.2.2-10.2.3 (Fig. 10.3), §10.2.4 (Eqs 10.4-10.5, Fig. 10.4), §10.2.5 (Fig. 10.5), §10.2.6 (Fig. 10.6),
  §10.3 (Eq. 10.6, Figs 10.9, 10.10), §10.4.1-10.4.2 (Figs 10.11, 10.12).
- dumoulin2016 (arXiv 1603.07285): the output-size formula with dilation is Relationship 15, §5.1 "Dilated convolutions".
- luo2016 (arXiv 1701.04128): "linearly by theory ... sub-sampling multiplicatively" is in §1 Introduction.
- cohen2016a (arXiv 1602.07576): §5 "Equivariance properties of CNNs" shows convolution/correlation are translation-equivariant.

## Judgement calls
- Notation (cross-correlation remark, Prince Eq. 10.3 footnote), "Building the symmetry in" (recap table), "The layer, in one table", Takeaways, References, Questions, Outline: \nosrc.
- "Proof, in both directions": \nosrc, the course's own proof (the result is tagged on the theorem frame).
- A "Why depth beats a wider kernel": \nosrc — own parameter table with no citation on the slide (the 3x3 argument is VGG's, but the slide does not draw on a cited text).
- "Three knobs, and one formula": dumoulin2016 §5.1 placed first because the proposition is exactly its Relationship 15; then Prince Fig. 10.3 / §10.2.2 / §10.2.3 and Bishop §10.2.3 / §10.2.4 as in the srcline.
- "The same operation, one index more" (A) / "2D data: Images" (B): srcline says "Figs. 10.8-10.10", but Prince Fig. 10.8 is the MNIST-1D result; tagged Eq. 10.6, §10.3 and Bishop §10.2.5 only (Figs 10.9, 10.10 are tagged on the B frames that show them).
- "What we want of the output" / "Some Definitions": Prince Fig. 10.1 kept (srcline) since the classify/segment bullets are its panels.
- B "2D Convolution" and "2D Convolution with Color Images" show photos of Prince Figs 10.9 and 10.10 (figs/*.JPG), so those figures are cited.
- B "A convolution is a dense layer with most of it deleted": the own weight-matrix figure (dense / kernel 3 / stride 2 matrices) is Prince Fig. 10.4's content, not Bishop Fig. 10.2 (which the A variant shows), so it cites Prince Fig. 10.4 / §10.2.4.
- B "One dimension, one kernel" (first of the two frames with this title) is the recreation of Prince Fig. 10.3.
- B "The recursion, checked" (own propagation experiment): tagged the result it tests, Prince §10.2.6 and luo2016 §1. B "How far back a unit can see, drawn" (3 frames): recreation of Prince Fig. 10.6.
- B "What a 3x3 kernel does" (own hand-chosen kernels): recreation in spirit of Bishop Fig. 10.4, cited as Fig. 10.4 / §10.2.2.
- B "Resolution, measured" (own experiment): tagged Bishop §10.2.6 (pooling's local translation invariance) and Prince §10.4.1-10.4.2 (down/up-sampling), the claims it measures.
- Exercise 10.1 (named in a srcline) is not an allowed type; covered by §10.2.1 / Eq. 10.1.
- Fukushima, LeCun 1989/1998, Yu & Koltun appear only in References; no tags.

## Unverified
1. "Convolution is the equivariant linear map" — \src{cohen2016a}{section}{5}: §5 contains the forward statement (convolution is translation-equivariant). The converse ("every translation-equivariant linear map is a convolution"), which the slide attributes to Cohen & Welling (2016), was not found in the arXiv text of that paper.

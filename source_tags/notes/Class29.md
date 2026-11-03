# Class 29 — RNNs for NLP tasks: source tags

| variant | frames | \src | \nosrc | check_sources | PDF vs baseline |
|---|---|---|---|---|---|
| A | 34 | 87 | 6 | 0 errors with the \srcline fix; official linter: 23 errors, all "malformed \src" false positives from `\srcline` | IDENTICAL (47 pages) |
| B | 37 | 95 | 6 | same: 0 with fix; 23 false positives with the official linter | IDENTICAL (50 pages) |

Linter false positive and bare `\newcommand` lines: as in Class27.md (tool not edited; checked with
`../slide-sources-work/tools_c27_29/check_sources_srcline.py`).

\srcdef keys (both variants): graves2013b, inan2017, jurafsky_slp3 (draft of August 19, 2026), mikolov2010,
pascanu2014, press2017. All existing keys; no new key.

## Verified places
- Jurafsky (Aug 2026 draft) outside Ch. 14: Eqs. 3.14-3.15 in §3.3 (perplexity), Eq. 3.42 in §3.7 (entropy), Eq. 7.51 in §7.6.3 (temperature), §7.6.2 (random sampling), Eq. 7.58 in §7.7.1 (LM cross-entropy). Ch. 14: Eqs. 14.10-14.11 and Fig. 14.6 in §14.2.2; Eqs. 14.12-14.14 in §14.2.3; Fig. 14.7 in §14.3.1 (the *Janet will back the bill* example is in it); Eq. 14.15, Fig. 14.8 in §14.3.2; Fig. 14.10 in §14.4.1; Eqs. 14.16-14.18, Figs. 14.11-14.12 in §14.4.2; Fig. 14.15 in §14.6.
- Goodfellow: Eqs. 10.29-10.33 and Figs. 10.7-10.8 in §10.2.3; Eq. 10.34 in §10.2.3; Fig. 10.13 in §10.5; Fig. 10.14 in §10.6.
- press2017 §3 Weight Tying (arXiv 1608.05859v3); inan2017 §4 Theoretically driven reuse of word embeddings (arXiv 1611.01462); graves2013b §2, Eqs. (11)-(12) deep (stacked) RNNs (arXiv 1303.5778); pascanu2014 §3.2 (deep input/transition/output, shortcut) and §3.2.4 Stack of hidden states (arXiv 1312.6026); mikolov2010 §2 Model description.

## Judgement calls
- "Where we are coming from", "Why the reader alone is not enough", "The approach": roadmap/recap frames whose task descriptions are J's §14.2-14.4 and §14.6; tagged with those sections rather than \nosrc (as in Classes 27-28).
- "The problems, with their examples": *a wonderfully warm little film* is the course's own example (not in the draft); *So long and thanks for* is J's running example; tagged J Fig. 14.7 (the tagging example) and the task sections.
- "The training loss is the log of the perplexity": the proposition is J's Exercise 7.4 ("Prove that the perplexity ... equal to the exp of the mean cross-entropy"); exercises are not a tag type, so tagged the equations it uses (3.14-3.15, 7.58, 3.42) with their sections.
- "Why the read-out matters": the proposition is the course's; tagged only J §14.3.2 (which motivates pooling in words, as the \srcline says).
- "Bidirectionality": tagged Eqs. 14.16-14.18 and §14.4.2; Figs. 14.11-14.12 (named in the \srcline) are drawn on the neighbouring frames and tagged there.
- "(d) Encoder–decoder": tagged J §14.7, §14.8; the figure range 14.17-14.22 in the \srcline is Class 27's material, not shown here.
- Shannon (1951), Schuster & Paliwal (1997), Pollack (1990): cited in \srcline as attributions the books themselves make (J §14.3.3, J §14.4.2, G §10.6); paywalled, not tagged (see Unverified).
- Schmidhuber (2012) and Socher et al. (2011, 2013): named on the slides as attributions G makes (§10.2.3, §10.6); the slide content is the book's, so not tagged. Graves (2013a) is not cited on the slides; the "Graves et al. (2013)" on the stacking slide is graves2013b (deep RNNs for speech).
- B-only experiment frames: the experiment -> J §14.2.2, §14.3.3, §3.3; the perplexity -> J Eqs. 3.14, 7.58, §3.3; the samples -> J §14.3.3, §7.6.3 (the results each tests). The training text (henry1905, O. Henry) is data, not a source of the slide's content; not tagged.
- "The block, in one table" (summary), Takeaways, References, Questions: \nosrc.

## Unverified (named on the slides, not tagged: no copy could be opened, so no place could be checked)
1. shannon1951 (Bell System Technical Journal, paywalled) — "Generation: sampling".
2. schuster1997 (IEEE Trans. Signal Processing, paywalled) — "Bidirectionality".
3. pollack1990 (Artificial Intelligence 46, paywalled) — "The tree-shaped cousin".

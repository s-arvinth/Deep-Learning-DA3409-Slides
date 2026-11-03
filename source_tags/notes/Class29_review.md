# Class 29 review (independent reviewer)

Tags checked: A 34 frames (87 \src + 6 \nosrc), B 37 frames (95 \src + 6 \nosrc), 6 \srcdef per deck.
Jurafsky (Aug 19, 2026 draft): Eqs. 3.14-3.15 in 3.3, Eq. 3.42 in 3.7, 7.6.2, Eq. 7.51 in 7.6.3, Eq. 7.58 in 7.7.1;
Fig. 14.5/Eqs. 14.4-14.9 in 14.2/14.2.1; Fig. 14.6, Eqs. 14.10-14.11 in 14.2.2 (Fig. 14.6 is "So long and thanks for");
Eqs. 14.12-14.14 in 14.2.3; Fig. 14.7 in 14.3.1; Eq. 14.15, Fig. 14.8 in 14.3.2; Fig. 14.9 in 14.3.3; Fig. 14.10 in 14.4.1;
Eqs. 14.16-14.18, Figs. 14.11-14.12 in 14.4.2; Fig. 14.15 in 14.6. Goodfellow: Eqs. 10.29-10.34, Figs. 10.7-10.8 in 10.2.3;
Fig. 10.13 in 10.5; Fig. 10.14 in 10.6.

## Changes (frame, old -> new, reason)
Missing (4 frames, both variants):
- "(a) Sequence labelling", "(b) Sequence classification", "(c) Language modelling", "(d) Encoder--decoder":
  added \src{jurafsky_slp3}{figure}{14.15} as first tag. Their figures arch_labelling / arch_classification / arch_lm /
  arch_encdec are drawn "in the layout of Jurafsky & Martin (2026), Fig. 14.15" (B src/figures/fig08_four_architectures.py),
  one panel each, so rule 9 applies (redrawn book figure).
Less specific (1 frame, both variants):
- "The problems, with their examples": added \src{jurafsky_slp3}{figure}{14.6} after Fig. 14.7. The next-word example
  "So long and thanks for -> all" is Fig. 14.6 (in 14.2.2); only section 14.2 was tagged.
srcdef (4, both variants; keys.md rows updated identically):
- graves2013b: "ICASSP 2013, 2013" -> "ICASSP 2013; arXiv:1303.5778, 2013"
- inan2017: "ICLR 2017, 2017" -> "ICLR 2017; arXiv:1611.01462, 2017"
- pascanu2014: "ICLR 2014, 2014" -> "ICLR 2014; arXiv:1312.6026, 2014"
- press2017: "EACL 2017, 2017" -> "EACL 2017; arXiv:1608.05859, 2017"

## Unverified / left untagged
1. shannon1951 ("Generation: sampling"): J 14.3.3 attribution of the sampling idea; slide content is J's. Not tagged.
2. schuster1997 ("Bidirectionality"): J 14.4.2 attribution; the definition is J Eqs. 14.16-14.18. Not opened, not tagged.
3. pollack1990 ("The tree-shaped cousin"): G 10.6 attribution; content is G Fig. 10.14 / 10.6. Not opened, not tagged.
None of the three is what the slide is based on, so tagging them would be over-tagging even with a copy.

Checks: check_sources 0 errors (A, B); verify_book_numbers 0 not found; builds IDENTICAL (47 / 50 pages).

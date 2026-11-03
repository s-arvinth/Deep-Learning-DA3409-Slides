# Class 27 — Encoder–Decoder / Seq2Seq: source tags

| variant | frames | \src | \nosrc | check_sources | PDF vs baseline |
|---|---|---|---|---|---|
| A | 33 | 80 | 7 | 0 errors with \srcline fix (see below); official linter: 17 errors, all "malformed \src" false positives | IDENTICAL (48 pages) |
| B | 37 | 87 | 7 | 0 errors with \srcline fix; official linter: 18 errors, all the same false positive | IDENTICAL (53 pages) |

**Linter false positive.** `tools/check_sources.py` line 59 tests `\\src(?!\{)` after removing only `\srcdef`, so every frame
that contains the deck's own `\srcline{...}` macro is reported as "malformed \src (needs three arguments)". No tag is
malformed. I did not edit the tool; I checked with a copy that also strips `\srcline`
(`../slide-sources-work/tools_c27_29/check_sources_srcline.py`, the only change). With it: 0 errors, 0 warnings.
(The same false positive appears in Class 24, which also uses `\srcline`.)

**Preamble.** The three `\newcommand` lines are written without the trailing `% ...` comments of the brief, because the
linter only accepts the bare lines (with the comment the line counts as "tag inside a % comment").

\srcdef keys (both variants): bahdanau2015, bengio2015, cho2014, jurafsky_slp3 (draft of August 19, 2026), luong2015,
mikolov2010, ranzato2016, sutskever2014, williams1989. All existing keys from keys.md; no new key created.

## Verified places (papers)
- cho2014 §2.2 RNN Encoder–Decoder (arXiv 1406.1078v3)
- sutskever2014 §2 The model; §3.2 Decoding and Rescoring (beam search) (arXiv 1409.3215v3)
- bahdanau2015 §3.1 Decoder: general description (attention); §5.2.1 Alignment (arXiv 1409.0473)
- luong2015 §3.1 Global Attention (dot / general scores) (arXiv 1508.04025v5)
- bengio2015 §2.3 Inference (errors compound), §2.4 Scheduled Sampling (arXiv 1506.03099)
- ranzato2016 §3.1.1 Cross Entropy Training (exposure bias) (arXiv 1511.06732v7)
- mikolov2010 §2 Model description (ISCA archive PDF)
- williams1989 §2.3 Teacher-Forced Real-Time Recurrent Learning (author's version, gwern.net/doc/ai/nn/rnn/1989-williams-2.pdf, "Appears in Neural Computation 1, pp. 270-280")

## Judgement calls
- "Where we are coming from": half recap, half statement of today's problem; tagged J §14.7 and G §10.4 (the problem statement comes from their openings) rather than \nosrc.
- "The problem": the word-order point (green witch / bruja verde) is J §13.1.1 (Word Order Typology, example 13.4); tagged with §14.7.
- "The probability of a whole sequence": the theorem is the conditional chain rule, so J Eq. 14.31 first, then 14.28, 14.8-14.9 (unconditional versions cited on the slide) and G Eq. 10.31.
- "The recurrent language model": J Fig. 14.5 is tagged (the "state remembers the words before" point is that figure's panel b); Eqs. 14.4-14.7 are in §14.2.1.
- "The equations": G Eq. 10.35 (cited in the \srcline) is the conditional-independence factorisation, not what the slide uses; tagged G §10.2.4 (the "extra input / initial state / both" list) instead.
- "Attention, defined": the bilinear score W_s on the slide is J Eq. 14.38, so the range is 14.34-14.38 (the \srcline says 14.34-14.37). Figs. 14.21-14.22 are the next two frames' figures, not tagged here. Luong §3.1 for the bilinear ("general") score.
- "Where the context can be" (convex hull): \nosrc — the course plan marks it "in neither book"; the course's own one-line result.
- "What the beam costs": the O(km|V|) bound is the course's own; tagged the parts from the book: Eq. 13.16 (length normalisation), §13.4, §13.4.1 (MBR).
- "Choosing the output": greedy Eq. 13.14 and beam score Eq. 13.15, §13.4, Sutskever §3.2 (beam search decoder).
- "Generating, one token at a time": Shannon (1951) in the caption is J's historical attribution of the sampling idea; not tagged (paywalled, and the slide is J Fig. 14.9).
- "Everything through one vector": G §10.4 added in both variants (its last paragraph states the too-small-context limitation the slide describes; B's caption cites it).
- B-only experiment frames (trained model): cite the book result each one tests — teacher forcing vs inference: J Figs. 14.19, 14.17, §14.7.1; bottleneck experiment/result: J §14.8; beam search on the trained model: J Fig. 13.7, §13.4.
- Tags are the first lines of every frame; no frame body starts with `{`.

## Unverified
- williams1989 §2.3: section number read from the authors' preprint ("Appears in Neural Computation"), not the published journal PDF (paywalled).
- shannon1951 (Bell System Technical Journal, paywalled): named in the caption of "Generating, one token at a time" as J's historical attribution; not tagged, no copy opened. (2 items in total)

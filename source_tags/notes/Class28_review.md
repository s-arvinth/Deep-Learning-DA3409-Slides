# Class 28 review (independent reviewer)

Tags checked: A 30 frames (65 \src + 6 \nosrc), B 33 frames (71 \src + 6 \nosrc), 10 \srcdef per deck.
Jurafsky (Aug 19, 2026 draft): 14.5 (example 14.19, LSTM Eqs. 14.20-14.27, Fig. 14.13), 14.5.1 (Fig. 14.14).
Goodfellow: Eqs. 10.36-10.39 and Fig. 10.15 in 10.7; 10.8 (ESN, Jaeger and Haas 2004 named); 10.9.1 (skip
connections, Lin et al. 1996 named); 10.9.2 (leaky units, Mozer 1992); Fig. 10.16 / Eqs. 10.40-10.44 in 10.10.1;
Eqs. 10.45-10.47 in 10.10.2; Fig. 10.17, Eqs. 10.48-10.49 in 10.11.1; Eqs. 10.50-10.52 in 10.11.2; Fig. 10.18 in 10.12.
Paper sections accepted as recorded by the tagger.

## Changes (frame, old -> new, reason)
Over-tagged (1 tag, both variants):
- "Clipping the gradient": removed \src{mikolov2012}{section}{3.3.2}. The slide states and proves norm clipping
  (G Eqs. 10.48-10.49, Pascanu 3.2) and the cliff argument (Pascanu 2.3); Mikolov's thesis is only the element-wise
  clipping variant that G 10.11.1 mentions and the slide does not show. Its \srcdef was then unused and was removed
  (the key stays in keys.md).
srcdef (5, both variants; keys.md rows updated identically):
- chung2014: "arXiv preprint (NeurIPS 2014 DL Workshop), 2014" -> "NeurIPS 2014 Deep Learning Workshop; arXiv:1412.3555, 2014"
- jozefowicz2015: "ICML 2015, 2015" -> "ICML 2015 (PMLR 37), 2015"
- pascanu2013: "ICML 2013, 2013" -> "ICML 2013; arXiv:1211.5063, 2013"
- greff2017: "IEEE TNNLS 28(10), 2017" -> "IEEE TNNLS 28(10); arXiv:1503.04069, 2017"
- cho2014 (shared with Class 27): "EMNLP 2014, 2014" -> "EMNLP 2014; arXiv:1406.1078, 2014"

## Unverified (re-examined, still untagged)
1. bengio1994: searched for an open copy; found only third-party presentation slides (cs.nyu.edu), not the
   paper. The slides' bound is Pascanu 2.1's form, already tagged; Bengio et al. is attribution only.
2. gers2000: no open copy of the Neural Computation version found (only the 1999 IDSIA-01-99 report). The forget
   gate content is tagged via J 14.5 / G 10.10.
3. lin1996: the author's page copy (clgiles.ist.psu.edu) timed out. G 10.9.1 attribution; content tagged there.
4. jaeger2004: not opened (Science, paywalled). G 10.8 attribution; content tagged there.

Checks: check_sources 0 errors (A, B); verify_book_numbers 0 not found; builds IDENTICAL (46 / 49 pages).

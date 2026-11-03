# Class 14 — Convergence: independent review of source tags

Tags checked: A 83 (72 \src + 11 \nosrc after review; 83 before), B 102 (89 + 13; 102 before), plus 10 \srcdef per deck.
Checks after edits: check_sources 0 errors / 0 warnings (A, B); verify_book_numbers 0 not found;
build + compare_pdfs IDENTICAL (A 59 pp, B 58 pp).

## Verified against the sources
- Bishop 2024 §7.1.1 (7.7-7.14, Fig 7.2), §7.3 (7.24-7.30, condition number after 7.30, Fig 7.3), §7.3.1
  (7.31-7.34, Figs 7.4-7.6, Alg 7.3, Nesterov remark "for batch GD ... less effective for SGD"), §7.3.2 (7.35-7.38),
  §7.3.3 (7.39-7.47, axis-aligned caveat, beta2 = 0.99, Alg 7.4) — booktext.
- Prince 2023 §6.2.2, §6.3 Eq 6.11, §6.3.1 Eq 6.12 Fig 6.8, §6.4 Eqs 6.13-6.16 Fig 6.9, §6.5, Fig 6.7;
  Notes p.93 (warm-up, Goyal; AdaGrad/RMSProp/Adam/AMSGrad) and p.94 (SGD vs Adam).
- Kingma & Ba (arXiv v9): Algorithm 1 (p.2), §3 "Initialization bias correction" Eqs (1)-(4).
- Hinton et al. 2012 lecture 6 slides: "rmsprop: A mini-batch version of rprop" is the 29th slide (counted
  independently in the fetched text; the PDF itself could not be downloaded in the shell to run pdfinfo).
- Polyak 1964: Russian original (mathnet.ru full text, Zh. Vychisl. Mat. Mat. Fiz. 4:5, 791-803) has §1
  "Convergence of multistep methods" and §2 "Examples, computational aspects"; Theorem 9 in §2 gives
  (part 2) the gradient method's optimal factor (M-m)/(M+m) at alpha = 2/(M+m), and (part 3) the heavy-ball
  method (9) with optimal q2 = (sqrt M - sqrt m)/(sqrt M + sqrt m) at alpha = 4/(sqrt M + sqrt m)^2,
  beta = q2^2; the text after it compares iteration counts ~kappa/2 vs ~sqrt(kappa)/2 and names the
  "heavy ball" method.

## Changes (frame: old -> new, reason)
1. A "What momentum buys, precisely": \src{polyak1964}{page}{1-17} -> \src{polyak1964}{section}{2}
   (less specific: full page range; the theorem is Theorem 9(3) in §2).
2. B "What that costs in iterations": same change (plots the two factors of Theorem 9(2)/(3)).
3. B "Momentum: ": same change (states Theorem 9(3) and 9(2)).
4. A "The best a fixed step size can do": \nosrc -> \src{polyak1964}{section}{2}
   (missing: eta* = 2/(lmin+lmax), rho* = (kappa-1)/(kappa+1) and the ~kappa/2 step count are Polyak's
   Theorem 9(2) and the discussion after it; the B deck already tags the same proposition with polyak1964).
5. B "Optimal learning rate for a quadratic": \nosrc -> \src{polyak1964}{section}{2} (same reason).

By kind: wrong place 0, missing 2, less specific 3, over-tagged 0, srcdef 0.

## Still unverified
- A and B "The guarantee behind the name": \src{nesterov2004}{chapter}{2}. The accelerated-rate theorem is
  probably in §2.2 ("Optimal methods") but no table of contents below chapter level could be opened
  (Springer page rate-limited / not readable); left at chapter level.
- polyak1964 section 2: section seen in the Russian original only; the English translation (USSR Comput.
  Math. Math. Phys. 4(5), 1-17, the edition in \srcdef) is paywalled. Sections of a translation keep their
  numbers, so the tag is expected to hold, but the English text was not seen.
- Not re-opened by the reviewer (tagger's verification accepted): duchi2011 Eq 1 / §1.1, robbins1951 (6), (26),
  §3, goyal2018 §2.2, loshchilov2017 Eq 5 / §3, reddi2018 §3 / §4, loshchilov2019 Algorithm 2 / §2.

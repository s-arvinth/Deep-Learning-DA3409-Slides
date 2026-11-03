# Class 25 — Convolution Filters: What They Learn

| variant | frames | \src | \nosrc |
|---|---|---|---|
| A_Textbook_Figures | 32 | 53 | 7 |
| B_Recreated_Figures | 20 | 15 | 12 |

Checks: `tools/check_sources.py` reports ERRORs "malformed \src (needs three arguments)" (A 16, B 6) on exactly the
frames that contain the deck's own `\srcline{...}` macro — a checker false positive (`\\src(?!\{)` matches `\srcline`;
only `\srcdef` is excluded). With that regex also excluding `\srcline` (patched copy in /tmp; tool not modified):
0 errors, 0 warnings for both. PDFs: both IDENTICAL to baseline (A 44 pages, B 25 pages). All tags at frame top.

## \srcdef keys
A: gatys2015, goodfellow2015, selvaraju2017, simonyan2015, yosinski2015, zeiler2014
B: simonyan2015
(all existing keys; no new key)

## Where the numbers were checked
- Bishop: 10.2.7, 10.2.8 (138M / 103M VGG-16; 14M images, 22,000 categories, 99.9%, 25.5%, 15.3%, ~3%, ~5%), 10.3.1–10.3.5, 10.6;
  Figs 10.9–10.18, 10.32; Eqs (10.6)–(10.8) in 10.3.1, (10.9)–(10.10) in 10.3.3, (10.11) in 10.3.4, (10.12) in 10.3.5, (10.13)–(10.17) in 10.6.
- Prince: 10.4.3 "Changing the number of channels" + Fig 10.14 (1×1 convolution); 10.5.1 (16.4%).
- Papers fetched: simonyan2015 §2.1 Architecture, Table 1 (config D), Table 2 (D = 138M); zeiler2014 §4 "Convnet Visualization";
  yosinski2015 §3 "Visualizing via Regularized Optimization" (L2 decay, blur, clipping); selvaraju2017 §3 "Grad-CAM", Eqs (1)–(2)
  (arXiv v4 numbering); goodfellow2015 §3 linear explanation (activation grows by ε‖w‖1 / mn), §4 fast gradient sign method;
  gatys2015 Methods, Eqs (1)–(7) (content loss, Gram matrix, style loss, total loss; printed pp. 10–12).

## Judgement calls
- \nosrc: Outline, "Opening the trunk" (overview table), Notation, "One gradient, five uses" (summary table), Takeaways, References, Questions.
- "The 1×1 convolution": the deck's \srcline says Prince §10.5.1, Fig 10.14; the content is printed in Prince §10.4.3 and the Fig 10.14
  caption (the figure only sits physically on a §10.5.1 page), so tagged figure 10.14 + section 10.4.3.
- Frames stating a result whose figure is on the next frame (A layer counted; VGG-16 layer by layer; Gabor filter; Reading the first layer;
  Deeper layers; Or optimise the image; Saliency) cite equation/section, not the figure.
- Captions "from Zeiler & Fergus / Yosinski / Selvaraju / Goodfellow / Eykholt / Gatys" on Bishop figures are attribution of the Bishop figure: not tagged.
- "Turning the gradient against the network": FGSM is goodfellow2015 §4 and the "why a tiny ε is enough" proposition is §3; both tagged after Bishop.
- "DeepDream": Mordvintsev et al. (2015) is a blog post with no section/page/figure numbers; not tagged (Bishop Eq 10.12, §10.3.5 only).
  Bishop "Exercise 10.10" is not a citable type.
- B "Composition, computed": the course's own measured check of the composition proposition; tagged with the result it tests (Bishop §10.2.8).
- B "Reach grows with depth": recreation of Bishop Fig 10.9.
- B frames "1. Overview …" to "5. Summary: Duality Between Operations" (backprop through conv / pooling / upsampling): no source named on
  the slides or in the references; treated as the course's own derivations, \nosrc.

## Unverified
- A "Reading the first layer directly": \srcline also names Hyvärinen et al. (2009), Natural Image Statistics (Springer). The book could not be
  opened (library/purchase; Springer page returned nothing), so no chapter/page could be verified; frame tagged only bishop2024 section 10.3.2
  and no hyvarinen2009 tag added.

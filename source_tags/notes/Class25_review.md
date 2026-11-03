# Class 25 review (Convolution Filters: What They Learn)

Tags checked: A 32 frames (53 \src, 7 \nosrc); B 20 frames (15 \src, 12 \nosrc; most of B's content frames are commented out). \srcdef lines (A 6, B 1) consistent with keys.md.

Checked against: Bishop 10.2.7 (parameter count (M^2 C_in + 1) C_out, Fig 10.9), 10.2.8 (LeNet, 14M images / 22,000 categories, 99.9%, 25.5%, 15.3%, ~3% / ~5%, VGG-16 layer by layer, 138M / 103M, composed sub-filters as inductive bias),
10.3.1 (Hubel and Wiesel, grandmother cells, neocognitron, Eqs 10.6-10.8), 10.3.2 (Figs 10.12-10.14, Hyvarinen remark, pre-activation, blurring and clipping), 10.3.3 (Eqs 10.9-10.10, 14x14 for VGG), 10.3.4 (Eq 10.11, Figs 10.16-10.17), 10.3.5 (Eq 10.12, Fig 10.18), 10.6;
Prince 10.4.3 (1x1 convolution text references Fig 10.14) and 10.5.1 (16.4%).

## Changes
None. Tags are correct, complete and specific.
- Agree with \nosrc for B frames "1. Overview ..." to "5. Summary: Duality Between Operations": the course's own derivations, and the slides name no source.
- DeepDream (A): Mordvintsev et al. (2015) is a blog post with no section, page or figure numbers, so no tag. The frame keeps Bishop Eq 10.12 / sec. 10.3.5 only. I agree.

## Checks
check_sources: A and B 0 errors 0 warnings. verify_book_numbers: 0 not found. PDFs: A IDENTICAL (44 pages), B IDENTICAL (25 pages).

## Unverified
- A "Reading the first layer directly": the \srcline names Hyvarinen, Hurri & Hoyer (2009), Natural Image Statistics. I tried the open preprint (naturalimagestatistics.net/nis_preprintFeb2009.pdf) with web_fetch and it returned an empty body, so no chapter or page could be verified. No hyvarinen2009 tag added. Still open.
- Paper places taken on the tagger's word: simonyan2015 Tables 1-2 / sec. 2.1, zeiler2014 sec. 4, yosinski2015 sec. 3, selvaraju2017 Eqs 1-2 / sec. 3, goodfellow2015 secs. 3-4, gatys2015 Eqs 1-7. They agree with these papers' known structure.

# Class 24 review (Applications: Detection, Segmentation)

Tags checked: A 39 frames (67 \src, 8 \nosrc); B 38 frames (70 -> 71 \src, 7 \nosrc). All \srcdef lines (A 9, B 7) checked against keys.md: consistent.

Checked against: Bishop ch. 10 text (10.1, 10.4 intro, 10.4.1-10.4.6, 10.5, 10.5.1-10.5.4, Figs 10.19-10.31 captions and placement);
Prince 10.4.2, 10.5.1-10.5.3 text and Figs 10.13, 10.15-10.21 (all AlexNet/VGG/ImageNet/YOLO numbers on the slides appear in 10.5.1/10.5.2);
Redmon et al. arXiv v5 fetched (sec. 2: S x S x (B*5+C), S=7, B=2, C=20; sec. 2.2: Eq. 3 loss, lambda_coord=5, lambda_noobj=.5).
B "Learned up-sampling" image figs/Image_Report.png viewed: it is Prince Fig 10.13 (panels a-d), so the tag is correct.

## Changes
- B "What the output structure decides": added `\src{redmon2016}{section}{2}` after the three Prince tags (missing).
  Reason: the figure gives YOLO's S=7, B=2, C=20 output (7x7x30); C=20 and the S x S x (5B+C) shape are in Redmon sec. 2, not in Prince or Bishop.

No changes to A.

## Checks
check_sources: A 0 errors 0 warnings; B 0 errors 0 warnings. verify_book_numbers: 0 not found. PDFs: A IDENTICAL (54 pages), B IDENTICAL (45 pages).

## Unverified
- Paper places taken on the tagger's word (not re-fetched here): sermanet2014 sec. 3.5, girshick2015 sec. 2, ren2015 sec. 3.1, long2015 sec. 3, badrinarayanan2017 sec. 3, dumoulin2016 sec. 4.1/4.2, krizhevsky2012 Table 2 / sec. 3.5, simonyan2015 Table 7 / sec. 2.2. They agree with these papers' known section structure.

#!/usr/bin/env python3
"""Compare two PDFs page by page: same page count and identical renders.

    python3 tools/compare_pdfs.py before.pdf after.pdf
Renders at 50 dpi; reports every page whose pixels differ.  Exit 1 on any difference."""
import subprocess, sys, tempfile, glob, os
import numpy as np
from PIL import Image
a, b = sys.argv[1:3]
with tempfile.TemporaryDirectory() as t:
    for tag, p in (("a", a), ("b", b)):
        subprocess.run(["pdftoppm", "-r", "50", "-png", p, os.path.join(t, tag)], check=True)
    A = sorted(glob.glob(t + "/a-*.png")); B = sorted(glob.glob(t + "/b-*.png"))
    bad = []
    if len(A) != len(B): bad.append(f"page count {len(A)} vs {len(B)}")
    for i, (x, y) in enumerate(zip(A, B), 1):
        X = np.asarray(Image.open(x).convert("L"), int); Y = np.asarray(Image.open(y).convert("L"), int)
        if X.shape != Y.shape or np.abs(X - Y).max() > 0: bad.append(f"page {i} differs")
    print(f"{os.path.basename(b)}: {'IDENTICAL' if not bad else 'DIFFERENT: ' + '; '.join(bad[:10])} ({len(A)} pages)")
    sys.exit(1 if bad else 0)

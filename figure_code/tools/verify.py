#!/usr/bin/env python3
"""Re-run every figure script of the given classes from a clean output/ and compare
each result with the figure the deck loads.

    python3 figure_code/tools/verify.py 5 6 26     # classes 5, 6, 26
    python3 figure_code/tools/verify.py all

For each figure: mean absolute pixel difference (0-255) of 60-dpi renders and the
share of pixels differing by more than 32.  match = mean < 1 and share < 0.5%.
Needs poppler (pdftoppm), numpy, Pillow."""
import glob, os, re, shutil, subprocess, sys, tempfile
import numpy as np
from PIL import Image
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def render(path, out):
    if path.lower().endswith(".pdf"):
        subprocess.run(["pdftoppm", "-r", "60", "-png", "-singlefile", path, out], check=True)
        return Image.open(out + ".png").convert("L")
    return Image.open(path).convert("L")

def compare(a, b):
    with tempfile.TemporaryDirectory() as t:
        A = render(a, os.path.join(t, "a")); B = render(b, os.path.join(t, "b"))
        if A.size != B.size:
            B = B.resize(A.size)
        d = np.abs(np.asarray(A, float) - np.asarray(B, float))
        return d.mean(), (d > 32).mean()

def deck_file(cdir, name):
    for v in ("B_Recreated_Figures", "A_Textbook_Figures", "."):
        for ext in (".pdf", ".png", ".jpg"):
            p = os.path.join(cdir, v, "figs", name + ext)
            if os.path.exists(p): return p
    return None

def main(args):
    classes = sorted(int(re.search(r"main_class(\d+)", d).group(1)) for d in glob.glob(ROOT + "/figure_code/main_class*")) if args == ["all"] else [int(a) for a in args]
    worst = 0
    for n in classes:
        fdir = os.path.join(ROOT, "figure_code", f"main_class{n}")
        cdir = glob.glob(os.path.join(ROOT, f"Class{n:02d}_*"))[0]
        shutil.rmtree(os.path.join(fdir, "output"), ignore_errors=True)
        helpers = {"common", "tape", "models", "experiment", "draw", "bookdraw", "bishopdraw", "data"}
        for py in sorted(glob.glob(fdir + "/*.py")):
            name = os.path.splitext(os.path.basename(py))[0]
            if name in helpers: continue
            r = subprocess.run([sys.executable, os.path.basename(py)], cwd=fdir, capture_output=True, text=True, timeout=600)
            outs = glob.glob(os.path.join(fdir, "output", name + ".*"))
            ref = deck_file(cdir, name)
            if r.returncode or not outs or not ref:
                print(f"class {n} {name}: FAIL rc={r.returncode} out={bool(outs)} ref={bool(ref)} {r.stderr[-200:]}"); worst = 2; continue
            m, s = compare(outs[0], ref)
            st = "match" if m < 1 and s < 0.005 else "DIFF"
            if st != "match": worst = max(worst, 1)
            print(f"class {n:2d} {name:32s} mean={m:6.3f} share={100*s:5.2f}% {st}")
    sys.exit(worst)

if __name__ == "__main__":
    main(sys.argv[1:])

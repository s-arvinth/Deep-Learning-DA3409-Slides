#!/usr/bin/env python3
"""Check that every number cited from the five fixed books with \\src exists in the
book PDF named in source_tags/SOURCES_GUIDE.md (figure caption, equation tag, section
heading, table, algorithm, chapter).  Prints each tag that is not found.

    python3 tools/verify_book_numbers.py            # all decks
Needs pdftotext.  Text of each book is cached in ../slide-sources-work/booktext/."""
import glob, os, re, subprocess, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TA = os.path.abspath(os.path.join(ROOT, "..", ".."))
BOOKS = {
 "bishop2024": TA + "/Christopher M. Bishop, Hugh Bishop - Deep Learning_ Foundations and Concepts-Springer (2024).pdf",
 "prince2023": TA + "/prince2023udl.pdf",
 "rojas1996": TA + "/neuron.pdf",
 "goodfellow2016": TA + "/Reference-Textbooks/DeepLearningBook_GoodfellowBengioCourville/book/Deep_Learning_Goodfellow_Bengio_Courville.pdf",
 "jurafsky_slp3": TA + "/Reference-Textbooks/SLP3_JurafskyMartin/book/ed3book_aug26.pdf",
}
CACHE = os.path.join(ROOT, "..", "slide-sources-work", "booktext"); os.makedirs(CACHE, exist_ok=True)

def index(key):
    txt = os.path.join(CACHE, key + ".txt")
    if not os.path.exists(txt):
        subprocess.run(["pdftotext", BOOKS[key], txt], check=True)
    t = open(txt, errors="ignore").read()
    t = t.replace("\u2013", "-")
    idx = {
      "figure": set(re.findall(r"(?:Figure|Fig\.)\s+(\d+\.\d+[a-z]?)", t)),
      "equation": set(re.findall(r"\((\d+\.\d+[a-z]?)\)", t)),
      "section": set(re.findall(r"(?m)^\s*(\d+(?:\.\d+)+)\s+\S", t)) | set(re.findall(r"(?:Section|§)\s*(\d+(?:\.\d+)+)", t)),
      "table": set(re.findall(r"Table\s+(\d+(?:\.\d+)+)", t)),
      "algorithm": set(re.findall(r"Algorithm\s+(\d+(?:\.\d+)+)", t)),
      "chapter": set(re.findall(r"(?m)^\s*(\d+)\.\d+\s+\S", t)) | set(re.findall(r"Chapter\s+(\d+)", t)),
      "pages": t.count("\f") + 1,
    }
    return idx

def main():
    idx = {}; bad = 0; n = 0
    for p in sorted(glob.glob(ROOT + "/Class*/[AB]_*/main.tex")):
        s = "\n".join(re.sub(r"(?<!\\)%.*", "", l) for l in open(p, errors="ignore").read().split("\n"))
        for f in re.finditer(r"\\begin\{frame\}(?:\[[^\]]*\])?(?:\{((?:[^{}]|\{[^{}]*\})*)\})?(.*?)\\end\{frame\}", s, re.S):
            for k, typ, num in re.findall(r"\\src\{([^}]*)\}\{([^}]*)\}\{([^}]*)\}", f.group(2)):
                if k not in BOOKS: continue
                n += 1
                if k not in idx: idx[k] = index(k)
                I = idx[k]
                ends = num.split("-")
                if typ == "page":
                    ok = all(e.isdigit() and int(e) <= I["pages"] + 50 for e in ends)
                else:
                    ok = all(e in I.get(typ, set()) for e in ends)
                if not ok:
                    bad += 1
                    print(f"{os.path.relpath(p, ROOT).split('/')[0][:7]} {p.split('/')[-2][0]}  '{(f.group(1) or '')[:45]}'  \\src{{{k}}}{{{typ}}}{{{num}}}")
    print(f"\n{n} book tags checked, {bad} not found")
    sys.exit(1 if bad else 0)

main()

#!/usr/bin/env python3
"""Check figure_code/ against REQUIREMENTS.txt: every .py in main_classN is a figure
name loaded by that class's decks or a declared helper, and every loaded figure that
a src/ module writes has a script.  Run from the repository root."""
import glob, os, re, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HELPERS = {"common", "tape", "models", "experiment", "draw", "bookdraw", "bishopdraw", "data"}
bad = 0
for cdir in sorted(glob.glob(os.path.join(ROOT, "Class*"))):
    n = int(re.search(r"Class(\d+)", cdir).group(1))
    names = set()
    for t in glob.glob(cdir + "/**/main.tex", recursive=True):
        s = re.sub(r"(?<!\\)%.*", "", open(t, errors="ignore").read())
        names |= set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", s))
    fdir = os.path.join(ROOT, "figure_code", f"main_class{n}")
    if not os.path.isdir(fdir):
        print(f"class {n}: no figure_code folder"); bad += 1; continue
    py = {os.path.splitext(os.path.basename(p))[0] for p in glob.glob(fdir + "/*.py")}
    stray = sorted(p for p in py - names - HELPERS)
    # loaded names that some src module writes, by string search
    src = " ".join(open(p, errors="ignore").read() for p in glob.glob(cdir + "/**/src/**/*.py", recursive=True))
    gen = {x for x in names if re.search(r"['\"]%s['\"]" % re.escape(os.path.splitext(x)[0]), src)}
    missing = sorted(x for x in gen if os.path.splitext(x)[0] not in py)
    hdr = [p for p in glob.glob(fdir + "/*.py") if os.path.splitext(os.path.basename(p))[0] in names
           and not re.search(r"Used in:.*\n(.|\n)*?Needs:", open(p).read()[:1500])]
    print(f"class {n:2d}: {len(py & names):3d} figure scripts, helpers={sorted(py & HELPERS)}"
          + (f"  STRAY={stray}" if stray else "") + (f"  MISSING={missing}" if missing else "")
          + (f"  NO-HEADER={len(hdr)}" if hdr else ""))
    bad += bool(stray or missing or hdr)
sys.exit(1 if bad else 0)

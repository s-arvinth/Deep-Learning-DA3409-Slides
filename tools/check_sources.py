#!/usr/bin/env python3
"""Lint the \\src / \\nosrc / \\srcdef tags of one or more decks against
tools/SOURCE_REQUIREMENTS.txt, and check that the only changes since the base
commit are added tag lines.

    python3 tools/check_sources.py Class05_*/A_*/main.tex Class05_*/B_*/main.tex
    python3 tools/check_sources.py all
Exit status 1 if any ERROR.  WARN lines are worth reading but do not fail."""
import glob, os, re, subprocess, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BASE = "03bf6e7"
FIXED = {"bishop2024", "prince2023", "rojas1996", "goodfellow2016", "jurafsky_slp3"}
TYPES = {"chapter", "section", "figure", "equation", "table", "algorithm", "page"}
PRE = [r"\newcommand{\src}[3]{}", r"\newcommand{\nosrc}{}", r"\newcommand{\srcdef}[2]{}"]
NUM = re.compile(r"^[0-9A-Za-z]+(\.[0-9A-Za-z]+)*(-[0-9A-Za-z]+(\.[0-9A-Za-z]+)*)?$")
KEY = re.compile(r"^[a-z]+[0-9]{4}[a-z]?$")
TAGLINE = re.compile(r"^\s*(\\src\{[^}]*\}\{[^}]*\}\{[^}]*\}\s*)+$|^\s*\\nosrc\s*$|^\s*\\srcdef\{.*$|^\s*$")

def strip_comments(s):
    return "\n".join(re.sub(r"(?<!\\)%.*", "", l) for l in s.split("\n"))

def srcdefs(s):
    out = {}; i = 0
    while True:
        j = s.find(r"\srcdef{", i)
        if j < 0: return out
        k = s.find("}", j); key = s[j + 8:k]
        depth = 0; m = k + 1
        if m < len(s) and s[m] == "{":
            for m in range(k + 1, len(s)):
                depth += s[m] == "{"; depth -= s[m] == "}"
                if depth == 0: break
        out.setdefault(key, []).append(s[k + 2:m]); i = m

def check(path):
    err, warn = [], []
    raw = open(path, errors="ignore").read()
    for n, l in enumerate(raw.split("\n"), 1):
        if re.search(r"(?<!\\)%.*\\(src|nosrc|srcdef)\b", l):
            err.append(f"line {n}: tag inside a % comment")
    s = strip_comments(raw)
    pre = s.split(r"\begin{document}")[0]
    for p in PRE:
        if p not in pre: err.append("preamble is missing: " + p)
    defs = srcdefs(pre)
    for k, v in defs.items():
        if len(v) > 1: err.append(f"\\srcdef{{{k}}} defined {len(v)} times")
        if not (KEY.match(k) or k == "jurafsky_slp3"): err.append(f"\\srcdef key {k!r} is not surname+year")
        if k in FIXED and k != "jurafsky_slp3": warn.append(f"\\srcdef for fixed key {k}")
    used = set()
    frames = list(re.finditer(r"\\begin\{frame\}(?:<[^>]*>)?(\[[^\]]*\])?(?:\{((?:[^{}]|\{[^{}]*\})*)\})?(.*?)\\end\{frame\}", s, re.S))
    nsrc = nnos = 0
    for f in frames:
        title = (f.group(2) or "(untitled)").strip()[:50]; body = f.group(3)
        tags = re.findall(r"\\src\{([^}]*)\}\{([^}]*)\}\{([^}]*)\}", body)
        nos = len(re.findall(r"\\nosrc\b", body))
        if not tags and not nos: err.append(f"frame '{title}': no \\src and no \\nosrc")
        if tags and nos: err.append(f"frame '{title}': has both \\src and \\nosrc")
        if re.search(r"\\src\{[^}]*\}\{[^}]*\}(?!\{)", body) or re.search(r"\\src(?!\{)", body.replace(r"\srcdef", "")):
            err.append(f"frame '{title}': malformed \\src (needs three arguments)")
        for k, t, num in tags:
            used.add(k); nsrc += 1
            if k not in FIXED and k not in defs: err.append(f"frame '{title}': key {k!r} has no \\srcdef")
            if t not in TYPES: err.append(f"frame '{title}': type {t!r} not allowed")
            if not NUM.match(num): err.append(f"frame '{title}': number {num!r} not in bare form")
        nnos += nos
        first = body.lstrip().split("\n")[0] if body.strip() else ""
        if not re.match(r"\\(src|nosrc)\b", first):
            warn.append(f"frame '{title}': tags are not the first lines of the frame body")
    if "jurafsky_slp3" in used and "jurafsky_slp3" not in defs:
        err.append("jurafsky_slp3 is used but has no \\srcdef giving the draft date")
    for k in defs:
        if k not in used: warn.append(f"\\srcdef{{{k}}} is defined but never used")
    # only tag lines may have been added since the base commit
    rel = os.path.relpath(path, ROOT)
    d = subprocess.run(["git", "-C", ROOT, "diff", "-U0", BASE, "--", rel], capture_output=True, text=True).stdout
    for l in d.split("\n"):
        if l.startswith("-") and not l.startswith("---"): err.append("removed or changed line: " + l[:90])
        if l.startswith("+") and not l.startswith("+++"):
            t = l[1:]
            if not (TAGLINE.match(t) or t.strip() in PRE or re.match(r"^\s*%", t) or re.match(r"^\s+\S.*\}\s*$", t)):
                err.append("added line that is not a tag: " + t[:90])
    return len(frames), nsrc, nnos, err, warn

def main(args):
    paths = sorted(glob.glob(ROOT + "/Class*/[AB]_*/main.tex")) if args == ["all"] else args
    bad = 0
    for p in paths:
        nf, ns, nn, err, warn = check(p)
        print(f"{os.path.relpath(p, ROOT)}: frames={nf} src={ns} nosrc={nn} errors={len(err)} warnings={len(warn)}")
        for e in err: print("   ERROR", e)
        for w in warn[:15]: print("   WARN ", w)
        if len(warn) > 15: print(f"   WARN  ... {len(warn) - 15} more")
        bad += bool(err)
    sys.exit(1 if bad else 0)

if __name__ == "__main__":
    main(sys.argv[1:])

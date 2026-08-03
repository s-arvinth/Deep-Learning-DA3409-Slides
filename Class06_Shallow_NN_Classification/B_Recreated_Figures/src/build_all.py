#!/usr/bin/env python3
"""Regenerate every figure in this deck.

    python3 build_all.py            # all figures
    python3 build_all.py buildup    # only figures whose name matches

Each figure also runs on its own:

    python3 figures/fig07_buildup.py
"""
import os, sys, glob, importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def modules():
    for path in sorted(glob.glob(os.path.join(HERE, "figures", "fig*.py"))):
        name = os.path.splitext(os.path.basename(path))[0]
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        yield name, mod


def main(patterns):
    n = 0
    for name, mod in modules():
        target = getattr(mod, "NAME", name)
        if patterns and not any(p in name or p in target for p in patterns):
            continue
        mod.build()
        n += 1
    print(f"\n{n} figure(s) written to ../figs/")


if __name__ == "__main__":
    main(sys.argv[1:])

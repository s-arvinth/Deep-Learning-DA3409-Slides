# Helper for the Class 26 figure scripts: ported from
# Class26_BPTT_Bidirectional_RNNs/B_Recreated_Figures/src/common/experiment.py.
# The stages, their seeds and their parameters are unchanged.  Only the
# cache changed: the original kept every stage in one ../figs/_cache.pkl;
# here each stage is cached on its own in output/_cache_<stage>.pkl next to
# the scripts, and a script runs only the stage(s) its figure needs, so it
# works on its own from a clean output/ (each stage seeds its own
# generators, so the result does not depend on which stages ran before).
# Needs: numpy
"""The shared experiments, cached stage by stage in output/_cache_<stage>.pkl:

check      BPTT by hand (Eqs. 10.17-10.28) against the tape and against
           finite differences on a small RNN
contrib    the per-step terms of the parameter gradients for a loss at
           the last step, at initialisation and after training
timing     wall-clock of forward and backward passes against tau
trunc      the recall task at length 12 trained with truncation k
influence  ||d o_t / d e_s|| maps for a unidirectional and a
           bidirectional RNN
"""
import os, pickle
import numpy as np
import models as M

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
T_TRUNC = 12
KS = (1, 2, 4, 6, 8, 10, 11, 12)
TAUS = (10, 20, 40, 80, 160, 320)
STAGES = ("check", "contrib", "timing", "trunc", "influence")


def _path(stage):
    return os.path.join(CACHE_DIR, "_cache_%s.pkl" % stage)


def _load(stage):
    p = _path(stage)
    if os.path.exists(p):
        with open(p, "rb") as f:
            return pickle.load(f)
    return None


def _save(stage, value):
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(_path(stage), "wb") as f:
        pickle.dump(value, f)


def run(force=False, stages=STAGES):
    """Return {stage: result} for the requested stages, computing (and
    caching) any stage whose cache file is absent."""
    out = {}
    for k in stages:
        v = None if force else _load(k)
        if v is None:
            print("stage", k)
            globals()["_stage_" + k](out)
            _save(k, out[k])
        else:
            out[k] = v
    return out


def _stage_check(out):
    rng = np.random.default_rng(0)
    m = M.SimpleRNN(M.NSYM, 8, M.K, seed=1)
    X = rng.integers(1, M.NSYM, size=(4, 6)); Y = rng.integers(0, M.K, size=(4, 6))
    g_tape = M.tape_gradients(m, X, Y)
    g_bptt, _, _ = M.bptt(m, X, Y)
    fd = M.finite_difference(m, X, Y, max_entries=60, rng=rng)
    pairs = {}
    for name, gb, gt, vals in zip(m.names, g_bptt, g_tape, fd):
        pairs[name] = dict(fd=np.array([v for _, v in vals]), bptt=np.array([gb[ij] for ij, _ in vals]),
                           tape_vs_bptt=float(np.abs(gt - gb).max()))
    out["check"] = dict(pairs=pairs, n_entries=sum(len(v) for v in fd), shape=dict(B=4, T=6, d=8))
    for name, p in pairs.items():
        print("  %s: |tape - bptt| %.1e, |fd - bptt| %.1e" % (name, p["tape_vs_bptt"], np.abs(p["fd"] - p["bptt"]).max()))


def _stage_contrib(out):
    T = 16
    m = M.SimpleRNN(M.NSYM, 16, M.K, seed=0)
    rng = np.random.default_rng(3)
    X, y = M.marker_batch(rng, T, 32)
    _, per0, dH0 = M.bptt(m, X, y, every_step=False)
    M.train(m, 1500, np.random.default_rng(1), T=T)
    acc = M.accuracy(m, np.random.default_rng(2), T)
    _, per1, dH1 = M.bptt(m, X, y, every_step=False)
    out["contrib"] = dict(T=T, init=per0, trained=per1, acc=acc,
                          dH_init=[float(np.linalg.norm(g, axis=1).mean()) for g in dH0],
                          dH_trained=[float(np.linalg.norm(g, axis=1).mean()) for g in dH1])
    print("  trained accuracy %.2f" % acc)


def _stage_timing(out):
    out["timing"] = M.timing(taus=TAUS, repeats=5)
    for r in out["timing"]:
        print("  tau %4d  forward %.4fs  backward %.4fs" % (r["T"], r["forward"], r["backward"]))


def _stage_trunc(out):
    res = []
    for k in KS:
        m = M.SimpleRNN(M.NSYM, 16, M.K, seed=0, tie_markers=True)
        g0 = M.marker_gradient(m, np.random.default_rng(3), T_TRUNC, k)
        hist = M.train(m, 800, np.random.default_rng(1), T=T_TRUNC, truncate=k)
        acc = M.accuracy(m, np.random.default_rng(2), T_TRUNC)
        res.append(dict(k=k, grad=g0, acc=acc, hist=hist))
        print("  k=%2d  grad to marker %.1e  accuracy %.2f" % (k, g0, acc))
    out["trunc"] = res


def _stage_influence(out):
    rng = np.random.default_rng(0)
    T = 8
    X = rng.integers(1, M.NSYM, size=(4, T))
    uni = M.SimpleRNN(M.NSYM, 12, M.K, seed=0)
    bi = M.Bidirectional(M.NSYM, 12, M.K, seed=0)
    out["influence"] = dict(uni=M.influence_map(uni, X, lambda ids: uni.E.v[ids]),
                            bi=M.influence_map(bi, X, lambda ids: bi.fwd.E.v[ids]), T=T)

"""The shared experiments of Class 29 that the slides use.

Ported from Class29_RNN_NLP_Tasks/B_Recreated_Figures/src/common/
experiment.py, same seeds and settings, with only the two stages a slide
uses (the original also holds untied, depth and classifier stages that no
slide shows; they are independent of these two -- each stage seeds its own
generators -- so dropping them changes nothing):

lm       the character-level LSTM language model (d = 64, tied) on
         data/magi.txt, 700 steps, its training and validation perplexity
         by step, and samples at three temperatures  -> char_lm_curve,
                                                        char_lm_samples
tagger   a left-to-right and a bidirectional RNN tagger on the grammar
                                                     -> pos_tagging

Cached stage by stage in output/_cache_<stage>.pkl; the LM's training also
writes output/_ckpt_lm.pkl after every evaluation (every 50 steps), so an
interrupted run resumes exactly where it stopped.  Both stages together
take well under a minute from scratch (the original's ~4 minutes was
mostly the three unused stages).  Delete output/ to recompute -- the seeds are
fixed, so the result is the same.

Needs: numpy
"""
import os, pickle
import numpy as np
import models as M
from common import OUTPUT

LM_STEPS = 700
LM_T, LM_B = 48, 16
TEMPS = (0.5, 1.0, 1.5)


def _path(kind, name):
    return os.path.join(OUTPUT, "_%s_%s.pkl" % (kind, name))


def _split():
    s = M.text(); vocab = M.Vocab(s); ids = vocab.encode(s)
    n = int(0.85 * len(ids))
    return s, vocab, ids[:n], ids[n:]


def _stage_lm(out, vocab, tr, va):
    print("training the tied LSTM LM")
    lm = M.CharLM(vocab.size, d=64, layers=1, tied=True, seed=0)
    hist = M.train_lm(lm, tr, va, LM_STEPS, np.random.default_rng(1), T=LM_T, B=LM_B,
                      ckpt=_path("ckpt", "lm"))
    samples = {t: lm.sample(vocab, "Della ", 160, temperature=t, rng=np.random.default_rng(3)) for t in TEMPS}
    out["lm"] = dict(model=lm, hist=hist, samples=samples, n_params=lm.n_params(),
                     val_ppl=M.perplexity(lm, va), train_ppl=M.perplexity(lm, tr))
    for t, sm in samples.items():
        print("  T=%.1f: %s" % (t, sm))


def _stage_tagger(out, vocab, tr, va):
    out["tagger"] = {}
    for bi in (False, True):
        name = "bidirectional" if bi else "left-to-right"
        print("training the %s tagger" % name)
        tg = M.Tagger(bidirectional=bi, seed=0)
        M.train_tagger(tg, 3000, np.random.default_rng(2))
        acc, acc_r = M.tagger_accuracy(tg, np.random.default_rng(9))
        print("  token accuracy %.3f; on right-context tokens %.3f" % (acc, acc_r))
        out["tagger"][name] = dict(model=tg, acc=acc, acc_right=acc_r)


def run(stages=("lm", "tagger")):
    """The requested stages, each from its own cache (computed if absent)."""
    s, vocab, tr, va = _split()
    out = dict(text=dict(chars=len(s), vocab=vocab.size, n_train=len(tr), n_val=len(va)))
    for k in stages:
        cache = _path("cache", k)
        if os.path.exists(cache):
            with open(cache, "rb") as f:
                out[k] = pickle.load(f)
            continue
        print("text: %d characters, %d distinct, %d train / %d validation" % (len(s), vocab.size, len(tr), len(va)))
        globals()["_stage_" + k](out, vocab, tr, va)
        os.makedirs(OUTPUT, exist_ok=True)
        with open(cache + ".tmp", "wb") as f:
            pickle.dump(out[k], f)
        os.replace(cache + ".tmp", cache)
        if os.path.exists(_path("ckpt", k)):
            os.remove(_path("ckpt", k))
    return out

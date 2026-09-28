"""The shared experiments, cached in ../figs/_cache.pkl:

lm       the character-level LSTM language model (d = 64, tied) on the
         story, with its training and validation perplexity by step and
         samples at three temperatures; the untied twin; and one-, two-
         and three-layer models at matched parameter count
tagger   a left-to-right and a bidirectional RNN tagger on the grammar
cls      three classifiers (last, mean, max read-out) on the signal task
"""
import os, pickle
import numpy as np
from . import models as M

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "figs", "_cache.pkl")
LM_STEPS = 700
LM_T, LM_B = 48, 16
CLS_LENS = [5, 10, 20, 40, 80]
CLS_TRAIN_LENS = (40, 80)
CLS_STEPS = 1500
TEMPS = (0.5, 1.0, 1.5)
DEPTHS = ((1, 64), (2, 47), (3, 39))        # layers, width: about 41k parameters each


def _split():
    s = M.text(); vocab = M.Vocab(s); ids = vocab.encode(s)
    n = int(0.85 * len(ids))
    return s, vocab, ids[:n], ids[n:]


def _load():
    if os.path.exists(CACHE):
        with open(CACHE, "rb") as f:
            return pickle.load(f)
    return {}


def _save(out):
    with open(CACHE, "wb") as f:
        pickle.dump(out, f)


def run(force=False, stages=("lm", "lm_untied", "depth", "tagger", "cls")):
    """Each stage is cached on its own, so a run can be resumed."""
    out = {} if force else _load()
    s, vocab, tr, va = _split()
    out["text"] = dict(chars=len(s), vocab=vocab.size, n_train=len(tr), n_val=len(va))
    todo = [k for k in stages if k not in out]
    if not todo:
        return out
    print("text: %d characters, %d distinct, %d train / %d validation" % (len(s), vocab.size, len(tr), len(va)))
    for k in todo:
        globals()["_stage_" + k](out, vocab, tr, va)
        _save(out)
    return out


def _stage_lm(out, vocab, tr, va):
    print("training the tied LSTM LM")
    lm = M.CharLM(vocab.size, d=64, layers=1, tied=True, seed=0)
    hist = M.train_lm(lm, tr, va, LM_STEPS, np.random.default_rng(1), T=LM_T, B=LM_B)
    samples = {t: lm.sample(vocab, "Della ", 160, temperature=t, rng=np.random.default_rng(3)) for t in TEMPS}
    out["lm"] = dict(model=lm, hist=hist, samples=samples, n_params=lm.n_params(),
                     val_ppl=M.perplexity(lm, va), train_ppl=M.perplexity(lm, tr))
    for t, sm in samples.items():
        print("  T=%.1f: %s" % (t, sm))


def _stage_lm_untied(out, vocab, tr, va):
    lm = out["lm"]["model"]
    print("training the untied LSTM LM")
    lmu = M.CharLM(vocab.size, d=64, layers=1, tied=False, seed=0)
    histu = M.train_lm(lmu, tr, va, LM_STEPS, np.random.default_rng(1), T=LM_T, B=LM_B, log=False)
    out["lm_untied"] = dict(hist=histu, n_params=lmu.n_params(), val_ppl=M.perplexity(lmu, va))
    print("  tied %d params, val ppl %.2f; untied %d params, val ppl %.2f" %
          (lm.n_params(), out["lm"]["val_ppl"], lmu.n_params(), out["lm_untied"]["val_ppl"]))


def _stage_depth(out, vocab, tr, va):
    out["depth"] = []
    for L, d in DEPTHS:
        print("training a %d-layer LM of width %d" % (L, d))
        m = M.CharLM(vocab.size, d=d, layers=L, tied=True, seed=0)
        h = M.train_lm(m, tr, va, LM_STEPS, np.random.default_rng(1), T=LM_T, B=LM_B, log=False)
        out["depth"].append(dict(layers=L, width=d, n_params=m.n_params(), hist=h, val_ppl=M.perplexity(m, va)))
        print("  %d params, val ppl %.2f" % (m.n_params(), out["depth"][-1]["val_ppl"]))



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



def _stage_cls(out, vocab, tr, va):
    """Three read-outs, each trained at a fixed sequence length; the
    gradient reaching the first hidden state is read off the tape at
    initialisation, before any training."""
    out["cls"] = {}
    for ro in ("last", "mean", "max"):
        print("training the %s-readout classifier" % ro)
        c0 = M.Classifier(readout=ro, seed=0)
        grad0 = M.gradient_to_first_state(c0, np.random.default_rng(6), CLS_LENS)
        rec = dict(grad_init=grad0, hist={})
        for T in CLS_TRAIN_LENS:
            c = M.Classifier(readout=ro, seed=0)
            rec["hist"][T] = M.train_classifier(c, CLS_STEPS, np.random.default_rng(4), lens=(T, T))
            rec.setdefault("acc", {})[T] = M.classifier_accuracy(c, np.random.default_rng(5), [T])[T]
            print("  length %d: final loss %.3f, accuracy %.2f" % (T, np.mean(rec["hist"][T][-50:]), rec["acc"][T]))
        out["cls"][ro] = rec

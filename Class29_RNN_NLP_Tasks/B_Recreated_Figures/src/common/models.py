"""The objects Class 29 computes with.

The cells of Class 28 (a simple RNN and an LSTM on the tape of Class
27), wired into the four NLP architectures of Jurafsky & Martin (2026)
Section 14.6:

`CharLM`      an RNN language model (Eqs. 14.4-14.6), character level,
              trained by self-supervision with the cross-entropy of
              Eq. 14.11; tied (Eqs. 14.12-14.14) or untied output;
              one or several stacked layers (Section 14.4.1)
`Tagger`      a sequence labeller (Section 14.3.1), left-to-right or
              bidirectional (Eqs. 14.16-14.18)
`Classifier`  a sequence classifier (Section 14.3.2) reading the last
              state, the mean of the states (Eq. 14.15) or their max

and the data they are trained on:

`text`        O. Henry, "The Gift of the Magi" (1905), public domain,
              11,500 characters, the last 15 percent held out
`grammar`     a small lexicon with a few ambiguous words and a handful
              of sentence templates, so that some tags can only be read
              from the right context
`signal_batch` sequences whose label is decided by one symbol placed
              near the start, so that the three read-outs differ
"""
import os
import numpy as np
from . import tape as tp

HERE = os.path.dirname(os.path.abspath(__file__))


def _orth(rng, d):
    q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    return q


def maximum(xs):
    """Elementwise max of a list of same-shape tensors, with gradient."""
    V = np.stack([x.v for x in xs], axis=0)
    idx = np.argmax(V, axis=0)
    out = np.max(V, axis=0)

    def bw(g):
        return tuple(g * (idx == k) for k in range(len(xs)))
    return tp.T(out, tuple(xs), bw)


# ---------------------------------------------------------------- cells
class RNNCell:
    kind = "RNN"

    def __init__(self, d_in, d, rng):
        s = 1.0 / np.sqrt(d)
        self.W = tp.T(rng.normal(size=(d_in, d)) * s); self.U = tp.T(_orth(rng, d))
        self.b = tp.T(np.zeros(d)); self.d = d
        self.params = [self.W, self.U, self.b]

    def n_params(self):
        return sum(p.v.size for p in self.params)

    def init_state(self, B):
        return tp.T(np.zeros((B, self.d)))

    def step(self, x, h):
        return tp.tanh(x @ self.W + h @ self.U + self.b)

    @staticmethod
    def hidden(state):
        return state


class LSTMCell:
    kind = "LSTM"

    def __init__(self, d_in, d, rng, forget_bias=1.0):
        s = 1.0 / np.sqrt(d)
        self.Wf = tp.T(rng.normal(size=(d_in, d)) * s); self.Uf = tp.T(_orth(rng, d)); self.bf = tp.T(np.full(d, forget_bias))
        self.Wi = tp.T(rng.normal(size=(d_in, d)) * s); self.Ui = tp.T(_orth(rng, d)); self.bi = tp.T(np.zeros(d))
        self.Wg = tp.T(rng.normal(size=(d_in, d)) * s); self.Ug = tp.T(_orth(rng, d)); self.bg = tp.T(np.zeros(d))
        self.Wo = tp.T(rng.normal(size=(d_in, d)) * s); self.Uo = tp.T(_orth(rng, d)); self.bo = tp.T(np.zeros(d))
        self.d = d
        self.params = [self.Wf, self.Uf, self.bf, self.Wi, self.Ui, self.bi,
                       self.Wg, self.Ug, self.bg, self.Wo, self.Uo, self.bo]

    def n_params(self):
        return sum(p.v.size for p in self.params)

    def init_state(self, B):
        return (tp.T(np.zeros((B, self.d))), tp.T(np.zeros((B, self.d))))

    def step(self, x, state):
        h, c = state
        f = tp.sigmoid(h @ self.Uf + x @ self.Wf + self.bf)
        i = tp.sigmoid(h @ self.Ui + x @ self.Wi + self.bi)
        g = tp.tanh(h @ self.Ug + x @ self.Wg + self.bg)
        o = tp.sigmoid(h @ self.Uo + x @ self.Wo + self.bo)
        c2 = c * f + g * i
        return (o * tp.tanh(c2), c2)

    @staticmethod
    def hidden(state):
        return state[0]


def run_layers(cells, xs, reverse=False):
    """Run stacked cells over a list of input tensors (one per step);
    returns the list of top-layer hidden states, in input order."""
    order = range(len(xs) - 1, -1, -1) if reverse else range(len(xs))
    seq = xs
    for cell in cells:
        B = xs[0].v.shape[0]
        state = cell.init_state(B)
        out = [None] * len(xs)
        for t in order:
            state = cell.step(seq[t], state)
            out[t] = cell.hidden(state)
        seq = out
    return seq


# ------------------------------------------------------ the text corpus
def text():
    with open(os.path.join(HERE, "..", "data", "magi.txt")) as f:
        s = f.read()
    s = " ".join(s.split())            # one space between tokens, no newlines
    return s


class Vocab:
    def __init__(self, s):
        self.chars = sorted(set(s)); self.idx = {c: i for i, c in enumerate(self.chars)}
        self.size = len(self.chars)

    def encode(self, s):
        return np.array([self.idx[c] for c in s])

    def decode(self, ids):
        return "".join(self.chars[i] for i in ids)


# ------------------------------------------------------ the language model
class CharLM:
    """Jurafsky Eqs. 14.4-14.6 at character level:
        e_t = E x_t,  h_t = cell(h_{t-1}, e_t),  y_t = softmax(V h_t)
    tied: V = E^T (Eqs. 14.12-14.14)."""

    def __init__(self, V, d=64, layers=1, tied=True, cell_cls=LSTMCell, seed=0):
        rng = np.random.default_rng(seed)
        self.V = V; self.d = d; self.tied = tied
        self.E = tp.T(rng.normal(size=(V, d)) * 0.3)
        self.cells = [cell_cls(d, d, rng) for _ in range(layers)]
        self.bo = tp.T(np.zeros(V))
        self.params = [self.E, self.bo] + [p for c in self.cells for p in c.params]
        if not tied:
            self.Vo = tp.T(rng.normal(size=(d, V)) / np.sqrt(d))
            self.params.append(self.Vo)

    def n_params(self):
        return sum(p.v.size for p in self.params)

    def logits(self, X):
        """X: (B, T) integer array -> list of T logit tensors (B, V)."""
        xs = [self.E[X[:, t]] for t in range(X.shape[1])]
        hs = run_layers(self.cells, xs)
        Vo = self.E.T_() if self.tied else self.Vo
        return [h @ Vo + self.bo for h in hs]

    def loss(self, X, Y):
        """Mean of Eq. 14.11 over every position: -log y_t[w_{t+1}]."""
        ls = self.logits(X)
        Z = tp.concat(ls, axis=0)                     # (T*B, V), step-major
        target = Y.T.reshape(-1)
        return tp.cross_entropy(Z, target)

    def sample(self, vocab, prefix, n, temperature=1.0, rng=None):
        rng = rng or np.random.default_rng(0)
        ids = list(vocab.encode(prefix))
        states = [c.init_state(1) for c in self.cells]
        out = []
        for t in range(len(ids) + n - 1):
            x = self.E[np.array([ids[t]])]
            for k, c in enumerate(self.cells):
                states[k] = c.step(x, states[k]); x = c.hidden(states[k])
            if t >= len(ids) - 1:
                Vo = self.E.T_() if self.tied else self.Vo
                z = (x @ Vo + self.bo).v[0] / temperature
                p = np.exp(z - z.max()); p /= p.sum()
                nxt = int(rng.choice(self.V, p=p)); ids.append(nxt); out.append(nxt)
        return prefix + vocab.decode(out)


def lm_batches(ids, T, B, rng):
    starts = rng.integers(0, len(ids) - T - 1, size=B)
    X = np.stack([ids[s:s + T] for s in starts]); Y = np.stack([ids[s + 1:s + T + 1] for s in starts])
    return X, Y


def perplexity(model, ids, T=64, B=32, rng=None):
    """exp of the mean cross-entropy over the whole of `ids`, in
    non-overlapping windows (Jurafsky Eq. 3.14 with Eq. 7.58)."""
    n = (len(ids) - 1) // T
    X = np.stack([ids[k * T:(k + 1) * T] for k in range(n)])
    Y = np.stack([ids[k * T + 1:(k + 1) * T + 1] for k in range(n)])
    losses = []; weights = []
    for k in range(0, n, B):
        L = model.loss(X[k:k + B], Y[k:k + B])
        losses.append(float(L.v)); weights.append(len(X[k:k + B]))
    return float(np.exp(np.average(losses, weights=weights)))


def train_lm(model, train_ids, val_ids, steps, rng, T=48, B=16, lr=3e-3, eval_every=50, log=True):
    opt = tp.Adam(model.params, lr=lr, clip=5.0)
    hist = []                          # (step, train ppl over last window, val ppl)
    window = []
    for s in range(1, steps + 1):
        X, Y = lm_batches(train_ids, T, B, rng)
        L = model.loss(X, Y)
        opt.zero(); L.backward(); opt.step(); window.append(float(L.v))
        if s % eval_every == 0:
            tr = float(np.exp(np.mean(window))); window = []
            va = perplexity(model, val_ids)
            hist.append((s, tr, va))
            if log:
                print("  step %5d  train ppl %6.2f  val ppl %6.2f" % (s, tr, va))
    return hist


# ------------------------------------------------------ tagging
TAGS = ["NNP", "MD", "VB", "DT", "NN", "CC", "VBD", "JJ", "RB"]
LEX = {
    "Janet": "NNP", "Mary": "NNP", "Tom": "NNP",
    "will": "MD", "can": "MD", "must": "MD",
    "the": "DT", "a": "DT", "this": "DT",
    "and": "CC", "was": "VBD", "were": "VBD", "hurt": "VBD", "ached": "VBD",
    "sore": "JJ", "long": "JJ", "late": "JJ", "again": "RB", "now": "RB",
    "bill": "NN", "race": "NN", "car": "NN", "boat": "NN", "neck": "NN", "door": "NN", "plan": "NN",
}
# words that are a verb (VB) or a noun (NN) depending on context
AMBIG = ["back", "run", "walk", "book", "park", "book"]
AMBIG = sorted(set(AMBIG))
NOUNS = ["bill", "race", "car", "boat", "neck", "door", "plan"]
NAMES = ["Janet", "Mary", "Tom"]
MODALS = ["will", "can", "must"]
DETS = ["the", "a", "this"]

TEMPLATES = [
    # right context decides: sentence-initial ambiguous word
    ("A D N", "VB DT NN"),                 # "Back the bill"          imperative
    ("A D N R", "VB DT NN RB"),            # "Run the race again"
    ("A C N V", "NN CC NN VBD"),           # "Back and neck ached"    subject noun
    ("A C N V R", "NN CC NN VBD RB"),
    # left context decides
    ("P M A D N", "NNP MD VB DT NN"),      # "Janet will back the bill"
    ("D A V J", "DT NN VBD JJ"),           # "The back was sore"
    ("D N M A D N", "DT NN MD VB DT NN"),  # "The car will park the boat" (nonsense, but tagged)
    ("P V J", "NNP VBD JJ"),
]
POOL = {"A": AMBIG, "D": DETS, "N": NOUNS, "P": NAMES, "M": MODALS,
        "C": ["and"], "V": ["was", "hurt", "ached", "were"], "J": ["sore", "long", "late"], "R": ["again", "now"]}
WORDS = sorted(set(LEX) | set(AMBIG))
WIDX = {w: i for i, w in enumerate(WORDS)}
TIDX = {t: i for i, t in enumerate(TAGS)}


def grammar(rng, n):
    """n (words, tags) sentences, with a flag per token: True where the
    tag can only be read from the right context."""
    out = []
    for _ in range(n):
        pat, tags = TEMPLATES[rng.integers(len(TEMPLATES))]
        words = [POOL[c][rng.integers(len(POOL[c]))] for c in pat.split()]
        flag = [c == "A" and k == 0 for k, c in enumerate(pat.split())]
        out.append((words, tags.split(), flag))
    return out


class Tagger:
    """Jurafsky Fig. 14.7: embeddings -> RNN -> softmax over tags at
    every step.  bidirectional: two RNNs, states concatenated (Eq. 14.18)."""

    def __init__(self, d=24, bidirectional=False, seed=0):
        rng = np.random.default_rng(seed)
        self.bi = bidirectional
        self.E = tp.T(rng.normal(size=(len(WORDS), d)) * 0.3)
        self.fwd = RNNCell(d, d, rng)
        self.params = [self.E] + self.fwd.params
        dout = d
        if bidirectional:
            self.bwd = RNNCell(d, d, rng); self.params += self.bwd.params; dout = 2 * d
        self.Vo = tp.T(rng.normal(size=(dout, len(TAGS))) / np.sqrt(dout)); self.bo = tp.T(np.zeros(len(TAGS)))
        self.params += [self.Vo, self.bo]

    def logits(self, words):
        xs = [self.E[np.array([WIDX[w]])] for w in words]
        hf = run_layers([self.fwd], xs)
        if self.bi:
            hb = run_layers([self.bwd], xs, reverse=True)
            hs = [tp.concat([f, b], axis=1) for f, b in zip(hf, hb)]
        else:
            hs = hf
        return [h @ self.Vo + self.bo for h in hs]

    def loss(self, words, tags):
        Z = tp.concat(self.logits(words), axis=0)
        return tp.cross_entropy(Z, np.array([TIDX[t] for t in tags]))

    def predict(self, words):
        ls = self.logits(words)
        P = [np.exp(z.v[0] - z.v[0].max()) for z in ls]
        return [p / p.sum() for p in P]


def train_tagger(model, steps, rng, lr=5e-3):
    opt = tp.Adam(model.params, lr=lr, clip=5.0)
    data = grammar(rng, 400)
    for s in range(steps):
        words, tags, _ = data[rng.integers(len(data))]
        L = model.loss(words, tags)
        opt.zero(); L.backward(); opt.step()


def tagger_accuracy(model, rng, n=300):
    """Overall token accuracy, and accuracy on the tokens whose tag
    needs the right context."""
    data = grammar(rng, n)
    tot = hit = tot_r = hit_r = 0
    for words, tags, flag in data:
        P = model.predict(words)
        for p, t, f in zip(P, tags, flag):
            ok = TAGS[int(np.argmax(p))] == t
            tot += 1; hit += ok
            if f:
                tot_r += 1; hit_r += ok
    return hit / tot, hit_r / max(tot_r, 1)


# ------------------------------------------------------ classification
NSYM = 40         # 1 = positive marker, 2 = negative marker, 3.. filler


def signal_batch(rng, T, B):
    """Filler symbols with one marker at position 0, 1 or 2 deciding the
    label: the evidence sits at the START of the sequence."""
    X = rng.integers(3, NSYM, size=(B, T))
    y = rng.integers(0, 2, size=B)
    pos = rng.integers(0, 3, size=B)
    X[np.arange(B), pos] = np.where(y == 1, 1, 2)
    return X, y


class Classifier:
    """Jurafsky Fig. 14.8: RNN, then a feedforward classifier on a
    read-out of the hidden states: 'last' h_n, 'mean' (Eq. 14.15) or
    'max' (elementwise)."""

    def __init__(self, readout="last", d=16, seed=0):
        rng = np.random.default_rng(seed)
        self.readout = readout
        self.E = tp.T(rng.normal(size=(NSYM, d)) * 0.3)
        self.cell = RNNCell(d, d, rng)
        self.W1 = tp.T(rng.normal(size=(d, d)) / np.sqrt(d)); self.b1 = tp.T(np.zeros(d))
        self.W2 = tp.T(rng.normal(size=(d, 2)) / np.sqrt(d)); self.b2 = tp.T(np.zeros(2))
        self.params = [self.E, self.W1, self.b1, self.W2, self.b2] + self.cell.params

    def forward(self, X):
        xs = [self.E[X[:, t]] for t in range(X.shape[1])]
        hs = run_layers([self.cell], xs)
        if self.readout == "last":
            r = hs[-1]
        elif self.readout == "mean":
            r = tp.stack(hs, axis=0).mean(axis=0)
        else:
            r = maximum(hs)
        z = tp.tanh(r @ self.W1 + self.b1) @ self.W2 + self.b2
        return z, hs, xs

    def loss(self, X, y):
        z, hs, xs = self.forward(X)
        return tp.cross_entropy(z, y), hs, xs


def train_classifier(model, steps, rng, lens=(5, 20), B=32, lr=3e-3):
    opt = tp.Adam(model.params, lr=lr, clip=5.0)
    hist = []
    for s in range(steps):
        T = int(rng.integers(lens[0], lens[1] + 1))
        X, y = signal_batch(rng, T, B)
        L, _, _ = model.loss(X, y)
        opt.zero(); L.backward(); opt.step(); hist.append(float(L.v))
    return hist


def classifier_accuracy(model, rng, lengths, B=300):
    out = {}
    for T in lengths:
        X, y = signal_batch(rng, T, B)
        z, _, _ = model.forward(X)
        out[T] = float(np.mean(np.argmax(z.v, axis=1) == y))
    return out


def gradient_to_first_state(model, rng, lengths, B=32):
    """||dL/dh_1|| (mean over the batch) against the sequence length:
    how much of the loss reaches the first hidden state."""
    out = {}
    for T in lengths:
        X, y = signal_batch(rng, T, B)
        L, hs, _ = model.loss(X, y)
        L.backward()
        out[T] = float(np.linalg.norm(hs[0].g, axis=1).mean())
    return out

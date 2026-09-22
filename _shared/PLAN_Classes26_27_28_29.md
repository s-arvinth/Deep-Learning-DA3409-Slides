# Classes 26, 27, 28 and 29 — the RNN block, four lectures

The course plan assigns these four classes the same two sources:

| Class | Date | Topic (course plan) | Sources |
|---|---|---|---|
| **26** | Fri 11 Sep | Recurrent Neural Network — **Back-prop over time and Bidirectional RNNs** | Jurafsky 13, Goodfellow Ch. 10 |
| **27** | Tue 15 Sep | Recurrent Neural Network — **Encoder-decoder Seq-to-Seq** | Jurafsky 13, Goodfellow Ch. 10 |
| **28** | Wed 16 Sep | Recurrent Neural Network — **LSTM and Gated RNNs** | Jurafsky 13, Goodfellow Ch. 10 |
| **29** | Fri 18 Sep | Recurrent Neural Network — **RNN for NLP tasks** | Jurafsky 13, Goodfellow Ch. 10 |
| 30–31 | Mon 21, Tue 22 Sep | Transformers — self-attention, multi-head; positional encoding, layers | Prince Ch. 12, Bishop Ch. 12 |

Class 26 was planned last (it was first thought not to be ours) but is built
to be exactly what Classes 27–29 assume: the simple recurrent network,
unrolling, backpropagation through time, bidirectional RNNs. The three later
decks were written against the "deferred to Class 26" table at the end of
this document, so adding 26 changes nothing in them.
Classes 30–31 are the transformer, so **self-attention is not ours**: the only
attention in this block is the RNN encoder–decoder attention of Bahdanau et
al. (2015), which is the fix to the bottleneck and nothing more.

## The sources, and one thing to know about them

**Jurafsky & Martin, *Speech and Language Processing*, 3rd ed. draft.** The
course plan says "Chapter 13". In the draft we have (August 2026), the RNN
chapter has moved: **Chapter 14 is *RNNs and LSTMs*** (pp. 302–327), and
Chapter 13 is now *Machine Translation* (pp. 276–301). The plan's "13" is the
RNN chapter of the older draft. Both chapters are used here — 14 for the
architectures, 13 §13.2–13.4 for the encoder–decoder as a translation model
and for beam search — and every citation on a slide gives the section number
of the August 2026 draft, so nothing depends on which numbering the reader has.

**Goodfellow, Bengio & Courville, *Deep Learning* (2016), Chapter 10,
*Sequence Modeling: Recurrent and Recursive Nets*** (pp. 373–420). This is the
mathematical source: the unfolded computational graph, the gradient, the RNN
as a graphical model, the eigenvalue argument for long-term dependencies, the
LSTM and GRU equations, gradient clipping.

**No figure source exists for either book.** Jurafsky & Martin publish no
code; Goodfellow's `book-figures` repository is empty. So the A variant will
carry the published figures as *page crops from the book PDFs*, cited by
figure number as before, and every B figure is built from the deck's own code
and experiments, as in Classes 13–25. The authors' own lecture decks
(`10_rnn.pdf`; `rnnjan25.pdf`) are consulted for narrative order only.

**Notation.** The two books swap letters: Jurafsky writes $\mathbf{W}$ for
input-to-hidden and $\mathbf{U}$ for hidden-to-hidden; Goodfellow writes
$\mathbf{U}$ for input-to-hidden and $\mathbf{W}$ for hidden-to-hidden, and
puts time in a superscript, $\mathbf{h}^{(t)}$. The decks follow
**Jurafsky's letters with time as a subscript**:

$$\mathbf{h}_t = g(\mathbf{U}\mathbf{h}_{t-1} + \mathbf{W}\mathbf{x}_t),
\qquad \hat{\mathbf{y}}_t = \mathrm{softmax}(\mathbf{V}\mathbf{h}_t),$$

with the frozen course notation otherwise ($\mathbf{x}$ inputs, $\hat{\mathbf{y}}$
predictions, $\mathbf{w}$ parameters, bold vectors and matrices). Goodfellow's
swap is noted once, on the notation slide of each deck. Superscripts $e$ and
$d$ distinguish encoder and decoder states, as Jurafsky does.

---

## The organising question

| Class | The question | The object |
|---|---|---|
| **26** | *What is a recurrent network, and how is its gradient computed?* | the **unfolded graph**: one function $f$, applied $\tau$ times, and back-propagation over it |
| **27** | *How does a network read one sequence and write another of a different length?* | the **conditional language model** $p(\mathbf{y} \mid \mathbf{x})$ |
| **28** | *Why does a simple RNN forget, and what does a gate change?* | the **cell**: what happens between $\mathbf{h}_{t-1}$ and $\mathbf{h}_t$ |
| **29** | *What do you build with a recurrent layer, and how is each thing trained?* | the **output structure**: a label per token, one per sequence, or the next token |

The order the course plan fixes is awkward in one place, exactly as it was
for Classes 23–25: the encoder–decoder (27) is built on the RNN language
model, which Jurafsky treats in §14.2 and which belongs by title to "NLP
tasks" (29). The resolution is the same as before. **Class 27 states the RNN
language model in three equations as its opening, treats the recurrent cell
$g$ as a black box, and builds the encoder–decoder on top.** Class 28 opens
the black box. Class 29 then treats the language model as one of four output
structures, going into what 27 did not — weight tying, perplexity, sampling,
pooling, stacking — and closes the block with Jurafsky's Fig. 14.15 summary.

---

## Section-by-section assignment

### Class 26 — Back-prop over time and Bidirectional RNNs

Jurafsky & Martin:

- **§14.1 *Recurrent Neural Networks* in full** — the Elman network, Fig. 14.1;
  the recurrence as a feedforward step, Fig. 14.2; **§14.1.1 inference**, Eqs.
  14.1–14.3, the forward algorithm Fig. 14.3; **§14.1.2 training**, the
  unrolled network Fig. 14.4, the two-pass algorithm named *backpropagation
  through time*, and unrolling long inputs into fixed-length segments
- §14.4.2 *Bidirectional RNNs* — the **definition** only, Eqs. 14.16–14.18 and
  Fig. 14.11 (its use for labelling and classification is Class 29's)

Goodfellow, Bengio & Courville:

- **§10.1 *Unfolding Computational Graphs* in full** — Eqs. 10.1–10.7, Figs.
  10.1–10.2: the dynamical system, unfolding, $g^{(t)}$ factorised into
  repeated $f$, the two advantages of parameter sharing
- **§10.2 opening** — Eqs. 10.8–10.14, Fig. 10.3; the three design patterns,
  Figs. 10.3–10.5; the $O(\tau)$ runtime and memory of the unrolled graph;
  the Turing-completeness remark (Siegelmann & Sontag) in one line
- **§10.2.2 *Computing the Gradient in a Recurrent Neural Network* in full**
  — Eqs. 10.17–10.28: the recursion for $\nabla_{h^{(t)}} L$ and the
  parameter gradients as sums over time
- **§10.3 *Bidirectional RNNs*** — Fig. 10.11, the forward state $h$ and the
  backward state $g$, the 2-D extension in one line
- §10.2.1 is Class 27's (teacher forcing), §10.7 is Class 28's (the product of
  Jacobians as a problem): here the product appears once, as a corollary, and
  is left for 28

Publications supplying results neither book states formally: Elman (1990);
Werbos (1990) and Rumelhart, Hinton & Williams (1986) for BPTT; Williams &
Peng (1990) for truncated BPTT; Schuster & Paliwal (1997); Siegelmann &
Sontag (1995).

Results the deck states, and which are proved:

1. **Unfolding** — the recurrence $\mathbf{h}_t = f(\mathbf{h}_{t-1}, \mathbf{x}_t; \theta)$
   unrolled $t$ times is a function $g^{(t)}(\mathbf{x}_1, \ldots, \mathbf{x}_t)$
   of the whole past, computed by one $f$ with one $\theta$; the parameter count
   does not depend on $\tau$ (Goodfellow Eqs. 10.4–10.7).
2. **Back-propagation through time** — for the network of Eqs. 10.8–10.11 with
   summed losses, $\nabla_{\mathbf{o}_t}L = \hat{\mathbf{y}}_t - \mathbf{1}_{y_t}$,
   $\nabla_{\mathbf{h}_\tau}L = \mathbf{V}^{\top}\nabla_{\mathbf{o}_\tau}L$, the
   backward recursion $\nabla_{\mathbf{h}_t}L = \mathbf{U}^{\top}\mathrm{diag}(1-\mathbf{h}_{t+1}^2)\nabla_{\mathbf{h}_{t+1}}L + \mathbf{V}^{\top}\nabla_{\mathbf{o}_t}L$,
   and the parameter gradients as sums over $t$ of per-step contributions
   (Goodfellow Eqs. 10.17–10.28; Jurafsky's letters).
3. **The gradient across a lag is a product of Jacobians** — unrolling the
   recursion, $\partial L_\tau/\partial\mathbf{h}_k$ contains
   $\prod_{s=k+1}^{\tau}\mathbf{U}^{\top}\mathrm{diag}(1-\mathbf{h}_s^2)$; stated as
   a corollary and handed to Class 28 (Goodfellow Eq. 10.21 iterated, §10.7).
4. **Cost** — forward and backward passes are $O(\tau)$ in time and the
   backward pass needs every $\mathbf{h}_t$ stored, $O(\tau)$ memory; the forward
   pass is inherently sequential (Goodfellow §10.2).
5. **Truncated BPTT** — back-propagating only $k$ steps drops from
   $\nabla_{\theta}L$ exactly the contributions of paths longer than $k$, so a
   dependency of lag greater than $k$ receives no gradient. *In neither book as
   a statement; Jurafsky §14.1.2 describes segmenting, Williams & Peng (1990)
   state it.*
6. **The bidirectional state depends on the whole input** — with
   $\mathbf{h}_t = [\mathbf{h}^f_t; \mathbf{h}^b_t]$ the output at $t$ is a
   function of $\mathbf{x}_1 \ldots \mathbf{x}_n$; the cost is twice a
   unidirectional pass, and the network is not causal, so it cannot run online
   or generate (Goodfellow §10.3, Jurafsky Eqs. 14.16–14.18).

Because 5 is outside both books, Class 26 has a **C variant** removing it.

**Built.** Class 26 ships as A (53 pages), B (59 pages) and C (57 pages)
after the first corrections round, which added an eight-slide
introduction — where convolution left us, what a sequence is, what NLP
asks, why a window fails (Jurafsky's example 14.19), sharing across space
against sharing across time, the network reading the sentence, and the
map of the block — and made every later slide refer back to it; this
comparative, motivate-before-formalise pattern is to be carried into the
corrections of 27–29. Its narrative order is then: the network with a cycle, the simple recurrent
network in four equations, unfolding and its two consequences, the three
design patterns, the summed loss; BPTT in two theorems (the recursion over
the states, the sums over time for the parameters), checked to 10⁻¹¹
against finite differences, the product of Jacobians as a corollary handed
to Class 28, the per-step terms measured; the O(τ) cost measured,
truncation stated (the measured slide was removed in the corrections; the
figure stays in `src/`); why a second direction, the bidirectional RNN
defined, what its output depends on, both books' drawings recreated, and
the influence maps of both kinds of network. The A variant uses Jurafsky Figs. 14.1–14.4,
14.11 and Goodfellow Figs. 10.3–10.5, 10.11.

Figures to build:

| Figure | What it computes |
|---|---|
| `unfolding` | Goodfellow Figs. 10.2–10.3's layout: the circuit with the delay square, and the unfolded graph with $\mathbf{U}, \mathbf{W}, \mathbf{V}$ shared, in our colours |
| `design_patterns` | Goodfellow Figs. 10.3–10.5 side by side: hidden recurrence, output recurrence, single output |
| `bptt_flow` | Jurafsky Fig. 14.4's staircase with the backward pass drawn: the two gradients arriving at each $\mathbf{h}_t$ |
| `bptt_check` | the tape's BPTT against finite differences for every entry of $\mathbf{U}, \mathbf{W}, \mathbf{V}$ on a small RNN — the theorem, verified numerically |
| `time_contributions` | the per-step terms of Eq. 10.26, $\|\mathrm{diag}(1-\mathbf{h}_t^2)\,\nabla_{\mathbf{h}_t}L\,\mathbf{h}_{t-1}^{\top}\|$, against $t$ for a loss at the last step: which steps the gradient comes from |
| `cost` | wall-clock time of forward and backward passes and stored states against $\tau$ — linear |
| `truncated_bptt` | the recall task of Class 28 at a fixed lag, trained with truncation $k$ below and above the lag: accuracy against $k$ |
| `bidirectional` | Goodfellow Fig. 10.11's layout in our colours, with the states $\mathbf{h}^f_t$, $\mathbf{h}^b_t$ named as Jurafsky does |

### Class 27 — Encoder-decoder Seq-to-Seq

Jurafsky & Martin (Aug 2026 draft):

- §14.2 *RNNs as Language Models*, opening and §14.2.1 only — Eqs. 14.4–14.9,
  Fig. 14.5. The three equations and the chain rule, as the bridge; nothing
  on training or weight tying (Class 29)
- §14.3.3 *Generation with RNN-based language models* — Fig. 14.9,
  autoregressive generation with $\langle s\rangle$ and $\langle/s\rangle$
- **§14.7 *The Encoder–Decoder Model with RNNs* in full** — Eqs. 14.28–14.33,
  Figs. 14.16–14.18; §14.7.1 *Training* — teacher forcing, Fig. 14.19
- **§14.8 *Attention*** — Eqs. 14.34–14.37, Figs. 14.20–14.22, dot-product
  and bilinear scores; stopping before anything self-attentive
- §13.2 *Machine Translation using Encoder–Decoder*, opening — Eq. 13.7 and
  the parallel-corpus framing; §13.2.2 on training data in one line
- **§13.4 *Decoding in MT: Beam Search*** — greedy versus beam, Figs. 13.6–13.7,
  length normalisation; §13.4.1 minimum Bayes risk in one line

Goodfellow, Bengio & Courville:

- §10.2.1 *Teacher Forcing and Networks with Output Recurrence* — Eqs.
  10.15–10.16, Figs. 10.4, 10.6: teacher forcing as maximum likelihood, and
  the train/test mismatch
- §10.2.3 *Recurrent Networks as Directed Graphical Models* — only the part
  about sequence length: Eq. 10.34 and the end-of-sequence symbol
- §10.2.4 *Modeling Sequences Conditioned on Context with RNNs* — Eq. 10.35,
  Figs. 10.9–10.10: a fixed vector as extra input at every step versus as the
  initial state — this is exactly Jurafsky's choice in Eq. 14.32
- **§10.4 *Encoder–Decoder Sequence-to-Sequence Architectures*** — Fig. 10.12,
  the context $C$, and the remark that the bottleneck motivated attention

Publications supplying results neither book states formally: Cho et al.
(2014); Sutskever, Vinyals & Le (2014); Bahdanau, Cho & Bengio (2015);
Luong, Pham & Manning (2015) for the bilinear score; Williams & Zipser (1989)
for teacher forcing; Ranzato et al. (2016) and Bengio et al. (2015) for
exposure bias and scheduled sampling.

Results the deck states, and which are proved:

1. **Chain-rule factorisation** — $p(\mathbf{y}\mid\mathbf{x}) = \prod_t p(y_t \mid \mathbf{y}_{<t}, \mathbf{x})$
   is exact, not an assumption (Jurafsky Eq. 14.31; Goodfellow Eq. 10.31).
2. **Teacher forcing is maximum likelihood** — the per-token cross-entropy
   with gold inputs is $-\log p(\mathbf{y}\mid\mathbf{x})$ exactly, by 1
   (Goodfellow Eqs. 10.15–10.16).
3. **The context is a convex combination** — with attention,
   $\mathbf{c}_i = \sum_j \alpha_{ij}\mathbf{h}^e_j$ lies in the convex hull of
   the encoder states; the fixed context $\mathbf{c} = \mathbf{h}^e_n$ is the
   vertex of that hull the bottleneck restricts it to. *In neither book.*
4. **Exposure bias** — under free-running decoding the model is evaluated on
   prefixes it never saw in training; a one-line statement of the mismatch
   Goodfellow describes in §10.2.1, with the scheduled-sampling remedy named.
   *Formal statement in neither book.*
5. **Greedy decoding is not optimal, and beam search is not either** — a
   two-step example where the greedy path loses; beam search costs
   $O(k\,m\,|V|)$ against $|V|^m$ for exhaustive search. *Complexity bound in
   neither book.*

Because 3–5 rest on statements outside both books, Class 27 has a **C
variant** that removes the convex-hull proposition, the exposure-bias
proposition and the beam-search bound, keeping the books' own words.

**Built.** Class 27 ships as A (48 pages), B (53 pages) and C (49 pages)
after its corrections round, which gave it Class 26's shape: a four-slide
opening (where Class 26 left us, the translation example and what moves in
it, why the tagger/classifier/language-model/window each fail, and the
read-then-write idea), every later slide phrased on the example, and the
book's figures recreated in full in the deck's palette — Jurafsky Figs.
14.9, 14.17–14.22 and 13.7 and Goodfellow Figs. 10.9 and 10.12 — with the
bottleneck experiment split into setup and result and the beam run redrawn
as an example-driven tree. Its narrative order is then: the recurrent language model in three equations
and autoregressive generation; the chain rule; the encoder–decoder and its
context, fed as initial state and at every step; teacher forcing as
maximum likelihood, the train/test mismatch, and the end symbol; the
bottleneck measured, attention defined and its weights shown, the
convex-hull reading, and what attention is not yet; greedy against beam
decoding on the trained model. The recurrent cell is a black box
throughout. One correction to the plan: in the August 2026 draft Fig. 13.6
is the transformer block, not a greedy-search picture, so the decoding
slides cite Fig. 13.7 only.

Figures to build (every number measured):

| Figure | What it computes |
|---|---|
| `seq2seq_unrolled` | the encoder–decoder unrolled on "the green witch arrived → llegó la bruja verde", Jurafsky Fig. 14.18's layout in our colours, with $\mathbf{c} = \mathbf{h}^e_n$ |
| `teacher_forcing` | training against inference, Fig. 14.19's layout: gold inputs and per-token losses versus the model's own outputs fed back |
| `bottleneck` | a toy copy/reverse task: a small RNN encoder–decoder trained here, accuracy against source length with and without attention — the bottleneck measured |
| `attention_weights` | the $\alpha_{ij}$ matrix of the trained attention model on a reversal task: the anti-diagonal it learns, Fig. 14.22's computation drawn |
| `context_hull` | encoder states in 2-D, the attention context as a point inside their hull, the fixed context as a vertex |
| `beam_search` | Jurafsky Fig. 13.7's layout: a beam of width 2 over a small vocabulary with log-probabilities, the greedy path marked, the winning path different |

### Class 28 — LSTM and Gated RNNs

Goodfellow, Bengio & Courville:

- **§10.7 *The Challenge of Long-Term Dependencies* in full** — Eqs.
  10.36–10.39 (the power method $\mathbf{W}^t = \mathbf{Q}\Lambda^t\mathbf{Q}^\top$),
  Fig. 10.15, and the remark that the same product of weights that vanishes
  in a recurrent net does not in a feedforward net because the weights differ
- §10.8 *Echo State Networks* — one slide: the spectral radius as the knob,
  and why fixing the recurrent weights is a way out
- §10.9 *Leaky Units and Other Strategies for Multiple Time Scales* — §10.9.1
  skip connections through time, §10.9.2 the leaky unit
  $\mu_t = \alpha\mu_{t-1} + (1-\alpha)v_t$ as the ancestor of the gate,
  §10.9.3 in one line
- **§10.10 *The LSTM and Other Gated RNNs* in full** — Fig. 10.16; §10.10.1
  LSTM Eqs. 10.40–10.44; §10.10.2 GRU Eqs. 10.45–10.47
- **§10.11 *Optimization for Long-Term Dependencies*** — §10.11.1 clipping,
  Eqs. 10.48–10.49, Fig. 10.17; §10.11.2 regularising to encourage
  information flow, Eqs. 10.50–10.52, one slide
- §10.12 *Explicit Memory* — one slide, Fig. 10.18, as where the gate idea
  leads (memory networks), no more

Jurafsky & Martin:

- **§14.5 *The LSTM* in full** — the "flights … were" example (14.19), the
  two jobs of the hidden state, vanishing gradients, Eqs. 14.20–14.27,
  Figs. 14.13–14.14; §14.5.1 gated units as drop-in modules
- Class 26's BPTT is assumed; the chain of Jacobians is recalled in one
  equation, not re-derived

Publications supplying results neither book states formally: Hochreiter
(1991); Bengio, Simard & Frasconi (1994); Hochreiter & Schmidhuber (1997);
Gers, Schmidhuber & Cummins (2000) for the forget gate; Cho et al. (2014) for
the GRU; Pascanu, Mikolov & Bengio (2013) for the clipping analysis; Jaeger
(2001) for echo state networks; Jozefowicz et al. (2015) and Greff et al.
(2017) for the ablations.

Results the deck states, and which are proved:

1. **Vanishing and exploding, linear case** — $\mathbf{h}_t = \mathbf{W}^t\mathbf{h}_0 = \mathbf{Q}\Lambda^t\mathbf{Q}^\top\mathbf{h}_0$:
   every component with $|\lambda_i| < 1$ decays geometrically and every one
   with $|\lambda_i| > 1$ explodes (Goodfellow Eqs. 10.36–10.39).
2. **The gradient bound** — $\|\partial\mathbf{h}_t/\partial\mathbf{h}_k\| \le (\|\mathbf{U}\|\,\gamma)^{t-k}$
   with $\gamma$ the bound on $|g'|$: the sufficient condition for vanishing
   is $\|\mathbf{U}\| < 1/\gamma$ (Pascanu et al. 2013). *In neither book as a
   theorem; Goodfellow states the conclusion.*
3. **The leaky unit is a running average** — unrolling
   $\mu_t = \alpha\mu_{t-1} + (1-\alpha)v_t$ gives geometric weights with
   time constant $1/(1-\alpha)$ (Goodfellow §10.9.2).
4. **The cell state's gradient is a product of forget gates** —
   $\partial\mathbf{c}_t/\partial\mathbf{c}_k = \prod_{s=k+1}^{t}\mathrm{diag}(\mathbf{f}_s)$
   with no weight matrix in the product, so a gate near one carries the
   gradient unattenuated (the "constant error carousel"). *In neither book as
   an equation; both describe it in words.*
5. **The GRU state is a convex combination** — $\mathbf{h}_t = (1-\mathbf{u}_t)\odot\mathbf{h}_{t-1} + \mathbf{u}_t\odot\tilde{\mathbf{h}}_t$
   stays in the convex hull of the old state and the candidate (Goodfellow
   Eq. 10.45).
6. **Clipping preserves direction** — $\mathbf{g} \to \mathbf{g}v/\|\mathbf{g}\|$
   changes the step length only, so the descent direction is unchanged and the
   step is bounded by $v$ (Goodfellow Eqs. 10.48–10.49).

Because 2 and 4 are outside both books, Class 28 has a **C variant**
removing the gradient-norm bound and the forget-gate product, keeping the
books' descriptions.

**Built.** Class 28 ships as A (39 pages), B (41 pages) and C (39 pages).
Its narrative order is: the power iteration and the gradient bound, with the
gradient by lag measured for three trained cells; skip connections, leaky
units and echo state networks as the ancestors of the gate; the LSTM in
equations, drawn in TikZ in the book's layout, the constant error carousel,
the gates' activity while a fact is held, and the cell as a drop-in module;
the GRU and its convex update, the three cells compared on one recall task;
norm clipping at a cliff, run, the information-flow regulariser and explicit
memory. The `lstm_cell` and `gru_cell` entries in the table are drawn in
TikZ inside the deck rather than by matplotlib; the A variant uses
Goodfellow Figs. 10.15–10.18 and Jurafsky Figs. 14.13–14.14 in their
place, and drops the three measured-only slides.

Figures to build:

| Figure | What it computes |
|---|---|
| `power_iteration` | $\|\mathbf{W}^t\mathbf{h}_0\|$ against $t$ for spectral radii $0.9, 1.0, 1.1$, and Goodfellow Fig. 10.15's composition of linear–tanh layers on a 1-D cross-section, recomputed |
| `gradient_lag` | $\|\partial L_t/\partial\mathbf{h}_{t-k}\|$ against lag $k$ for a simple RNN and an LSTM at initialisation and after training on a toy task — the vanishing measured |
| `leaky_unit` | the leaky unit's impulse response for three $\alpha$, time constants read off |
| `lstm_cell` | the cell as a computation graph, Jurafsky Fig. 14.13 / Goodfellow Fig. 10.16 layout in our colours, gates labelled with their equations |
| `gru_cell` | the same for the GRU |
| `gate_activity` | the trained LSTM on a "remember the first symbol" task: forget-gate values over time — near one while remembering, dropping when the symbol is emitted |
| `long_dependency` | accuracy against sequence length on that task: RNN, LSTM, GRU — Jurafsky's claim that LSTMs handle distance, measured |
| `clipping` | Goodfellow Fig. 10.17's layout: a loss surface with a cliff, the step with and without clipping |

### Class 29 — RNN for NLP tasks

Jurafsky & Martin:

- **§14.2 *RNNs as Language Models* in full** — §14.2.1 forward inference
  (recalled from 27), **§14.2.2 training** with self-supervision, Eqs.
  14.10–14.11, Fig. 14.6; **§14.2.3 weight tying**, Eqs. 14.12–14.14
- **§14.3 *RNNs for other NLP tasks* in full** — §14.3.1 sequence labelling,
  Fig. 14.7; §14.3.2 sequence classification, end-to-end training, pooling
  Eq. 14.15, Fig. 14.8; §14.3.3 generation, Fig. 14.9, sampling
- **§14.4 *Stacked and Bidirectional RNN architectures*** — §14.4.1 stacked,
  Fig. 14.10; §14.4.2 bidirectional *as applied* to labelling and
  classification, Eqs. 14.16–14.18, Figs. 14.11–14.12 (the definition is
  Class 26's; here it is used)
- **§14.6 *Summary: Common RNN NLP Architectures*** — Fig. 14.15, the four
  architectures side by side, as the closing slide of the block
- §3.3 *Evaluating Language Models: Perplexity* — Eq. 3.14 only, since it
  is how a language model is scored; §3.7 in one line for the entropy link

Goodfellow, Bengio & Courville:

- §10.2 opening — Eqs. 10.8–10.14 (the forward equations and the summed loss,
  in his notation, reconciled once), Fig. 10.3
- **§10.2.3 *Recurrent Networks as Directed Graphical Models*** — Eqs.
  10.29–10.33, Figs. 10.7–10.8: the RNN as a parameter-sharing scheme for a
  fully connected graphical model over the sequence, and what that buys
- §10.3 *Bidirectional RNNs* — Fig. 10.11, as used (recalled)
- **§10.5 *Deep Recurrent Networks*** — Fig. 10.13, the three ways to make an
  RNN deep, and why stacking helps (Pascanu et al. 2014)
- §10.6 *Recursive Neural Networks* — one slide, Fig. 10.14: trees instead of
  chains, depth $O(\log n)$

Publications supplying results neither book states formally: Elman (1990);
Mikolov et al. (2010); Press & Wolf (2017) and Inan et al. (2017) for weight
tying; Schuster & Paliwal (1997); Pascanu et al. (2014) for deep RNNs;
Socher et al. (2011) for recursive nets; Graves (2013) for generation.

Results the deck states, and which are proved:

1. **Cross-entropy per token is log-perplexity** — the average training loss
   of a language model equals $\log$ of its perplexity, so the two numbers
   are one number (Jurafsky Eqs. 3.14 and 14.10).
2. **Weight tying halves the embedding parameters** — with
   $\mathbf{V} = \mathbf{E}^\top$ the model drops $|V|\,d$ parameters, which for
   a $50{,}000$-word vocabulary at $d = 512$ is $25.6$M (Jurafsky §14.2.3).
3. **The RNN is a parameter-sharing scheme for the fully connected graphical
   model** — the number of parameters is independent of sequence length,
   against $O(|V|^\tau)$ for the unshared model (Goodfellow §10.2.3).
4. **Last state versus mean pooling** — the last state's gradient reaches
   token $t$ through $n-t$ recurrent steps, the mean's through one, so the
   pooled classifier sees the beginning of the sequence as well as the end.
   *In neither book as a statement; Jurafsky motivates it in words.*
5. **A stacked RNN's parameter count** — $L$ layers of width $d$ cost
   $O(L d^2)$, one layer of width $\sqrt{L}\,d$ costs the same; what stacking
   buys is depth per time step, not parameters (Goodfellow §10.5).

Because 4 is outside both books, Class 29 has a **C variant** removing it.

**Built.** Class 29 ships as A (39 pages), B (40 pages) and C (39 pages).
Its narrative order is: the recurrent language model recalled, training by
self-supervision, perplexity as the exponent of the loss, a character LSTM
trained and sampled, weight tying counted and tried; labelling and
classification with the three read-outs, their gradient paths stated and
measured; stacking and bidirectionality as modules, each tried; the RNN as
the complete graphical model, deciding when to stop, the recursive net;
the four architectures as the closing figure. Result 1 (perplexity as the
exponent of the mean cross-entropy) turned out to be stated by Jurafsky
(Eq. 3.42 and the exercise of §7.5), so C keeps it; the `char_lm` corpus is
O. Henry's *The Gift of the Magi* (11,136 characters), and `rnn_lm_training`,
`four_architectures` and the stacked/bidirectional layouts are drawn in the
book's layouts rather than measured. The A variant uses Jurafsky Figs.
14.6–14.12, 14.15 and Goodfellow Figs. 10.7, 10.8, 10.13, 10.14.

Figures to build:

| Figure | What it computes |
|---|---|
| `rnn_lm_training` | Jurafsky Fig. 14.6's layout: the unrolled LM with per-token losses on a real sentence, in our colours |
| `char_lm` | a character-level RNN LM trained here on a small public-domain text: training and validation perplexity against epoch, and samples at three temperatures |
| `weight_tying` | parameter count against vocabulary size with and without tying, at $d = 512$; the perplexity of the tied and untied char-LM |
| `pos_tagging` | Fig. 14.7's layout on "Janet will back the bill", with the tag distributions from a tagger trained here on a small synthetic grammar |
| `pooling` | last-state, mean and max pooling on a toy sentiment task where the signal sits at the start of the sequence: accuracy against length |
| `bidirectional_labelling` | the same labelling task with a left-to-right RNN and a bidirectional one: where each fails |
| `stacked_depth` | Fig. 14.10 / Goodfellow Fig. 10.13's layout, and the char-LM's perplexity against number of layers at matched parameter count |
| `four_architectures` | Jurafsky Fig. 14.15 in our colours: the four output structures side by side |

---

## What is explicitly not in this block, and where it goes

| Deferred | To |
|---|---|
| The simple RNN, unrolling, BPTT, the gradient (Goodfellow §10.2.2, Jurafsky §14.1) | Class 26 |
| Bidirectional RNNs as a definition (Goodfellow §10.3, Jurafsky §14.4.2) | Class 26; used in 27 and 29 |
| Self-attention, multi-head attention, positional encoding, transformer encoders and decoders (Jurafsky Ch. 7 and 13 §13.3) | Classes 30–31 |
| Transformer-based MT, MT evaluation, BLEU, bias (Jurafsky §13.3, §13.5–13.7) | not in the course plan |
| Contextual embeddings, BERT, fine-tuning (Jurafsky Ch. 9) | not in the course plan |
| Speech features (Jurafsky Ch. 15) | not in the course plan |

## Cross-references to honour

- Class 11 covered backpropagation and Class 26 its application through
  time. Class 28 recalls the product of Jacobians in one equation and
  measures it; it does not re-derive it.
- Class 13 introduced the Hessian eigen-decomposition and the power-method
  picture of a linear recurrence. Class 28's Result 1 is the same picture
  applied to $\mathbf{U}$, and says so.
- Class 18 discussed parameter sharing as a hard constraint and named
  convolution as the instance; Class 29's Result 3 names recurrence as the
  other instance.
- Class 19 covered dropout and early stopping; Class 29's language-model
  training slides use both by name and do not re-explain them.
- Class 24's encoder–decoder for segmentation (down, then back up) is the
  same idea in space that Class 27's is in time; one line, no more.
- Classes 30–31 will replace the recurrent encoder and decoder with
  attention alone. Class 27 ends by saying that the attention it introduced
  is the seed of that, and stops.

## Notation

The frozen course notation (`_shared/NOTATION.md`) applies. Time is a
subscript, $\mathbf{h}_t$; Jurafsky's letters $\mathbf{W}$ (input),
$\mathbf{U}$ (recurrent), $\mathbf{V}$ (output) are used and Goodfellow's swap
is noted once per deck. Encoder and decoder states carry superscripts $e$ and
$d$. The LSTM gates are $\mathbf{f}_t, \mathbf{i}_t, \mathbf{o}_t$ with
candidate $\mathbf{g}_t$ and cell $\mathbf{c}_t$ (Jurafsky's letters; Goodfellow's
$\mathbf{s}$ for the cell and $\mathbf{q}$ for the output gate are noted). The
GRU's update and reset gates are $\mathbf{u}_t$ and $\mathbf{r}_t$, as in
Goodfellow. Attention weights are $\alpha_{ij}$ over encoder position $j$ at
decoder step $i$, as in Jurafsky.

# Class 28 — figure sources

Every measured figure comes from one experiment run here, on the
reverse-mode autodiff tape written for this block (the sandbox has no
PyTorch): a simple RNN, an LSTM and a GRU of width 32 trained the same way
on a recall task — read 4 to 24 symbols, emit the first — and tested to
length 60. Nothing is loaded from a library, nothing is traced from a book,
and every number on a slide is printed by the code below.

```bash
cd src
python3 build_all.py                     # all figures -> ../figs/
python3 build_all.py fig04               # only matching modules
python3 figures/fig04_long_dependency.py # or one directly
```

The first run trains the three cells (a few minutes) and caches them in
`../figs/_cache.pkl`; delete the cache to retrain.

## What `common/` provides

| Module | What it is |
|---|---|
| `tape.py` | a minimal reverse-mode tape: tensors, matmul, tanh, sigmoid, softmax, cross-entropy, concat, Adam with norm clipping; checked against finite differences on import |
| `models.py` | `RNNCell`, `LSTMCell` (forget bias 1), `GRUCell` in Jurafsky's letters; the recall task; training; accuracy by length; `grad_by_lag`, which reads ∂L_T/∂h_k off the tape; the power iteration, the linear–tanh composition, the leaky unit and the two-parameter cliff loss |
| `experiment.py` | the shared experiment: the three cells, 3,000 steps, lengths 4–24, tested to 60, gradients by lag before and after training |
| `style.py` | the deck's palette and figure sizes |

## The figures

| Module | Figure | What is computed or drawn |
|---|---|---|
| `fig01_power_iteration` | `power_iteration` | (a) the factor λᵗ for five eigenvalues; (b) ‖Uᵗh₀‖ for a random 20×20 matrix at spectral radius 0.9, 1.0, 1.1 |
| `fig07_composition` | `composition` | a random linear–tanh layer composed 1–5 times along one line through a 100-dimensional state (Goodfellow Fig. 10.15 recomputed) |
| `fig02_gradient_lag` | `gradient_lag` | ‖∂L_T/∂h_{T−k}‖ against lag on length-40 sequences, three cells, at initialisation and after training |
| `fig08_gradient_paths` | `gradient_paths` | the three paths, drawn: skip connections, a leaky self-loop (with impulse responses), a gated self-loop |
| `fig03_leaky_unit` | `leaky_unit` | impulse responses of the leaky unit for three α |
| `fig09_lstm_cell` | `lstm_cell` | the layout of Jurafsky Fig. 14.13: inputs, projection fans, the four gates, the cell line |
| `fig10_lstm_block` | `lstm_block` | the layout of Goodfellow Fig. 10.16: input, three gates, the state's self-loop, the output |
| `fig11_dropin` | `dropin_modules` | the layout of Jurafsky Fig. 14.14: feedforward, simple recurrent and LSTM units |
| `fig05_gate_activity` | `gate_activity` | the trained LSTM's forget gates over one length-40 sequence, and the mean forget and add gates |
| `fig04_long_dependency` | `long_dependency` | training loss and accuracy by length for the three trained cells |
| `fig06_clipping` | `clipping` | the loss surface J(w, b) of h_t = w h_{t−1} + b over 50 steps, in three dimensions, with descent from the same start with and without norm clipping (Goodfellow Fig. 10.17's layout) |
| `fig12_explicit_memory` | `explicit_memory` | the layout of Goodfellow Fig. 10.18: memory cells, task network, writing and reading mechanisms |

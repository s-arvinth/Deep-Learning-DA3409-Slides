# Class 26 — figure sources

Every measured figure comes from experiments run here, on the reverse-mode
autodiff tape written for this block (the sandbox has no PyTorch). The
theorem of the class — back-propagation through time, Goodfellow Eqs.
10.17–10.28 — is written out by hand in NumPy in `common/models.py` and
checked against the tape and against finite differences; nothing is loaded
from a library, nothing is traced from a book, and every number on a slide
is printed by the code below.

```bash
cd src
python3 build_all.py                     # all figures -> ../figs/
python3 build_all.py fig07               # only matching modules
python3 figures/fig04_bptt_check.py      # or one directly
```

The first run trains the small models (about a minute) and caches every
stage in `../figs/_cache.pkl`; delete the cache to rerun.

## What `common/` provides

| Module | What it is |
|---|---|
| `tape.py` | the minimal reverse-mode tape of Classes 27–29; checked against finite differences on import |
| `models.py` | `SimpleRNN` (Jurafsky Eqs. 14.1–14.3 / Goodfellow Eqs. 10.8–10.11) on the tape and in plain NumPy; `bptt`, Eqs. 10.17–10.28 by hand with the per-step terms; `finite_difference`; truncated training on the recall task; `timing`; `Bidirectional` and `influence_map` |
| `experiment.py` | the cached stages: the gradient check, the per-step terms before and after training, the timings, truncation, the influence maps |
| `draw.py` | the layout helpers of Class 27 in the deck's colours |
| `style.py` | the deck's palette and figure sizes |

## The figures

| Module | Figure | What is computed or drawn |
|---|---|---|
| `fig00_space_vs_time` | `space_vs_time` | the introduction: one kernel over a grid against one cell over a sequence with a state passed along |
| `fig00_sequence_tasks` | `sequence_tasks` | the four shapes of NLP problem (labelling, classification, next token, sequence to sequence) |
| `fig00_window_fails` | `window_fails` | Jurafsky's sentence (14.19): the agreement six words apart, a window of three, and a state carried word by word |
| `fig00_running_example` | `running_example` | the network reading the sentence, one word at a time, schematic |
| `fig00_elman` | `elman_network`, `elman_feedforward` | the layouts of Jurafsky Figs. 14.1 and 14.2 |
| `fig01_unfolding` | `unfolding` | the layouts of Goodfellow Figs. 10.2–10.3: circuit and unfolded graph |
| `fig02_design_patterns` | `design_patterns` | the layouts of Goodfellow Figs. 10.3–10.5 |
| `fig03_bptt_flow` | `bptt_flow` | the layout of Jurafsky Fig. 14.4 with the backward pass drawn |
| `fig04_bptt_check` | `bptt_check` | BPTT by hand against finite differences on 224 sampled entries, and against the tape |
| `fig05_time_contributions` | `time_contributions` | ‖∇_{h_t}L‖ and the per-step terms of Eq. 10.26 against t, at initialisation and after training |
| `fig06_cost` | `cost` | wall-clock of forward and backward passes and stored states against τ |
| `fig07_truncated_bptt` | `truncated_bptt` | gradient reaching the first input, and accuracy, against the truncation k on the length-12 recall task (marker embeddings start identical); built, not on a slide since the corrections round |
| `fig08_bidirectional` | `bidirectional_goodfellow`, `bidirectional_jurafsky`, `influence_maps` | the layouts of Goodfellow Fig. 10.11 and Jurafsky Fig. 14.11; ‖∂o_t/∂e_s‖ maps for a one- and a two-directional RNN |

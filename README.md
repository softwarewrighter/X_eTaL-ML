# X_eTaL ML

<p align="center">
  <img src="images/modern-xetal-logo.jpg" alt="X_eTaL: eXperimental Extensible Typed Array Language" width="480">
</p>

<p align="center">
  <b><a href="https://softwarewrighter.github.io/X_eTaL-ML/">The live ML demos</a></b>
  -- every demo running in your browser (WebAssembly)<br>
  <b><a href="https://softwarewrighter.github.io/X_eTaL-ML/recorded/">Run them yourself</a></b>
  -- every demo at the command line, from a clone<br>
  <b><a href="https://softwarewrighter.github.io/X_eTaL-ML/doc/">Read the code</a></b>
  -- the cross-reference of every demo program and library (`xetal doc`)
</p>

Machine learning in [X_eTaL](https://github.com/softwarewrighter/X_eTaL),
the eXperimental Extensible Typed Array Language: small models you can
watch think, and the ML libraries they are made of.

## Why an array language for ML

Most of what an ML framework hides behind layers and modules is array
algebra. A dense layer is one inner product and an addition. Softmax
is an exponential, a row sum and a division. Attention is two matrix
products and a softmax. A mixture-of-experts router is a matrix
product and a top-k. Quantizing a network to ternary weights is a
rounding of one array.

In X_eTaL each of those is one short, typed expression over whole
arrays, so a demo can show the model itself: the weights, every
intermediate array with its shape, and the moment where a loop nest
you would write elsewhere turns out to be a single array
transformation.

Every demo here follows one arc:

> a small X_eTaL program -> a visible model -> the array
> transformations stepped through -> something about ML revealed.

## How X_eTaL differs from APL

X_eTaL keeps APL's whole-array thinking but is designed with ideas
from Haskell and Rust: ASCII source rendered as mathematical
typography, inferred static types (Haskell-style inference, so the
terse code needs no annotations, and pieces that cannot compose are
rejected before they run), and explicit, checked interfaces in the
Rust spirit. For ML that means a shape or type mistake in a layer is
an error before the forward pass, not a wrong number after it.

"Extensible" has three meanings, and this repo uses them in turn:

- **Libraries extend the vocabulary.** ML building blocks (softmax,
  dense layers, quantization, convolution, attention) are ordinary
  `.xtl` libraries here, imported with `u_se<`, typed and tested.
- **Macros extend the language.** The [Net](libs/Net/docs/README.md) macro
  library lets a network be written as
  `"784 128 relu 10 softmax" net:n_etwork< "w1 w2"`, which expands into
  ordinary, visible, type-checked code (`xetal expand` shows it).
- **Native extensions extend the machine.** Fast kernels and data
  loaders can come from Rust through typed facades
  ([X_eTaL-extensions](https://github.com/softwarewrighter/X_eTaL-extensions)).

## The X_eTaL repositories

| Repo | What it is |
| ---- | ---------- |
| [X_eTaL](https://github.com/softwarewrighter/X_eTaL) | the language, interpreter, REPL, editor, notebook and browser playground |
| [X_eTaL-demos](https://github.com/softwarewrighter/X_eTaL-demos) | visual array and scientific computing demos |
| **X_eTaL-ML** (this repo) | machine-learning demos and libraries |
| [X_eTaL-games](https://github.com/softwarewrighter/X_eTaL-games) | interactive games and puzzles |
| [X_eTaL-libraries](https://github.com/softwarewrighter/X_eTaL-libraries) | reusable function and macro libraries |
| [X_eTaL-extensions](https://github.com/softwarewrighter/X_eTaL-extensions) | native Rust extensions and the extension ABI |

## Demos

All eight run live in your browser and at the command line.
[Start with the Tiny CNN](https://softwarewrighter.github.io/X_eTaL-ML/cnn-digits/):
draw a digit and watch a network read it.

| Demo | What you see | The line that does it |
| ---- | ------------ | --------------------- |
| [Tiny CNN](demos/cnn-digits/README.md) ([live](https://softwarewrighter.github.io/X_eTaL-ML/cnn-digits/)) | draw a digit; filters, feature maps, pooling and probabilities, every stage computed by X_eTaL; click a map for its arithmetic | `-1 0 1 o_-_2 -1 0 1 o_-_2 x`: every 3 x 3 window of the picture at once |
| [Backprop microscope](demos/backprop/README.md) ([live](https://softwarewrighter.github.io/X_eTaL-ML/backprop/)) | one training step with every array shown: forward, loss, each layer's gradient, the step; every gradient checked by nudging its weight; take steps and watch the loss fall | `(o_\ u:o_nes X) '+ '* i_nner D1`: a layer's gradient, one matrix product |
| [CNN training](demos/cnn-backprop/README.md) ([live](https://softwarewrighter.github.io/X_eTaL-ML/cnn-backprop/)) | X_eTaL trains a tiny CNN from random weights on 600 handwritten digits in your browser: the backward pass through softmax, dense, max-pooling, ReLU and the convolution, checked by finite differences; the filters form and the test digits come to be read right | `GK := DM '+ '* i_nner o_\ V`: the filters' gradient over a whole batch, one matrix product |
| [Training live](demos/train-live/README.md) ([live](https://softwarewrighter.github.io/X_eTaL-ML/train-live/)) | X_eTaL trains a small network on a spiral in your browser; the decision regions bend to follow it as the loss falls | `s := 25 'u:a_dam p_ower s`: Adam's step, iterated |
| [Attention microscope](demos/attention/README.md) ([live](https://softwarewrighter.github.io/X_eTaL-ML/attention/)) | one head of attention on your sentence: scores, weights, a causal mask; "tired" finds the animal, "wide" the street | `(Q '+ '* i_nner o_\ K) / 2.0 ^ 0.5`: every query against every key |
| [MoE routing microscope](demos/moe-router/README.md) ([live](https://softwarewrighter.github.io/X_eTaL-ML/moe-router/)) | a sentence's tokens routed to their top-2 of 16 experts; nudge a token and see where the experts switch | `u:t_op2 u:s_oftmax u:s_cores x`: scores, softmax and top-2 for all tokens |
| [Network macro](demos/net-macro/README.md) ([live](https://softwarewrighter.github.io/X_eTaL-ML/net-macro/)) | a network written as one line; the function the macro wrote for it; three networks on a spiral; type a spec of your own and train it in the browser | `"u:d_eep c" net:m_odel< "2 16 relu 16 relu 3 softmax"`: a macro call that becomes the weights' loading and the forward function |
| [1.58-bit network](demos/ternary-net/README.md) ([live](https://softwarewrighter.github.io/X_eTaL-ML/ternary-net/)) | one classifier with FP32, FP16, INT8 and ternary weights compared; a ternary layer as additions | `(f_loat x > s) - f_loat x < n_eg s`: a layer's weights to -1, 0, +1 |

Every page shows all the code it runs, beside the arrays that code
computed. Programs are short: weights and other data live in each
demo's `data/` directory, not in the source. Every program and library
is in [the cross-reference](https://softwarewrighter.github.io/X_eTaL-ML/doc/)
(`xetal doc`, `just doc`): its `##` documentation and `###` sections,
its source drawn as X_eTaL draws it, every name linked to where it is
defined and used.

### Run them yourself

Rust (stable), git and [`just`](https://github.com/casey/just);
`just xetal` fetches and builds the X_eTaL commit this repository
pins, so nothing else is installed:

```bash
git clone https://github.com/softwarewrighter/X_eTaL-ML
cd X_eTaL-ML
just xetal               # fetch and build the pinned X_eTaL (once, a few minutes)
just tour cnn-digits     # each statement, then its result
just run attention       # just the results
just serve cnn-digits    # its page, at http://127.0.0.1:8435/
```

## Libraries

| Library | Alias | What |
| ------- | ----- | ---- |
| [NN](libs/NN/docs/README.md) | `nn:` | activations, softmax by row (any rank), dense layers, argmax, one-hot, loss, accuracy |
| [Net](libs/Net/docs/README.md) | `net:` | a macro library: `"u:n_et w" net:m_odel< "784 128 relu 10 softmax"` becomes the weights' loading, checked, and an ordinary function of NN calls, shown by `xetal expand`; `"u:s_tep X Y lr" net:t_rain< "2 4 tanh 2 softmax"` writes the network's backprop and an Adam step |

The demos import it (`"nn:" u_se< "NN"`), at the command line and in
the browser. From
[X_eTaL-libraries](https://github.com/softwarewrighter/X_eTaL-libraries),
pinned like X_eTaL itself, the tests use Check and the pages draw
training curves with Plot (and Strings, Format and Lists, which it
imports).

## Status

| | Today |
| - | ----- |
| Demos | 8, all live, each with command-line and browser tests; three of them train in X_eTaL, a CNN among them |
| Libraries | NN (12 functions, typed, tested); Net (6 macros, every expansion pinned; written gradients checked against finite differences); every item documented, 28 doc examples run by the gate |
| X_eTaL | pinned at X_eTaL main d284a8c, after v0.1.0: tuples with patterns (the training state is one), `h:` private names, the doc site by directory (`XETAL_COMMIT`, `just xetal-version`); X_eTaL-libraries at 69bb369 (Plot's charts, every library documented) |
| Speed | measured and guarded (`just bench-check`, [`docs/speed.md`](docs/speed.md)); the CNN's whole program runs in a third of a second |
| Waiting on X_eTaL | a grade per row, arrays in and out of the browser engine, `e_ach` returning arrays ([`docs/xetal-asks.md`](docs/xetal-asks.md)) |
| Later | more libraries (quantization, convolution, attention, sampling), a tiny GPT, an embedding explorer, training in X_eTaL ([`docs/plan.md`](docs/plan.md)) |

## Build

Requirements: Rust (stable), git, and
[`just`](https://github.com/casey/just). The first `just xetal` needs
the network (it clones X_eTaL).

```bash
just                # list the recipes
just xetal          # fetch and build the pinned X_eTaL (work/xetal, bin/xetal)
just xetal-version  # which commits are pinned, and the binary's own report
just eval "'+ r_/ 1 2 3"
just gate           # the pre-commit gate
```

This repository tracks no copy of X_eTaL: `XETAL_COMMIT` holds the
commit it is known to work with, and `just xetal` clones X_eTaL into
the gitignored `work/xetal/`, checks that commit out, builds it and
links `bin/xetal` to the binary (as X_eTaL's `docs/vendoring.md`
describes). Every recipe runs that binary, so results do not depend
on whatever `xetal` is on your PATH. The libraries used from
X_eTaL-libraries are pinned the same way (`XETAL_LIBRARIES_COMMIT`).
`just xetal-pin [REF]` moves to a newer X_eTaL.

### Layout

- `demos/<slug>/` -- one demo per directory: its `.xtl` programs, a
  README, reg-rs baselines in `reg/`, an offline trainer in `train/`
  when it has learned weights, and its web page in `web/`.
- `libs/<Name>/` -- one library per directory: `src/<Name>.xtl`, its
  reference page `docs/README.md`, `demos/`, and reg-rs `tests/`
  (including `types.rgt`, which pins every export's type).

```bash
just pages                # build the live site into pages/ (not tracked), the cross-reference in pages/doc
just doc                  # the cross-reference alone (xetal doc), into pages/doc
just doc-test             # run every ## >> example in the libraries' doc comments
just publish              # put it live: pages/ becomes the gh-pages branch
just serve-pages          # preview it at http://127.0.0.1:8435/X_eTaL-ML/
just demos                # the demos, in catalog order
just run SLUG             # run a demo's program; just show SLUG as a notebook
just tour SLUG            # the notebook paced, long lines clipped (as recorded)
just record [SLUG]        # re-record the CLI-only demos (vhs, ffmpeg, gif2webp)
just libs                 # the libraries and their aliases
export XETAL_PATH="$(just path)"   # use the libraries from your own programs
just test                 # every demo and library baseline
just bench                # time the ML workloads (docs/speed.md)
```

## Development

Development is tracked with agentrail sagas, as in X_eTaL:
`agentrail status` shows the current step, `agentrail next` its
instructions. Every step ends with the gate passing, docs updated, a
commit to `main` and a push. Every commit is listed in
[`CHANGES.md`](CHANGES.md).

## Related Projects

- [sw-mlpl](https://github.com/sw-ml-study/sw-mlpl) -- Software
  Wrighter's Machine Learning Programming Language, a Rust array
  language inspired by APL, APL2, J, and BQN.
- [microgpt-mlpl](https://github.com/softwarewrighter/microgpt-mlpl)
  and [microgpt-rs](https://github.com/softwarewrighter/microgpt-rs)
  -- Karpathy's microgpt in MLPL and in Rust.

## Links

- Blog: [Software Wrighter Lab](https://software-wrighter-lab.github.io/)
- Discord: [Join the community](https://discord.com/invite/Ctzk5uHggZ)
- YouTube: [Software Wrighter](https://www.youtube.com/@SoftwareWrighter)

## Copyright

Copyright (c) 2026 Michael A Wright

## License

MIT. See [`LICENSE`](LICENSE) and [`COPYRIGHT`](COPYRIGHT).

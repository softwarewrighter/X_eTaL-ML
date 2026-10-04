# X_eTaL ML

<p align="center">
  <img src="images/modern-xetal-logo.jpg" alt="X_eTaL: eXperimental Extensible Typed Array Language" width="480">
</p>

<p align="center">
  <b><a href="https://softwarewrighter.github.io/X_eTaL-ML/">The live ML demos</a></b>
  -- every demo running in your browser (WebAssembly)<br>
  <b><a href="https://softwarewrighter.github.io/X_eTaL-ML/recorded/">Recorded CLI demos</a></b>
  -- the demos that run only at the command line, and how to run any demo yourself
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
- **Macros extend the language.** With `.xtlm` macro libraries (being
  built in X_eTaL) a network can be written as
  `"784 128 relu 10 softmax" net:n_etwork< ...`, expanding into
  ordinary, visible, type-checked code.
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

| Demo | What you see | Array ideas | Status |
| ---- | ------------ | ----------- | ------ |
| [1.58-bit network](demos/ternary-net/README.md) ([live](https://softwarewrighter.github.io/X_eTaL-ML/ternary-net/)) | one classifier with FP32, FP16, INT8 and ternary weights compared; a ternary layer as additions | inner product, quantization, masks | live |
| [MoE routing microscope](demos/moe-router/README.md) ([live](https://softwarewrighter.github.io/X_eTaL-ML/moe-router/)) | a sentence's tokens routed to their top-2 of 16 experts; nudge a token and see where the experts switch | matrix product, softmax, top-k by masks | live |
| [Tiny CNN](demos/cnn-digits/README.md) ([live](https://softwarewrighter.github.io/X_eTaL-ML/cnn-digits/)) | draw a digit; filters, feature maps, pooling and probabilities, every stage computed by X_eTaL; click a map for its arithmetic | windows, convolution, reshape | live |
| Attention microscope | the attention heatmap, rows meeting columns | matrix product, transpose, softmax | next |
| Embedding explorer | PCA from 64 dimensions to a 3-D cloud | covariance, projection | later |
| Gradient descent | a small model trained in X_eTaL itself | outer product, reduce, iteration | later |
| micro-gpt | a tiny GPT's forward pass, sampling names | all of the above | later |
| Network macro | a network written as one macro call and its expansion | `.xtlm` macros | waiting on X_eTaL |

### Recorded at the command line

A demo that runs interactively in the browser is shown live; a demo
that runs only at the command line is shown recorded on the
[recorded CLI demos](https://softwarewrighter.github.io/X_eTaL-ML/recorded/)
page (`just tour SLUG`: every statement as X_eTaL draws it, then its
result). Every demo has a live page now, and every one also runs at
the command line with the X_eTaL vendored in this repository.

To run them yourself (Rust stable and `just`; nothing else to
install, X_eTaL comes vendored):

```bash
git clone https://github.com/softwarewrighter/X_eTaL-ML
cd X_eTaL-ML
just xetal               # build the vendored X_eTaL (once)
just tour moe-router     # each statement, then its result
just run cnn-digits      # just the results
```

## Libraries

| Library | Alias | What | Status |
| ------- | ----- | ---- | ------ |
| [NN](libs/NN/docs/README.md) | `nn:` | activations, softmax by row, dense layers, argmax, one-hot, loss, accuracy | ready |
| Quant | `qz:` | FP16, INT8 and ternary quantization, storage and error | later |
| Conv | `cv:` | windows, 2-D convolution, pooling | later |
| Attention | `at:` | scaled dot-product attention, causal masks, heads | later |
| Norm | `nm:` | layer norm, RMSNorm, standardizing | later |
| Embed | `em:` | covariance, principal components, cosine similarity | later |
| Sample | `sm:` | temperature, top-k and top-p sampling | later |
| Optim | `op:` | gradients for small models, SGD and momentum | later |
| Net (`.xtlm`) | `net:` | networks as a macro | waiting on X_eTaL |

What the "waiting" items need from X_eTaL is listed in
[`docs/xetal-asks.md`](docs/xetal-asks.md); the roadmap is
[`docs/plan.md`](docs/plan.md).

## Build

Requirements: Rust (stable), [`just`](https://github.com/casey/just),
and a checkout of [X_eTaL](https://github.com/softwarewrighter/X_eTaL)
beside this one (only to refresh the vendored copy).

```bash
just                # list the recipes
just xetal-version  # the vendored X_eTaL (vendor/xetal/VENDORED)
just eval "'+ r_/ 1 2 3"
just gate           # the pre-commit gate
```

X_eTaL is used through a snapshot of a committed X_eTaL revision in
`vendor/xetal/` (`just vendor [REF]` refreshes it); every recipe runs
the CLI built from it into `target/xetal/`, so results do not depend
on whatever `xetal` is on your PATH.

### Layout

- `demos/<slug>/` -- one demo per directory: its `.xtl` programs, a
  README, reg-rs baselines in `reg/`, an offline trainer in `train/`
  when it has learned weights, and its web page in `web/`.
- `libs/<Name>/` -- one library per directory: `src/<Name>.xtl`, its
  reference page `docs/README.md`, `demos/`, and reg-rs `tests/`
  (including `types.rgt`, which pins every export's type).

```bash
just pages                # build the live site into pages/ (committed; a push publishes it)
just serve-pages          # preview it at http://127.0.0.1:8435/X_eTaL-ML/
just demos                # the demos, in catalog order
just run SLUG             # run a demo's program; just show SLUG as a notebook
just tour SLUG            # the notebook paced, long lines clipped (as recorded)
just record [SLUG]        # re-record the CLI-only demos (vhs, ffmpeg, gif2webp)
just libs                 # the libraries and their aliases
export XETAL_PATH="$(just path)"   # use the libraries from your own programs
just test                 # every demo and library baseline
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

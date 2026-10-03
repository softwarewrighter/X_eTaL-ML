# X_eTaL-ML -- Implementation Plan

Machine learning in X_eTaL (the eXperimental Extensible Typed Array
Language, developed in `../X_eTaL`): visual ML demos that run from the
command line and live in the browser, and the ML libraries they are
made of. This repo was split out of `../X_eTaL-demos` (whose ML demos
it takes over) so that each X_eTaL repository answers one question:

| Repo | Answers |
| ---- | ------- |
| X_eTaL | what the language is |
| X_eTaL-demos | what array programming looks like |
| **X_eTaL-ML** | **why array languages fit machine learning** |
| X_eTaL-games | general-purpose, stateful, interactive programming |
| X_eTaL-libraries | language-level extensibility (`.xtl`, `.xtlm`) |
| X_eTaL-extensions | system-level extensibility (native code) |

Sources: `../X_eTaL/docs/research3.txt` (the split and the release
plan: "Are we X_eTaL yet?"), the ML part of
`../X_eTaL-demos/docs/research.txt` (attention, ternary networks, MoE
routing, CNN, embeddings, world models, diffusion), and
`../X_eTaL-libraries/docs/plan.md` (the library layout). All are
archival, not normative; this plan turns them into sagas and steps.
(`docs/research.txt`, named by the user as this repo's source, does
not exist yet; see the open questions at the end.)

Development is driven by agentrail sagas (one active saga in
`.agentrail/`, finished sagas archived to `.agentrail-archive/`), as in
the sibling repos. Every step ends with the gate (`just gate`), docs
updated (`README.md`, `CHANGES.md`, this plan, `docs/xetal-asks.md`,
the demo's or library's pages), a sane `.gitignore`, a detailed commit
to `main` (with the `.agentrail/` changes), `agentrail complete`, and a
push.

## Guiding principle

Most of what an ML framework hides behind objects is array algebra.
Every demo here shows that: a small X_eTaL program, a visible model
(weights, activations, scores, routes), each array transformation
stepped through, and one moment where a reader sees that a layer, an
attention head or a router is one array expression.

The libraries are the same idea made reusable: the expressions the
demos are built from (softmax by row, a dense layer, quantization,
convolution windows, scaled dot-product attention, normalization)
written once, typed, tested and documented, so a new model is a short
program that imports them. That is X_eTaL's first kind of
extensibility (libraries extend the vocabulary) applied to ML; the
second (macros extend the language: `"784 128 relu 10 softmax"
net:n_etwork<`) waits for `.xtlm` in X_eTaL (ask M1).

## Architecture decisions

| # | Decision | Why |
| - | -------- | --- |
| A1 | X_eTaL is **vendored** into `vendor/xetal/` as a source snapshot of a committed ref of `../X_eTaL` (`just vendor [REF]`, default `HEAD`), recorded in `vendor/xetal/VENDORED`. Uncommitted work in `../X_eTaL` is never vendored. Same scripts as the sibling repos. | X_eTaL moves fast; demos and libraries need a recent but stable interpreter, refreshed deliberately, never mid-step. |
| A2 | The vendored CLI builds into `target/xetal/` (`just xetal`); every recipe runs that binary, not one on the PATH. | Baselines and pinned types are tied to `VENDORED`. |
| A3 | **Each demo is its own sub-project**, `demos/<slug>/`, exactly as in X_eTaL-demos: `demo.toml` (title, summary, concepts, status, order, needs), `README.md`, `<slug>.xtl` programs, `reg/` reg-rs baselines (`cli-NAME` per `.xtl`, `browser-SLUG` for its page), optional `test.sh`, `train/` (an offline trainer, std-only Rust, when the demo has learned weights) and `web/` (its own Cargo workspace: a Yew app on the vendored `xetal-play` and `shared/microscope`). A demo never reaches into another demo. | Demos evolve independently; the three taken over from X_eTaL-demos keep their layout and history. |
| A4 | **Each library is its own directory**, `libs/<Name>/`, exactly as in X_eTaL-libraries: `src/<Name>.xtl` (later also `<Name>.xtlm`), `tests/` (reg-rs: `NAME.xtl` + `.rgt/.out/.err`, `types.rgt` pinning the exports' types, `demo-D.rgt` per demo), `docs/README.md` (the reference page), `demos/*.xtl` and a short `README.md`. `scripts/xt` runs the vendored xetal with every `libs/*/src` on `XETAL_PATH`. | One place per library; a library can be lifted out whole (into X_eTaL-libraries, or a user's `userlibs/`). |
| A5 | **Library conventions** are X_eTaL's style guide (lang-choices section 16) as X_eTaL-libraries applies them: `l:` exports, private helpers unprefixed, function-first operands, `?`/`!` suffixes, no top-level expressions, a header with the import line and recommended alias, no export shadowing a built-in, no library named like a standard one. Recommended aliases do not collide with X_eTaL's (`c:` `m:` `s:`) or X_eTaL-libraries' (`k:` `t:` `se:` `n:` `cb:` `q:` `mx:` `r:` `f:` `p:` `d:` `sx:` `g:` `b:` `x:` `test:`). | The libraries teach the style and can be imported beside the general-purpose ones. |
| A6 | **Demos are built from the libraries** once a library exists: a demo's `.xtl` imports `nn:`, `qz:` and the rest instead of defining softmax again. A demo taken over keeps its own definitions until the step that moves it onto a library, and its baselines prove the output unchanged. | The libraries earn their place by being used; the demos stay short. |
| A7 | **Learned weights come from offline trainers** in `demos/<slug>/train/` (std-only Rust, deterministic seeds, `just <slug>-train`) and are written into the demo's `.xtl` as literals. Training in X_eTaL itself (hand-derived gradients for small models) is a library and demo of its own (saga 3), not a requirement. Datasets are fetched into gitignored `work/` (MD5-checked) and never committed. | Demos load instantly and are reproducible; X_eTaL's evaluator speed (asks) is not a training-loop speed. |
| A8 | **Ported, not copied.** A function or model taken from another source (MLPL, microgpt, BitNet, a paper) is reimplemented from its documented behavior and cited in the source and on the page. Code from this author's own MIT repos (X_eTaL-demos, microgpt-mlpl) may be adapted with a note of where it came from. | Credit and lineage without license entanglement. |
| A9 | A missing X_eTaL feature or bug is **not** fixed here nor hidden: it goes in `docs/xetal-asks.md` (status, kind, demos and libraries, why, minimal repro, workaround) and on the demo's or library's page. A demo or library that cannot be built waits in the deferred saga. | X_eTaL owns its language decisions; this repo is a consumer. |
| A10 | **Macro libraries (`.xtlm`) wait for X_eTaL** (its Saga 19; ask M1). They are designed here on paper (saga 4), never emulated. The ML one is `Net.xtlm`: `"784 128 relu 10 softmax" net:n_etwork<` expanding into ordinary, type-checked `nn:` calls, inspectable with `xetal expand`. | research3.txt: the release needs one domain-specific macro, not only control flow. |
| A11 | The **live site** is built locally into `pages/` (`just pages`): a catalog `pages/index.html` from every `demos/*/demo.toml`, `pages/<slug>/` from trunk, and (saga 2) a library reference. `pages/` is committed; `.github/workflows/pages.yml` only uploads it. | Same model as the sibling repos: simple, fast, deterministic deploys. |
| A12 | `just` is the entry point (recipes call `scripts/*.sh`); `CHANGES.md` gets a line for every commit (as in `../X_eTaL`); docs are ASCII-only markdown (`sw-markdown-checker`); user-facing docs say what and how, saga talk lives only here. | Same process as the sibling repos. |

## Layout

```
demos/<slug>/            one ML demo per directory (A3)
  demo.toml README.md *.xtl reg/ train/ web/
demos/_template/         what just new-demo copies
libs/<Name>/             one ML library per directory (A4)
  README.md src/ tests/ docs/README.md demos/
templates/Library/       what just new-lib copies
shared/microscope/       the demo pages' shared Yew shell
tools/vendor-probe/      proves xetal-play builds here, natively and for wasm32
scripts/                 the logic behind the just recipes
vendor/xetal/            the vendored X_eTaL (never edited)
pages/                   the built live site (committed)
```

## The gallery (demos)

| Demo (slug) | Visual payoff | X_eTaL ideas | From | Saga |
| ----------- | ------------- | ------------ | ---- | ---- |
| ternary-net | one classifier in FP32, FP16, INT8 and ternary; a ternary layer as additions | inner product, quantization, masks | X_eTaL-demos (live) | 1 |
| moe-router | tokens routed to their top-2 of 16 experts; nudge a token and watch the experts switch | matrix product, softmax, top-k by masks | X_eTaL-demos (live) | 1 |
| cnn-digits | draw a digit; conv, ReLU, pool, dense, softmax, every stage visible | windows by rotation, reshape, inner product | X_eTaL-demos (weights done, page planned) | 1 (CLI), 3 (page) |
| attention | Q, K, V, S = Q K^T / sqrt d, softmax heatmap, rows meeting columns; a causal mask | matrix product, transpose, softmax | X_eTaL-demos (deferred there: transpose, now landed) | 3 |
| embedding-explorer | 64-dimensional embeddings projected to a rotatable 3-D cloud by PCA | covariance, power iteration, transpose | X_eTaL-demos (deferred there) | 3 |
| gradient-descent | a tiny model trained in X_eTaL itself: the loss surface, the path, the update as one expression | outer product, reduce, iteration | new | 3 |
| micro-gpt | microgpt's forward pass (embedding, RMSNorm, attention, MLP) on weights trained offline; sample names | everything above | microgpt-mlpl (port) | 3 |
| net-macro | a network written as `"..." net:n_etwork<` and its expansion beside it | `.xtlm` macros, `xetal expand` | new | 4 (blocked: M1) |
| world-model | a ball's next frame predicted from the last three | recurrence, prediction | X_eTaL-demos (deferred) | 4 |
| diffusion | noise to image, step by step | tensor transforms, iteration | X_eTaL-demos (deferred) | 4 |

## The libraries

Ranked by how many demos need them. Aliases are recommendations; the
alias is the importer's choice.

| Library | Alias | What | Taken from | Saga |
| ------- | ----- | ---- | ---------- | ---- |
| NN | `nn:` | activations (ReLU, leaky ReLU, sigmoid, tanh), softmax and log-softmax by row, a dense layer, argmax by row, one-hot, cross-entropy, accuracy | ternary-net, moe-router, cnn-digits | 1 |
| Quant | `qz:` | FP16 rounding, INT8 symmetric quantize and dequantize, ternary (BitNet b1.58 absmean) weights, storage bits, quantization error | ternary-net | 1 |
| Conv | `cv:` | windows by rotation, 2-D convolution of several filters, max- and average-pooling by reshape, padding | cnn-digits, X_eTaL-demos image-pipeline | 2 |
| Attention | `at:` | scaled dot-product attention, causal mask, masked softmax, heads split and merged | research (attention) | 2 |
| Norm | `nm:` | layer norm, RMSNorm, standardize columns, batch statistics | microgpt | 2 |
| Embed | `em:` | centering, covariance, power iteration for the top components, PCA projection, cosine similarity, nearest neighbours | research (embedding explorer) | 2 |
| Sample | `sm:` | temperature, top-k and top-p filtering, sampling from a distribution with `r_oll` | microgpt | 2 |
| Optim | `op:` | hand-derived gradients for linear and logistic regression and a dense layer, SGD and momentum steps, a training loop by `p_ower` | new | 3 |
| Net (`.xtlm`) | `net:` | `"784 128 relu 10 softmax" net:n_etwork<` -> a forward function built from `nn:` calls | research3.txt | 4 (blocked: M1) |

## Saga 1 -- foundation  [ACTIVE]

Goal: the process, the vendored interpreter, both layouts (demos and
libraries), the three ML demos taken over from X_eTaL-demos and live,
and the first two libraries.

| # | Step slug | Delivers |
| - | --------- | -------- |
| 1 | scaffold | DONE. agentrail saga; CLAUDE.md (AGENTS.md a symlink); README (value proposition, the ecosystem, gallery, libraries, build, status, copyright, license); COPYRIGHT; LICENSE; CHANGES.md; .gitignore; justfile; the gate (markdown); this plan; docs/xetal-asks.md |
| 2 | vendor-xetal | DONE: vendored abb8274 (its own commit); scripts taken from X_eTaL-demos with X_eTaL-libraries' additions (XETAL_BUILD_SHA so `xetal --version` names the vendored commit; check-vendor imports Stats); `tools/vendor-probe`; recipes vendor, xetal, xetal-version, eval, check-vendor. Planned: `just vendor [REF]`, `vendor/xetal/VENDORED`, `just xetal`, `xetal-version`, `eval`; `tools/vendor-probe` (xetal-play natively and for wasm32); `scripts/check-vendor.sh` in the gate; the snapshot in its own commit |
| 3 | layout | DONE: both tool sets taken from the sibling repos (demo scripts and `build-catalog.py`, which the demo self-test uses; library scripts and `check-examples.py`); `demos/_template`, `templates/Library` (provenance "Written for X_eTaL-ML"); `shared/microscope` (links now name this repo); library recipes renamed where they would clash with the demos' (`run-lib`, `demo-lib`, `show-lib`, `bless-lib`); `just eval` puts the libraries on `XETAL_PATH`; the gate runs both self-tests, the shell's tests and wasm32 check, both test runners and the page examples. Planned: the demo sub-project layout (`demos/_template`, `scripts/demos.py`, `test-demos.sh` with reg-rs, `run-demo.sh`, `new-demo.sh`, self-test) and the library layout (`templates/Library`, `scripts/xt`, `libs.py`, `test-libs.sh`, `run-lib.sh`, `new-lib.sh`, self-test); `shared/microscope` taken over; recipes |
| 4 | port-demos | DONE: the three demos copied (web apps, trainers, reg-rs CLI baselines, screenshots; browser baselines wait for step 5); every baseline and web test passes unchanged on abb8274; both trainers reproduce the committed weights byte for byte (`just ternary-train`, `just cnn-train`, `scripts/mnist.sh`); asks re-run: M7 and M8 landed (moe-router now passes Int word numbers), M3 regressed (about 700 ns per multiply-add), M4, M5, M9 open; X_eTaL-demos' handoff (`docs/xetal-ml-asks.md` there) read. Planned: ternary-net, moe-router and cnn-digits (CLI) copied from X_eTaL-demos with their trainers (`just ternary-train`, `just cnn-train`, `scripts/mnist.sh`); baselines re-run on the new vendor (differences explained); each demo's asks re-checked against it (landed ones marked) |
| 5 | pages | the live site: catalog from `demo.toml`, trunk per demo, browser checks in headless Chrome, `.github/workflows/pages.yml` (upload only), screenshots; GitHub Pages enabled; README links |
| 6 | nn | the NN library (`nn:`): tests, pinned types, page, a demo; moe-router and cnn-digits moved onto it, baselines unchanged |
| 7 | quant | the Quant library (`qz:`): tests, pinned types, page, a demo; ternary-net moved onto it, baselines unchanged |

## Saga 2 -- building blocks

| # | Step slug | Delivers |
| - | --------- | -------- |
| 1 | conv | Conv (`cv:`); cnn-digits moved onto it |
| 2 | attention-lib | Attention (`at:`) |
| 3 | norm | Norm (`nm:`) |
| 4 | embed | Embed (`em:`) |
| 5 | sample | Sample (`sm:`) |
| 6 | lib-site | the libraries in the live site: a page per library (reference, demos runnable in the browser, types), as X_eTaL-libraries' site |
| 7 | release-1 | catalog, docs, asks reviewed against a fresh vendor, retrospective |

## Saga 3 -- the ML gallery

| # | Step slug | Delivers |
| - | --------- | -------- |
| 1 | cnn-digits-page | draw a 28 x 28 digit; conv -> ReLU -> pool -> dense -> softmax in X_eTaL; click a conv output for patch x kernel = value |
| 2 | attention | the attention microscope: Q, K, V, scores, softmax heatmap, output; a causal mask; heads |
| 3 | embedding-explorer | PCA from 64 dimensions to a rotatable 3-D cloud, every stage inspectable |
| 4 | optim | Optim (`op:`) |
| 5 | gradient-descent | training in X_eTaL: loss surface, path, update |
| 6 | micro-gpt | microgpt inference in X_eTaL on offline-trained weights (port from microgpt-mlpl) |
| 7 | release-2 | catalog, docs, retrospective |

## Saga 4 -- macros and deferred (blocked)

Started when a vendored X_eTaL supports `.xtlm` (ask M1) or the other
asks below land; until then only the designs are kept current.

| # | Step slug | Delivers | Waits on |
| - | --------- | -------- | -------- |
| 1 | macro-survey | refresh the vendor; read what X_eTaL implemented (its Saga 19); confirm A10 and the Net design | M1 |
| 2 | net-macro | `libs/Net/src/Net.xtlm`: `m:n_etwork<` turning a layer list into a forward function of `nn:` calls; tests of the expansion and the result; the net-macro demo showing source and expansion side by side | M1, M2 |
| 3 | world-model | the tiny world model | training speed, maybe records |
| 4 | diffusion | the diffusion panels | a learned denoiser, speed |
| 5 | release-3 | catalog, docs, retrospective | |

### Net.xtlm design sketch

```
"nn:" u_se< "NN"
"net:" u_se< "Net"
f := "784 128 relu 10 softmax" net:n_etwork< "W1 b1 W2 b2"
```

expands (visible with `xetal expand`) to an ordinary definition:

```
f := { x -> nn:s_oftmax (nn:r_elu (x nn:d_ense W1 b1)) nn:d_ense W2 b2 }
```

so the macro adds notation, never semantics: the expansion is
type-checked like any code. The exact shape of a dyadic macro call
and of a definition it produces follows X_eTaL's decisions MC10 to
MC13 as implemented.

## Cross-cutting

- Refresh the vendored X_eTaL (`just vendor`) at the start of a saga,
  or when an ask in `docs/xetal-asks.md` has landed upstream; never
  in the middle of a step. The refresh is its own commit, with the
  baselines re-run.
- When an ask lands, remove the workaround in the step that refreshes
  the vendor, and mark the ask landed.
- The split: once ternary-net, moe-router and cnn-digits are pushed
  and live here (saga 1 steps 4 and 5), the user deletes them from
  X_eTaL-demos (the user's decision, 2026-10-03); X_eTaL-demos then
  links here.

## Open questions

- `docs/research.txt`: the user named it as this repo's source, but
  the repo had only an empty README. Until it is added, research3.txt
  and X_eTaL-demos' research.txt stand in. If one is added, this plan
  is reviewed against it.

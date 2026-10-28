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
`../X_eTaL-libraries/docs/plan.md` (the library layout), and
`../X_eTaL/docs/research4.txt` (the launch priorities). All are
archival, not normative; this plan turns them into sagas and steps.

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
| A1 | X_eTaL is **pinned, not tracked** (the user's decision, 2026-10-04, after `../X_eTaL/docs/vendoring.md`; saga 1 tracked a snapshot in `vendor/xetal/`): `XETAL_COMMIT` holds the known-good commit; `just xetal` (scripts/xetal.sh, as X_eTaL-demos') clones X_eTaL into the gitignored `work/xetal/`, checks that commit out, builds it and links `bin/xetal`; `just xetal-pin [REF]` moves the pin to a committed ref of `../X_eTaL`. | X_eTaL moves fast; this repo needs a recent but stable interpreter, changed deliberately, never mid-step, without carrying a copy of its source. |
| A2 | The pinned CLI builds into `target/xetal/` and is reached as `bin/xetal`; every recipe runs that binary, not one on the PATH. The web apps name X_eTaL's crates by path inside `work/xetal/`. | Baselines and pinned types are tied to `XETAL_COMMIT`; the binary reports that commit itself. |
| A3 | **Each demo is its own sub-project**, `demos/<slug>/`, exactly as in X_eTaL-demos: `demo.toml` (title, summary, concepts, status, order, needs), `README.md`, `<slug>.xtl` programs, `reg/` reg-rs baselines (`cli-NAME` per `.xtl`, `browser-SLUG` for its page), optional `test.sh`, `train/` (an offline trainer, std-only Rust, when the demo has learned weights) and `web/` (its own Cargo workspace: a Yew app on the vendored `xetal-play` and `shared/microscope`). A demo never reaches into another demo. | Demos evolve independently; the three taken over from X_eTaL-demos keep their layout and history. |
| A4 | **Each library is its own directory**, `libs/<Name>/`, exactly as in X_eTaL-libraries: `src/<Name>.xtl` (later also `<Name>.xtlm`), `tests/` (reg-rs: `NAME.xtl` + `.rgt/.out/.err`, `types.rgt` pinning the exports' types, `demo-D.rgt` per demo), `docs/README.md` (the reference page), `demos/*.xtl` and a short `README.md`. `scripts/xt` runs the vendored xetal with every `libs/*/src` on `XETAL_PATH`. | One place per library; a library can be lifted out whole (into X_eTaL-libraries, or a user's `userlibs/`). |
| A5 | **Library conventions** are X_eTaL's style guide (lang-choices section 16) as X_eTaL-libraries applies them: `l:` exports, private helpers unprefixed, function-first operands, `?`/`!` suffixes, no top-level expressions, a header with the import line and recommended alias, no export shadowing a built-in, no library named like a standard one. Recommended aliases do not collide with X_eTaL's (`c:` `m:` `s:`) or X_eTaL-libraries' (`k:` `t:` `se:` `n:` `cb:` `q:` `mx:` `r:` `f:` `p:` `d:` `sx:` `g:` `b:` `x:` `test:`). | The libraries teach the style and can be imported beside the general-purpose ones. |
| A6 | **Demos are built from the libraries** once a library exists: a demo's `.xtl` imports `nn:`, `qz:` and the rest instead of defining softmax again (the command line puts `libs/*/src` on `XETAL_PATH`; pages get them from an in-memory store, `microscope::libs`). A demo taken over keeps its own definitions until the step that moves it onto a library, and its baselines prove the output unchanged. Exception: a microscope page whose lesson is a definition it shows and highlights (moe-router's softmax and top-2) keeps that definition in its program. | The libraries earn their place by being used; the demos stay short; the pages keep teaching. |
| A7 | **Learned weights come from offline trainers** in `demos/<slug>/train/` (std-only Rust, deterministic seeds, `just <slug>-train`) and are written into the demo's `.xtl` as literals. Training in X_eTaL itself (hand-derived gradients for small models) is a library and demo of its own (saga 3), not a requirement. Datasets are fetched into gitignored `work/` (MD5-checked) and never committed. | Demos load instantly and are reproducible; X_eTaL's evaluator speed (asks) is not a training-loop speed. |
| A8 | **Ported, not copied.** A function or model taken from another source (MLPL, microgpt, BitNet, a paper) is reimplemented from its documented behavior and cited in the source and on the page. Code from this author's own MIT repos (X_eTaL-demos, microgpt-mlpl) may be adapted with a note of where it came from. | Credit and lineage without license entanglement. |
| A9 | A missing X_eTaL feature or bug is **not** fixed here nor hidden: it goes in `docs/xetal-asks.md` (status, kind, demos and libraries, why, minimal repro, workaround) and on the demo's or library's page. A demo or library that cannot be built waits in the deferred saga. | X_eTaL owns its language decisions; this repo is a consumer. |
| A10 | **Macro libraries (`.xtlm`)** are written against X_eTaL v0.1.0, which has them (asks M1, M2 landed; sagas 1 and 2 only designed them). The ML one is `Net.xtlm` (saga 3): `"784 128 relu 10 softmax" net:n_etwork< "w1 w2"` becomes an ordinary, type-checked lambda of `nn:` calls, shown by `xetal expand`; its tests pin each program's expansion. Because a macro's text cannot name the importer's alias (ask M11), the expansion says `nn:`. | research3.txt: the release needs one domain-specific macro, not only control flow. |
| A11 | The **live site** is built locally into `pages/` (`just pages`, and by the gate): a catalog `pages/index.html` from every `demos/*/demo.toml` and `pages/<slug>/` from trunk. `pages/` is **not tracked on main** (the user's decision, 2026-10-05, as X_eTaL-games; sagas 1 and 2 committed it and a workflow uploaded it): `just publish` makes it the only commit of the `gh-pages` branch, replaced on every publish, which GitHub Pages serves. | Built files (four WebAssembly binaries per build) never accumulate in history; the published site always names the commit of main it was built from. |
| A12 | `just` is the entry point (recipes call `scripts/*.sh`); `CHANGES.md` gets a line for every commit (as in `../X_eTaL`); docs are ASCII-only markdown (`sw-markdown-checker`); user-facing docs say what and how, saga talk lives only here. | Same process as the sibling repos. |
| A13 | **Live or recorded** (the user's decision, 2026-10-03): a demo that runs interactively in the browser is shown live (its page); a demo that runs only at the command line is shown recorded (a VHS tape, `just record`, an animated WebP on the recorded page, its README and its catalog card). When a CLI-only demo gets its page, its tape and recording go. Every demo runs at the command line (`just run`, `just tour`). | One way to see each demo, no duplicated upkeep. |
| A14 | **Helpful libraries from X_eTaL-libraries are used, pinned the same way** (the user's decision, 2026-10-03): `XETAL_LIBRARIES_COMMIT`, cloned into `work/xetal-libraries/` by scripts/xetal-libraries.sh (run by `just xetal`), each library used linked under `work/libs/`, on `XETAL_PATH` beside `libs/` and in the pages' store; `just libs-pin [REF]`. Now: Check, for the tests; Plot (with Strings, Format, Lists) for the pages' training curves (2026-10-07). | Reuse instead of re-writing; the same discipline as X_eTaL itself. |
| A15 | **Programs are short; data lives in files** (the user's review, 2026-10-03: "APL is known for being compact... 100s of repeated lines is not good"): weights, samples and other data are files in `demos/<slug>/data/` (plain numbers, readable rows, written by the demo's trainer from a downloaded, uncommitted dataset), read with `n_umbers []N_GET`; the pages put the same files in their in-memory store. A program never carries walls of literals or repeated lines (`e_ach`, not ten copies of a call), and says things the short way the language offers: number literals side by side are an array (`8 13 2 13 2 r_eshape x`, not a `c_at` chain: the user's review, 2026-10-05), one reduce takes several axes (`'m_ax r_/_35`); the gate refuses literal `c_at` chains. | The demos must show what the language is good at: a network in a handful of dense lines. |
| A16 | **A page shows all the code it runs** (the user's review, 2026-10-04: the live source was a subset, "e.g. m is not assigned"; "show explicit code that was omitted, the CLI and web code are not the same"): the source panel is the head the page's programs share, then each run as run, under a comment naming it; an elided value (a large array passed between runs) is written `...` with a comment saying what it is. A test checks that every line of every program the page runs is shown. | What a reader sees is what ran. |

## Layout

```
demos/<slug>/            one ML demo per directory (A3)
  demo.toml README.md *.xtl reg/ train/ web/
demos/_template/         what just new-demo copies
libs/<Name>/             one ML library per directory (A4)
  README.md src/ tests/ docs/README.md demos/
templates/Library/       what just new-lib copies
shared/microscope/       the demo pages' shared Yew shell
tools/xetal-probe/       proves xetal-play builds here, natively and for wasm32
scripts/                 the logic behind the just recipes
XETAL_COMMIT             the X_eTaL commit this repo is known to work with
work/xetal/              its clone (gitignored; just xetal), never edited
pages/                   the built live site (not tracked; just publish)
```

## The gallery (demos)

| Demo (slug) | Visual payoff | X_eTaL ideas | From | Saga |
| ----------- | ------------- | ------------ | ---- | ---- |
| ternary-net | one classifier in FP32, FP16, INT8 and ternary; a ternary layer as additions | inner product, quantization, masks | X_eTaL-demos (live) | 1 |
| moe-router | tokens routed to their top-2 of 16 experts; nudge a token and watch the experts switch | matrix product, softmax, top-k by masks | X_eTaL-demos (live) | 1 |
| cnn-digits | draw a digit; conv, ReLU, pool, dense, softmax, every stage visible | windows by rotation, reshape, inner product | X_eTaL-demos (weights done, page planned) | 1 (CLI), 2 (page) |
| attention | Q, K, V, S = Q K^T / sqrt d, softmax heatmap, rows meeting columns; a causal mask | matrix product, transpose, softmax | X_eTaL-demos (deferred there: transpose, now landed) | 2 |
| embedding-explorer | 64-dimensional embeddings projected to a rotatable 3-D cloud by PCA | covariance, power iteration, transpose | X_eTaL-demos (deferred there) | 5 |
| gradient-descent | a tiny model trained in X_eTaL itself: the loss surface, the path, the update as one expression | outer product, reduce, iteration | new | 5 |
| micro-gpt | microgpt's forward pass (embedding, RMSNorm, attention, MLP) on weights trained offline; sample names | everything above | microgpt-mlpl (port) | 5 |
| net-macro | a network written as `"..." net:n_etwork<` and its expansion beside it; three networks on a spiral; any spec typed | `.xtlm` macros, `xetal expand` | new | 3 (live) |
| world-model | a ball's next frame predicted from the last three; all 1,936 states as one batch; dreaming on its own predictions | a batch through dense layers, `p_ower` over a tuple | X_eTaL-demos (deferred there) | 5 (live) |
| diffusion | noise to shapes in ten DDIM steps; a learned denoiser; the same noise, the same pictures | a batch through dense layers, `p_ower` over a tuple | X_eTaL-demos (deferred there) | 5 (live) |

## The libraries

Ranked by how many demos need them. Aliases are recommendations; the
alias is the importer's choice.

| Library | Alias | What | Taken from | Saga |
| ------- | ----- | ---- | ---------- | ---- |
| NN | `nn:` | softmax and log-softmax by row (its activations moved to Activations, row 28), a dense layer, a bias along a named axis (`a_ddAxis`, `b_ias`: no implicit broadcasting), argmax by row, one-hot, cross-entropy, accuracy | ternary-net, moe-router, cnn-digits | 1 |
| Quant | `qz:` | FP16 rounding, INT8 symmetric quantize and dequantize, ternary (BitNet b1.58 absmean) weights, storage bits, quantization error | ternary-net | 5 |
| Conv | `cv:` | windows by rotation, 2-D convolution of several filters, max- and average-pooling by reshape, padding | cnn-digits, X_eTaL-demos image-pipeline | 5 |
| Attention | `at:` | scaled dot-product attention, causal mask, masked softmax, heads split and merged | research (attention) | 5 (the attention demo keeps its definitions) |
| Norm | `nm:` | layer norm, RMSNorm, standardize columns, batch statistics | microgpt | 5 |
| Embed | `em:` | centering, covariance, power iteration for the top components, PCA projection, cosine similarity, nearest neighbors | research (embedding explorer) | 5 |
| Sample | `sm:` | temperature, top-k and top-p filtering, sampling from a distribution with `r_oll` | microgpt | 5 |
| Optim | `op:` | hand-derived gradients for linear and logistic regression and a dense layer, SGD and momentum steps, a training loop by `p_ower` | new | 5 |
| Net (`.xtlm`) | `net:` | `"784 128 relu 10 softmax" net:n_etwork<` -> a forward function built from `nn:` calls; `net:p_arams<`, `net:s_hapes<` | research3.txt | 3 |
| Flow (`.xtlm`) | `flow:` | `@ flow:p_ipe< c"X |> it nn:d_ense W1 |> nn:r_elu"` -> a block of one line per stage (X_eTaL D155, `docs/pipeline-macro.md`): function stages applied, `it` stages, `$name` intermediates and `$(Q, K, V)` tuples, ` [N O]` shape contracts checked at run time; `flow:t_ext<` the pipeline read back | X_eTaL's spatial design, built here | 5 |
| Integrals (`.xtlm`) | `integ:` | `@ integ:i_ntegral< "integral tau:=a..b by trapezoid 64 \| body"` -> the value by the rule (X_eTaL Saga 42 decisions 2 to 4: integral read like sum, option words with a default, improper refused); `integ:t_ex<` the LaTeX; lift-able into X_eTaL-libraries' Equations (its decision 1) | X_eTaL's Saga 42 plan, built here | 5 |
| Activations | `act:` | built (row 27; rows 28 to 31 move the callers onto it, draw and compare): the item-wise activations, ReLU, leaky ReLU, sigmoid and tanh out of NN (NN keeps softmax, log-softmax, argmax: along an axis, and what Attention, Sample and Learn use), with PReLU (a slope per column), ELU, SELU, GELU (the tanh form; `g_eluErf` exact, a private erf until X_eTaL's `e_rf`, M19), SiLU and `b s_wish` (one function, Swish-1 is SiLU), Mish, the stable softplus, hard sigmoid and hard swish; the GLU family as gated layers (`(W, V) act:s_wiglu x`: GLU, ReGLU, GEGLU, SwiGLU); a derivative per function in terms of the input (`act:d_gelu z`: X_eTaL has no automatic differentiation; relu, tanh and sigmoid's slopes are written at the output today, which the smooth ones cannot be), each checked against central differences; the reference page with each function's formula, LaTeX, derivative, cost (transcendental calls per element) and where it is used | the papers (Hendrycks and Gimpel 2016; Ramachandran et al. 2017; Misra 2019; Shazeer 2020; Klambauer et al. 2017; Howard et al. 2019) | 5 |

## Saga 1 -- foundation  [DONE, archived]

Goal: the process, the vendored interpreter, both layouts (demos and
libraries), the three ML demos taken over from X_eTaL-demos and live,
and the first two libraries.

| # | Step slug | Delivers |
| - | --------- | -------- |
| 1 | scaffold | DONE. agentrail saga; CLAUDE.md (AGENTS.md a symlink); README (value proposition, the ecosystem, gallery, libraries, build, status, copyright, license); COPYRIGHT; LICENSE; CHANGES.md; .gitignore; justfile; the gate (markdown); this plan; docs/xetal-asks.md |
| 2 | vendor-xetal | DONE: vendored abb8274 (its own commit); scripts taken from X_eTaL-demos with X_eTaL-libraries' additions (XETAL_BUILD_SHA so `xetal --version` names the vendored commit; check-vendor imports Stats); `tools/vendor-probe`; recipes vendor, xetal, xetal-version, eval, check-vendor. Planned: `just vendor [REF]`, `vendor/xetal/VENDORED`, `just xetal`, `xetal-version`, `eval`; `tools/vendor-probe` (xetal-play natively and for wasm32); `scripts/check-vendor.sh` in the gate; the snapshot in its own commit |
| 3 | layout | DONE: both tool sets taken from the sibling repos (demo scripts and `build-catalog.py`, which the demo self-test uses; library scripts and `check-examples.py`); `demos/_template`, `templates/Library` (provenance "Written for X_eTaL-ML"); `shared/microscope` (links now name this repo); library recipes renamed where they would clash with the demos' (`run-lib`, `demo-lib`, `show-lib`, `bless-lib`); `just eval` puts the libraries on `XETAL_PATH`; the gate runs both self-tests, the shell's tests and wasm32 check, both test runners and the page examples. Planned: the demo sub-project layout (`demos/_template`, `scripts/demos.py`, `test-demos.sh` with reg-rs, `run-demo.sh`, `new-demo.sh`, self-test) and the library layout (`templates/Library`, `scripts/xt`, `libs.py`, `test-libs.sh`, `run-lib.sh`, `new-lib.sh`, self-test); `shared/microscope` taken over; recipes |
| 4 | port-demos | DONE: the three demos copied (web apps, trainers, reg-rs CLI baselines, screenshots; browser baselines wait for step 5); every baseline and web test passes unchanged on abb8274; both trainers reproduce the committed weights byte for byte (`just ternary-train`, `just cnn-train`, `scripts/mnist.sh`); asks re-run: M7 and M8 landed (moe-router now passes Int word numbers), M3 regressed (about 700 ns per multiply-add), M4, M5, M9 open; X_eTaL-demos' handoff (`docs/xetal-ml-asks.md` there) read. Planned: ternary-net, moe-router and cnn-digits (CLI) copied from X_eTaL-demos with their trainers (`just ternary-train`, `just cnn-train`, `scripts/mnist.sh`); baselines re-run on the new vendor (differences explained); each demo's asks re-checked against it (landed ones marked) |
| 5 | pages | DONE: the pages tooling from X_eTaL-demos (base `/X_eTaL-ML/`, serve-pages on 8098, screenshots on 8099), the catalog retitled ML with links to the sibling sites; ternary-net and moe-router built and passing their browser baselines in headless Chrome (cnn-digits has no page yet: CLI only, its card says In progress); GitHub Pages enabled (build type workflow) at https://softwarewrighter.github.io/X_eTaL-ML/; README live links. Planned: the live site: catalog from `demo.toml`, trunk per demo, browser checks in headless Chrome, `.github/workflows/pages.yml` (upload only), screenshots; GitHub Pages enabled; README links |
| 6 | nn | DONE: `libs/NN` with 12 exports (r_elu, l_eaky, s_igmoid, t_anh, s_oftmax, l_ogSoftmax, d_ense, a_rgmax, o_neHot, c_rossEntropy, m_se, a_ccuracy); softmax, log-softmax and argmax work along the last axis of any rank; tests (basics, 16 property checks), pinned types, page, the XOR demo; cnn-digits moved onto it (only its guesses line changed: Ints, not Floats); moe-router keeps its own definitions (A6 refined); demos run with `libs/` on XETAL_PATH and pages get the libraries from an in-memory store (`microscope::libs`). Planned: the NN library (`nn:`): tests, pinned types, page, a demo; moe-router and cnn-digits moved onto it, baselines unchanged |
| 7 | recorded-demos | DONE: `demos/<slug>/<slug>.tape` for all three; `just tour SLUG` (scripts/tour.py: the notebook paced, lines clipped, data runs collapsed, 627 lines to 97 for cnn-digits); `just record` (scripts/record.sh: vhs, then ffmpeg to 8 fps and 960 px, then gif2webp lossy q50: about 2 MB each; `--encode` skips recording); `pages/recorded/` (how to run them in a clone, each recording with links); catalog cards link Recorded, and a demo without a live page (cnn-digits) shows its recording as the card picture; READMEs show them; the favicon blue (the user's request: distinct from X_eTaL-demos' lavender). Planned: inserted at the user's request: a VHS tape per demo rendered to an animated WebP (`just record`), `just tour SLUG` (the notebook paced, long lines clipped, data runs collapsed), a "Recorded CLI demos" page with how to run them in a clone |
| 8 | quant | DEFERRED post-launch (research4.txt: no new breadth before the launch); blocked in the saga with that reason, moved to saga 4 |

## Reprioritized for the launch (2026-10-03, research4.txt)

`../X_eTaL/docs/research4.txt` (an audit of all six repos) says the
ecosystem has enough breadth; what is left before promoting it is to
stabilize, synchronize, explain and give people one path through it.
For this repo that means:

- **Stop adding breadth.** No new library before the launch: NN is
  enough to show the idea; Quant, Conv, Attention, Norm, Embed,
  Sample and Optim wait (saga 4). No MicroGPT, embedding explorer,
  gradient descent, world model or diffusion before the launch.
- **Strongly desirable:** the CNN live page and the attention demo
  (saga 2). The attention demo keeps its definitions in its program,
  as moe-router does (A6), so it needs no new library.
- **Macros right after them:** the network macro (`Net.xtlm`) is one
  of three downstream consumers that make `.xtlm` convincing (with
  X_eTaL-libraries' Control.xtlm and X_eTaL-extensions' binding
  macros), so it is saga 3, ahead of all post-launch work, the day
  `.xtlm` lands. Asked upstream: implement `.xtlm` before the
  broader course (research4's order).
- **The table / inner regression (M3)** is upstream's Saga 30; this
  repo supplies the ML measurements (`just bench`) and re-checks them
  when a fixed X_eTaL is vendored.
- **One path for a newcomer:** the catalog's front page says what
  each demo is, why it is an array expression and shows the X_eTaL;
  the recorded CLI demos and the run-it-yourself instructions sit
  beside the live pages; the ecosystem front door (upstream) links
  here.
- **Promotion blockers** this repo hits (M9 bound Bool arithmetic,
  M3 speed) are listed for the cross-repo audit, and a known-good
  X_eTaL commit is recorded for the six-repo release tag.

## Saga 2 -- launch  [DONE but release-1, parked: the launch is delayed; archived]

| # | Step slug | Delivers |
| - | --------- | -------- |
| 0 | live-or-recorded | inserted: A13 applied (ternary-net's and moe-router's recordings and tapes removed; cnn-digits keeps its own until its page), A14 (Check vendored from X_eTaL-libraries fd93e45; NN's checks use it), research.txt question settled |
| 0b | cnn-digits-cli | inserted after the user's review ("the demo looks bad"): A15 for cnn-digits (595 lines with 533 `c_at` continuation lines -> 64; data/filters.txt, dense.txt, samples.txt, expected.txt written by the trainer); output that shows the network (digit, filters, feature maps, pooled maps, probability bars, the ten digits read and how sure); a final check that X_eTaL equals the trainer's Rust pass; `just tour` streams; re-recorded |
| 0d | american-spelling | DONE: inserted at the user's request (as in X_eTaL-demos): `scripts/check-spelling.py` with its self-test in the gate; the audit's 47 British forms fixed (of color, center, labeled, modeled, specialize, neighbors); the shared shell's color module is now `color`; CLAUDE.md rule 10 |
| 0c | ternary-net-data | DONE: A15 for ternary-net (38 continuation lines -> data/fp.txt, qa.txt, test.txt from `just ternary-train`; output identical) and moe-router (the 37 x 8 feature literal -> data/features.txt; output identical); pages given the files (`microscope::libs::add`); and, after the user's review ("the live demo's source is incomplete"), A16 applied to both pages |
| 1 | cnn-digits-page | DONE: `demos/cnn-digits/web` live: draw (strokes interpolated) or pick a digit; X_eTaL (the program's own head, data from the store) runs every stage in about 100-150 ms in the browser; filters, 8 feature maps (click for the patch x filter arithmetic, checked against X_eTaL's value), 8 pooled maps, probability bars; the program panel shows all it runs (A16). Tests: every stage equals a direct Rust computation for all ten samples, the trainer's probabilities, the patch arithmetic, strokes unbroken, every line shown; browser baseline; screenshot; its tape and recording removed (A13). Verified drawing in Chrome. Planned: draw a 28 x 28 digit (or pick one of ten); conv -> ReLU -> pool -> dense -> softmax in X_eTaL, every stage shown with its shape; click a conv output for patch x kernel = value; the X_eTaL pass equals the trainer's Rust pass; browser baseline, screenshot, recording |
| 2 | attention | DONE: `demos/attention` live: the vocabulary, features and one hand-set head in data/; attention in two lines (`u:s_cores`, `u:a_ttend`), the transpose `o_\`, a causal mask; the CLI prints heatmaps, what each word looks at, an output, the causal map and the "wide" sentence; the page (sentence, presets, mask switch, weights and scores as word tables, a word's query against every key, its output, all it runs). Tests: scores, weights and outputs equal the direct algebra; tired and it find the animal, wide the street; the causal mask; tokens; every line shown. Planned: the attention microscope: X, Q, K, V, S = Q K^T / sqrt d, softmax heatmap, Y = A V for a short sentence; a causal mask; definitions in the program, each line inspectable; CLI baseline, page, browser baseline, recording |
| 3 | bench | DONE: `bench/` (dense, inner-wide, conv, softmax, table-spread) and every demo's program; `just bench` (table), `just bench-check` (SLOWER over 15% and 15 ms, faster listed), `just bench-bless` (per host, `bench/baseline/max.tsv`), as X_eTaL's own; docs/speed.md: about 770 ns per multiply-add in `i_nner`, 620 ns per `t_able` cell at abb8274; ask M3 points at it. Planned: `just bench`: the ML workloads timed (a 1024 x 16 by 16 x 16 `i_nner`, `t_able` spreads, each demo's run), best of three, the abb8274 baseline in `docs/speed.md`; a warning on a regression over 15 percent, to verify upstream's Saga 30 fix |
| 4 | start-here | DONE: the catalog opens with what you are looking at, why an array language, where to start (the Tiny CNN first; links to the five sibling sites, all answering); each card shows the demo's key line drawn by the vendored xetal (`render --html`, `line` and `why` in demo.toml, self-tested) ; two columns; the run-it-yourself page retitled; README: the four demos with their lines, run them yourself, NN, a status table (today / waiting / later), the later rows gone from the tables; screenshots retaken. The ecosystem front door upstream does not exist yet: linked when it does. Planned: the catalog's front: what each demo shows, why it is one array expression, its X_eTaL; recorded and live demos side by side; how to run them in a clone; README trimmed to what works today, with a status table; a link to the ecosystem front door |
| 4b | pinned-xetal | DONE: inserted at the user's request: A1, A2 and A14 as above; `vendor/` removed (548 files of X_eTaL source and the Check snapshot no longer tracked); scripts/xetal.sh, xetal-pin.sh, xetal-libraries.sh, check-xetal.sh; tools/xetal-probe; every path and doc; same commits as before (abb8274, fd93e45), every baseline unchanged |
| 4c | publish-and-purge | DONE (the site live from gh-pages; history 12.2 MB to 2.1 MB, tree identical, saga references mapped, a fresh clone verified): inserted, the user's decision (as X_eTaL-games): A11 as above (scripts/publish-pages.sh, `just publish`, the workflow removed, the gate builds the site); then, with the user's explicit approval of a force-push before launch, history rewritten without `pages/`, `vendor/` and the deleted `recording.webp` files (a backup bundle first; the saga records' commit references mapped) |
| 5 | asks-audit | DONE: pinned X_eTaL v0.1.0 (512b3ee, 163 commits on from abb8274; its own commit); every baseline unchanged; the ML workloads 4 to 6 times faster (`just bench-check`; new baseline, docs/speed.md); asks re-run: M1 `.xtlm`, M2 `xetal expand`, M3 speed and M9 bound Bool landed, M4 grade per row and M5 arrays in and out still open, M10 (`e_ach` returning arrays) added; no workaround was left to remove (ternary-net and cnn-digits READMEs no longer call the matrix product slow); promotion blockers for the cross-repo audit: none from this repo (M4, M5, M10 are conveniences); tagged v0.1.1, built against X_eTaL v0.1.0. Planned: move the pin (after upstream's Saga 30 and terminal work), re-run every ask's repro, mark landed asks and remove their workarounds, list this repo's promotion blockers for the cross-repo audit |
| 6 | release-1 | catalog, docs, retrospective; the X_eTaL commit this release is known to work with, for the six-repo tag |

## Saga 3 -- macros  [ACTIVE]

`.xtlm` macro libraries and `xetal expand` are in the X_eTaL this
repo pins since 2026-10-05. The launch being delayed (the user,
2026-10-05), this saga goes ahead of release-1; it starts with a
survey of what X_eTaL implemented.

| # | Step slug | Delivers | Waits on |
| - | --------- | -------- | -------- |
| 1 | macro-survey | DONE: X_eTaL v0.1.0's macros read and tried (MC10 to MC30, `lib/Macros.xtlm`, `Combinators.xtlm`, the hooks, hygiene, `xetal expand`); finding: a macro's text cannot name the importer's alias (ask M11, X_eTaL-libraries' X14), so Net's expansion says `nn:`; A10 and the sketch revised; the library tooling accepts `.xtlm` (macro types pinned, every test's and demo's expansion a baseline, `just expand-lib`) | |
| 2 | net-macro | DONE: `libs/Net/src/Net.xtlm`: `net:n_etwork<`, `net:p_arams<`, `net:s_hapes<`; bad specs rejected at the call (`[]R_EJECT`); tests: basics, 6 checks against hand-written networks (relu, sigmoid, tanh), three rejections, every expansion pinned; page; the XOR demo | |
| 3 | net-demo | DONE: `demos/net-macro`, live: three networks on a three-arm spiral, each one line (`2 3 softmax`, `2 4 tanh 3 softmax`, `2 16 relu 16 relu 3 softmax`: 37.5%, 87.5%, 100% of the test points); weights from `just net-train`, which reads the specs from the program; the page shows the call, what the macro wrote (`microscope::run::expanded`), the parameter count and the decision map, expands and counts any typed spec and shows the macro's message for a refused one; `just expand SLUG`; the shell embeds `.xtlm` libraries. Tests: each macro-written network decides what a direct computation decides at every map point; a typed spec; refusals; a quote cannot break out; every line shown | |
| 4 | concise-shapes | DONE: literal `c_at` chains became strands (46 places), pooling one reduce; the gate refuses the long form |
| 5 | one-source-of-truth | DONE (the user's decision after reviewing what the macro checks): `net:m_odel<` writes, from one spec, the weights' loading (shapes from the spec; a file of the wrong count stops the program naming the layer: `r_eshape` would silently repeat or cut) and the function; the demo, its trainer and its page use it; ask M12 (shapes in types: an ascription built-in, rank in types, sized types as research) |
| 3 | release-2 | catalog, docs, retrospective | |

### Net.xtlm as built

```
"nn:"  u_se< "NN"
"net:" u_se< "Net"
u:x_or := "2 2 relu 2 softmax" net:n_etwork< "w1 w2"
```

expands (visible with `xetal expand`) to

```
u:x_or := ({ g1:x -> nn:s_oftmax (nn:r_elu g1:x nn:d_ense w1) nn:d_ense w2 })
```

so the macro adds notation, never semantics. Differences from the
first sketch: one weight array per layer (NN keeps the bias as the
weights' last row), the function's name is bound by the caller, and
two companions: `net:p_arams<` (the parameter count, a number written
at compile time) and `net:s_hapes<` (the weights against the spec).

## Saga 4 -- training in X_eTaL  [ACTIVE]

After the APL Wiki's neural-network resources (2026-10-07, the
user's request to look for more demos): every APL work it lists
(Shinkarov's CNN in APL, Hsu and Serrao's U-Net, Serrao's workshop,
Powell's tutorials) trains its network in the array language, while
every demo here infers with weights trained offline. Backpropagation
is where array notation pays off most: each layer's gradient is one
expression, as its forward pass is. This saga trains in X_eTaL.

| # | Step slug | Delivers |
| - | --------- | -------- |
| 1 | backprop-microscope | DONE: `demos/backprop`, live: a 2-4-3 network (tanh, softmax) on six spiral points; forward, loss, D2, G2, D1, G1 (each one expression, the bias row as `nn:d_ense` keeps it), the step; all 27 gradients checked against central finite differences (largest gap printed, below 1e-8); the page takes steps (the loss history), a learning-rate slider (too large overshoots), every array as a colored table, all code shown. Tests: equal to a hand-written Rust backprop to 1e-12; ten steps each lower the loss. Planned: one training step of a small network, every array visible: forward, loss, each layer's gradient as one expression, the update; every analytic gradient checked against finite differences (Powell's emphasis); CLI and page |
| 2 | train-live | DONE: `demos/train-live`, live: 300 spiral points made in X_eTaL (seeded), a 2-16-3 network (tanh, softmax), the microscope's gradient as one vector, Adam written out, `p_ower` for the steps; the page runs 25 steps a frame to step 1000, redraws the decision map and the loss and accuracy curves, has a learning-rate slider (about 8 ms a step natively); 1.319 to 0.062 loss and 30% to 100% right in 400 steps. Tests: 50 Adam steps equal Adam in Rust to 1e-9; learning to over 95% run by run. The state packed into one vector (ask M13). `just run` now passes `--seed 1`, as the tests do. Planned: the spiral classifier trained in the browser by X_eTaL: the loss curve and the decision map changing as it learns |
| 2b | boxed-state | DONE: train-live's state is seven boxed arrays (W1 W2 M1 M2 V1 V2 k) instead of one packed vector with offsets; output unchanged; ask M13 updated (tuples being added upstream) |
| 3 | net-train-macro | DONE: `net:g_radient<` writes a spec's backprop (each layer's gradient, each activation's slope); `net:t_rain<` an Adam step on a boxed state; `net:s_tate<` its start. Gradients checked against finite differences (relu, sigmoid, tanh, none); 20 written steps equal train-live's hand-written ones; demo train-xor learns XOR from random weights in 300 steps |
| 3b | net-macro-train | DONE: inserted at the user's go: the net-macro page trains any spec from 2 inputs to 3 softmax in the browser, from random weights, by the step `net:t_rain<` writes (train-it.xtl, its CLI form); curves drawn by X_eTaL-libraries' Plot (`p:l_ine!`, in the program), the first use of Plot; the runner returns `[]S_HOW` pictures |
| 3c | plot-curves | DONE: the backprop microscope's losses drawn by Plot after every step (the steps taken, now, after this one); train-live's curves drawn by Plot instead of the page's Rust SVG (removed); Plot's limits worked around and named; ask M14 (pictures from runs at once). When X_eTaL-libraries' Plot PR (axes, several lines, a free range, one point; asked 2026-10-07) is merged: move `XETAL_LIBRARIES_COMMIT`, drop the stretching, one chart for loss and accuracy |
| 3d | xref-docs | DONE: inserted at the user's go: X_eTaL's `xetal doc` cross-reference, as X_eTaL publishes its own /doc: NN and Net fully documented (`##` on every item, `###` sections, 28 `## >>` examples run by the gate via `just doc-test`), every demo program's comments as file docs, definition docs and `###` sections; `just doc` builds pages/doc (live at /doc), linked from the catalog, each card ("Read the code"), every page's footer and the README; CLAUDE.md rule 5e; the library template documented |
| 3e | plot-pin | DONE: X_eTaL-libraries pinned at 6e7ef0a (Plot's axes, several lines, `p:c_hart!`), which needs X_eTaL 96060b5 (`h:` names): X_eTaL pinned there too, NN's and Net's private helpers migrated to `h:` (`xetal migrate`), every baseline unchanged, bench-check 1-15% faster; the three training pages draw loss and accuracy in one titled chart with axes, from the first run; the stretching and the two-point wait dropped. Re-pin when X_eTaL ships its tooling fixes (the user, 2026-10-08) |
| 3f | tuple-state | DONE: X_eTaL pinned at d284a8c (tuple patterns, the doc site by directory) and X_eTaL-libraries at 69bb369 (every library documented, its helpers h:); train-live's Adam state and Net's training state (`net:t_rain<`, `net:s_tate<`, `net:g_radient<`'s weights and gradients) are tuples taken apart by name; the pages write a tuple and read the parts by name; every output unchanged; Net's expansions reblessed (the step takes its state by pattern) |
| 4 | cnn-backprop | DONE: demos/cnn-backprop, live: the Tiny CNN trained in X_eTaL from random weights on 600 MNIST digits (200 to test), after Shinkarov (reimplemented and cited): the batched forward, then softmax, dense, unpooling (each block's gradient to its first largest item: ties are common), ReLU and the convolution undone, checked against finite differences; Adam on a tuple state, 20 digits a step; 12.5% to 91% of the test digits in 40 steps. The page trains 2 steps a run, the state in files it keeps, the filters, 20 readings and Plot's chart redrawn. Ask M16 (matrix-product speed). Unpooling uses a block reshape, not `r_eplicate` |
| 4b | char-builtins | DONE: `[]D` and `[]A` were already in the pinned X_eTaL (its quads lane); X_eTaL pinned at 4952874 anyway (a file's doc header needs no blank line; stray doc blocks warn: none here), X_eTaL-libraries at c9de461 (its literals replaced too); `"0123456789"` became `[]D` (cnn-digits, Net's size check), the letters that label map cells `s_elect []A` (moe-router, net-macro, ternary-net: their maps print A, B, C, the baselines changed in case only). Lowercase letters have no built-in yet. cnn-backprop's screenshot added. X_eTaL's next design (names bound once) will forbid the rebinding the training steps use |
| 4c | bound-once | DONE: the user's go: ahead of X_eTaL's step 091 (lang-choices M1: a name is bound once in its scope, a lambda's body with its parameters one scope), the repository is bound-once clean with pure code, no `!`: the Adam steps (train-live, cnn-backprop, Net's train test) name their new values (`m1`, `v1`, `t`/`n`), Net's `t_rain<` writes new names and the moved weights in the returned tuple, its `names` is one expression, the data reads name their raw numbers (`nums`), ternary-net's layers `h1`, `h2`, train-it's trained weights `v1 .. v3`, the pages' runs `s0`/`s` to `s1` with pattern names apart from the program's. `scripts/check-rebind.py` (self-tested, in the gate, `just check-rebind`) over programs, libraries and expansions; `microscope::source::rebound` in the training pages' tests; CLAUDE.md rule 5f. Every output unchanged; Net's expansions reblessed |
| 5 | classic-ml | DONE: libs/Learn (`ml:`), after APLearn's list, each a fit and a predict, every iterative fit a step under `p_ower`, nothing by chance: k-means (`a_ssign`, `k_step`, `i_nertia`, farthest-first `f_arthest`, `k_means`), kNN (`k_nn`, nearest-first with masking: ask M4), PCA (`p_ca` by power iteration with deflation, `p_roject`), logistic regression (`l_ogistic`, `p_redict`); 8 doc examples, 12 property checks, basics, a blobs demo, types pinned (tuples in them). demos/k-means, live: five blobs, a step at a time from a poor start (settles at 50.17, three centers in one blob) or farthest-first (1.897), k from 2 to 8; the step equal to k-means in Rust; Learn's k_step quoted on the page |
| 5b | orange-brand | DONE: the user's ask: ML is orange. The favicon recolored from blue to orange (each entry's hue turned, its design kept); X_eTaL's orange logo (images/xetal-logo-orange.png, from ../X_eTaL/images) heads the README, the catalog, the recorded page and every demo page, 60 px high (its 704 x 402 kept: about 3 rows by 15 columns); the heading the logo then "X_eTaL ML" in bright orange (#FF7A00: CSS on the site, an SVG title in the README, as GitHub drops CSS); the old logo removed; every screenshot retaken |

Left out for now: MENACE (it fits X_eTaL-games, and X_eTaL ships
TTTML, which learns tic-tac-toe by self-play); a U-Net (a tiny one on
synthetic shapes is a later stretch).

## Saga 5 -- libraries and demos

The libraries the demos were written ahead of, the demos built on
them, and the research demos (from research4). Begun 2026-10-09 as
"post-launch"; renamed 2026-10-12, the user having postponed the launch
until there is enough progress across all the X_eTaL repositories:

| # | Step slug | Delivers |
| - | --------- | -------- |
| 0 | xetal-091 | DONE: X_eTaL pinned at f7f9821 (its step 091: each name bound once, `_ :=`, `[]E_X` in sessions), X_eTaL-libraries at 746e666; every baseline unchanged (step bound-once had made the repository clean); bench-check 2-13% faster |
| 0b | catalog-toc | DONE: the user's ask: the catalog's table of contents, a sticky sidebar at the top left as X_eTaL-games' site has it: the demos alphabetized by title, each linking to its card, then the recorded page, the cross-reference and the repository; above the cards on a phone (under 760 px) |
| 0c | embedding-explorer | DONE, moved first (the user's choice, 2026-10-10: a new demo sooner): libs/Embed (`em:`: `c_ounts` by sorting, `c_ooccur`, `p_pmi`, `p_roject` (random indexing, signs from a hash), `u_nit`, `c_osines`, `n_earest`), 3 doc examples, 7 property checks, a pets demo; demos/embedding-explorer, live: 4,000 generated sentences over 70 words (train/, habits within kinds), 64-number embeddings, Learn's PCA to a 3-D cloud to drag or spin, k-means' clusters as colors, a word's 5 nearest on click; measured how many sentences are enough (about 1,600) |
| 1 | quant | DONE: libs/Quant (`qz:`): `f_loats` (b bits after the leading one), `s_nap` (to a step), `i_nt` (symmetric b-bit), `c_ut` and `t_ernary` (ternary codes), `s_cale` (the kept mean size), `a_dds` (a ternary layer by additions); a step or threshold per layer passed as an array; 12 doc examples, 8 property checks, a formats demo. ternary-net moved onto it: its per-layer choices kept in the program, the rounding Quant's; output byte for byte unchanged |
| 2 | conv | DONE: libs/Conv (`cv:`): `w_indows`, `c_onv` (`(K, B) cv:c_onv x`), `p_ool`, `u_npool` (to each block's first largest), `g_rads`; any batch and picture size; 6 doc examples, 7 property checks (equal to cnn-digits' one-picture formula, gradients against finite differences), an edges demo. cnn-backprop moved onto it, output byte for byte unchanged. cnn-digits keeps its own one-picture convolution: it is the catalog's first lesson, the convolution written out as one expression; Conv is tested equal to it |
| 3 | attention-lib | DONE: libs/Attention (`at:`): `s_cores` (`(Wq, Wk) at:s_cores X`), `w_eights` (masked softmax), `c_ausal`, `a_ll`, `h_ead` (`(Wq, Wk, Wv) at:h_ead (X, M)`); 7 doc examples, 7 property checks (rows sum to 1, masks, the head as weights times values, reordering), a previous-token demo. The attention demo keeps its two lines, now through the library (its README shows the library's two lines too); output byte for byte unchanged |
| 4 | norm, sample | DONE: libs/Norm (`nm:`: `s_tandard`, `l_ayer` with `(g, b)`, `r_ms`) and libs/Sample (`sm:`: `g_reedy`, `p_robs` at a temperature, `t_op` k (repeated masked max: ask M4), `d_raw` by uniform numbers the caller gives, no chance inside); 14 doc examples, 13 property checks, a demo each |
| 4b | uv-python | DONE: the user's rule: Python only through a uv venv: `scripts/py` (the repository's `.venv`, `uv venv --managed-python`, Python 3.12 from `.python-version`, made on first use); every script call, `python3 -c` and `python3 -m http.server` through it; the `.py` shebangs `uv run` with a managed Python (removed in lib-site); `.venv` gitignored; README requirements, CLAUDE.md rule 8a. Prompted by a Homebrew upgrade that unlinked `python3` and left the system's 3.9 (no `tomllib`) in its place |
| 5 | micro-gpt | DONE: demos/micro-gpt, live: Karpathy's microgpt (1 layer, 16 wide, 4 heads, 4,192 weights) trained offline by train/ (adapted from microgpt-rs, same model, training and seed; makemore's names fetched into work/) in 2 s; its forward pass in X_eTaL on Norm, Attention and Sample, the whole name under a causal mask, equal to the Rust model's scores to 1e-9; names generated by `p_ower` with uniform numbers from the caller (r_oll! at the command line, the browser's randomness on the page); the page: ten names, temperature, starting letters, a name's next-letter probabilities and its four heads' attention |
| 5b | asks-sync | DONE: docs/xetal-asks.md in step with X_eTaL's ledger: M4, M10, M12 declined (their workarounds now permanent unless the user decides a per-row grade or a mix), M5 planned, M6 partly landed, M11 and M14 new; M15 and M16 detailed, in order, marked not yet taken with entries ready for X_eTaL's asks.toml; `scripts/check-asks.py` (self-tested, in the gate, `just check-asks`) checks every ask against the pinned ledger; CLAUDE.md rule 5 |
| 6 | embed, embedding-explorer | Embed (`em:`); PCA from 64 dimensions to a rotatable 3-D cloud (PCA may come with saga 4's classic-ml) |
| 7 | optim | DONE: libs/Optim (`op:`): `s_gd`, `a_dam` (`(lr, t) op:a_dam (w, m, v, g)`), `a_damWith` (its own decay rates); exactly train-live's written-out Adam (tested); 5 doc examples, 5 property checks, a narrow-bowl demo (gradient descent 10 to 8.35, Adam to 2.24 in 100 steps). CNN training moved onto it, output byte for byte unchanged; train-live and Net's `t_rain<` keep Adam written out, their lesson |
| 8 | lib-site | DONE: site/, the libraries' own live site at /libs/, adapted from X_eTaL-libraries' site (Yew, trunk): a landing page with the libraries in four groups (networks, transformers, pictures and weights, classic machine learning), each library's reference page, its types and its demos to run and edit in the browser (Net's macros expanded beside them); `cargo test` runs every library demo through the browser engine against its baseline and every library's pinned types; built by `just pages`, checked in the browser by the gate and `just check-live` (site/browser.txt); linked from the catalog (lede, table of contents, footer) and every demo page's footer. The `.py` scripts lost their `uv run` shebangs and executable bit: Python runs only as `scripts/py`, the uv venv (the user's rule, again) |
| 9 | world-model, diffusion | DONE. world-model: demos/world-model, live: a 2 x 2 ball bouncing in a 12 x 12 box (1,936 states, the world written in X_eTaL: `u:s_tep` moves and bounces every state at once), a 432-64-144 network trained offline (train/: Adam with weight decay 0.2, a fifth of the states held out by a seeded shuffle; without decay it memorized: 7 in 10 unseen, with it 9 in 10); the program runs all 1,936 states as one batch (8.5 s, the GPU candidate: row 1 below) and counts the next frames predicted exactly (1,897; 348 of the 387 unseen), checks one window against the Rust model to 1e-6, and dreams twelve steps on its own predictions (`u:p_lace`, the likeliest 2 x 2 block), on track for all twelve; the page: place the ball, pick its speed, a state it never saw, dream step by step, the predicted frame as probabilities with the true ball outlined, the dreamed ball against the true one. diffusion: demos/diffusion, live: 10 x 10 pictures of a disk, a hollow square or a cross; a denoiser (110 in: the pixels and the level one-hot, 256 ReLU, 100 sigmoid) trained offline (train/: pictures made fresh per batch, a cosine schedule of ten levels, 40,000 Adam steps, 2.5 min); DDIM sampling in X_eTaL (`u:s_tep` over a tuple by `p_ower`, no new noise, so the same noise gives the same pictures), the trainer's starting noise reproduced to 1e-6; 128 pictures at once at the command line (6.3 s, the GPU candidate: row 3 below), 97 of them a shape's worth of pixels; the page: eight pictures from the browser's noise, every level as a strip, one large, the same noise again. Disks and crosses come out clean, hollow squares often as bars: a dense denoiser has no notion of neighbors (a convolutional one is a later step) |
| 10 | gpu-results | DONE (2026-10-13): X_eTaL-gpu pinned fac586c and ran rows 1 and 2 on the Apple M1 Max and the RTX 5060 Ti (2026-10-12; its `docs/ml-results.md`, `docs/rtx5060-ml-results.md`): the world model's counts 1,897 and 348 the same on the evaluator, its interpreter, both cards in f32 and through the `xetal-x` facade; the benchmarks within a relative 1e-5; the kernels a fraction of a millisecond, compiling and the text bridge the cost; a kept session runs the batch in 2.5 ms on the M1 Max against NumPy's 1.3 ms (the gap closes as the batch grows). Summarized in docs/speed.md, "On GPUs", with the links; gpu-asks.md: GA2 and GA3 landed there (transpose; rotate, drop, reshape for Conv's windows), GA5's input half (binary file binding, 13 ms against 460 ms), GA6's G3 declined by X_eTaL (per-shape specialization instead). Row 3, the denoiser, is queued there after its re-pin past 1eded2a |
| 11 | gpu-source | DONE (2026-10-13, the user's note): X_eTaL-gpu builds the lowering itself (its A4b; `xetal-gpu run FILE.xtl`, the macro `gp:a_ccel<`, its step models-from-source for our world model and benchmarks from their own `.xtl`). gpu-asks.md: GA6 in progress there, the subset it lowers and where each program stands (the world model's `u:p_redict` and `u:e_xact` and the benchmarks fit; the denoiser's `c_at_2`, NN's `o_neHot` and `^` do not); GA7 filed (those three in the subset; meanwhile the joined input bound per call, or `u:c_lean` split here on request). The results paragraph above |
| 12 | gpu-pipeline-note | DONE (2026-10-13, the user's note): X_eTaL-gpu queued gpu-pipeline (pinned f32 buffers, non-blocking transfers, the batch axis chunked on streams with events, `explain` of the stages, `--repeat` for sustained items per second; the world model its first measurement). Recorded above and in gpu-asks.md what our workloads offer it: the world model a stream of independent batches (chunking), the denoiser ten dependent stages per batch, a stream of batches pipelined stage by stage (parallel stages); no program here changes for it |
| 13 | denoiser-split | DONE (2026-10-13, the user's go; GA7's meanwhile): the denoiser's network is `u:n_et`, the two dense layers over the joined input, one function of arrays in the subset X_eTaL-gpu lowers; `u:c_lean` is the join around it (`c_at_2`, NN's `o_neHot`), the evaluator's; the schedule's roots taken once as the vectors `ra`, `rb`, so `u:s_tep` only selects from them. Output byte for byte unchanged (the CLI and browser baselines and the page's tests pass as they were); the demo's README quotes the new shape |
| 14 | gpu-batches | DONE (2026-10-13, the user's go): the batch programs for rows 4 and 5 above, each a second `.xtl` in its demo with its own CLI baseline and its data from the demo's trainer (the weights unchanged: both trainers are deterministic). cnn-digits/batch.xtl: the first 1,000 test digits (pixels as whole percents, 1.7 MB) as one batch through Conv, the reader `u:r_ead` one function of arrays; 971 read right, the Rust count (checked); 9 s. micro-gpt/batch.xtl: the first 1,000 names as token rows, scored by the demo's forward pass by `p_ower` over (the totals, the name); the mean loss per token 2.2496 (the Rust model's to 1e-9) and 2,098 of 7,000 next letters right (exactly); 3.5 s. Also: two top-level `##` blocks above expressions in ternary-net and world-model made `#` (the doc site's two warnings; the upstream fix is for blocks inside definitions) |
| 15 | finite-shift | DONE (2026-10-13, X_eTaL-gpu's finding while lowering its copy of `sm:t_op`: 1e300 is infinity in f32 and 0 times it NaN): Sample's `t_op` shifts a taken item by 1 + the range of the values, Learn's `k_nn` by 1 + the largest distance, both finite; Attention's 1e9 on the mask is within f32 and stays. Every output unchanged (the libraries' baselines, 87 doc examples, the demos). The rule in gpu-asks.md: a mask's shift is a finite function of the data |
| 16 | conv-denoiser | DONE (2026-10-14, the user's go): the denoiser on the Conv library: `u:n_et` is a 3 x 3 convolution of the noisy picture (8 filters, valid, 8 maps of 8 x 8) with ReLU, the 512 features through the dense layer with the level's own weights added (`wh`, a row per level: no join inside), and the output layer; still one function of arrays. The trainer (eight threads, a fixed order, 100,000 steps, about 7 min) reports the held-out cross-entropy (2,000 pictures) and trains the earlier dense denoiser on request for the comparison: 13.36 dense, 12.77 convolutional, at levels 4 to 6 a quarter to a third less; pooling the maps first lost (13.64). Hollow squares come out as squares more often; shapes thinner, so 85 of 128 samples reach 8 pixels on (97 before): the count is a floor, not a score. The program (20 s for the batch), the page (unchanged in form), the README with the measurements; baselines reblessed (the pictures and the count), the Rust check still 1 |
| 17 | re-pin | DONE (2026-10-17, the user's go): X_eTaL f7f9821 to b8a6ac3b (hygiene by choice with `[]G_ENSYM`, MC30; raw strings, ST4; `xetal expand` laid out as a program, D159; the doc-unattached fix for Net.xtlm and Sample.xtl, D153; the pipeline design, D155) and X_eTaL-libraries 746e666 to ed91e6d (Equations, its reader total; the aliases). Every run baseline, pinned type and page passed unchanged; Net's six expansion goldens were rebased on purpose: the automatic renaming is gone, so Net's macros take every name they bind from `[]G_ENSYM` (`h:n_ames`), which also closes a hole the old goldens show (`u:g_a := { (W1, W2, W3) -> A0 := X ...` was never renamed: a caller's `A0` or `n` was captured); `tests/hygiene.xtl` runs the three macros with the caller's inputs named `n`, targets `k` and weights `x`. GA1 and GA7 marked landed there (0fbf5f2: every program runs from its own source); M15 and M16 are in X_eTaL's ledger |
| 18 | pipeline-macro | DONE (2026-10-17, the user's go; `docs/pipeline-macro.md`, X_eTaL D155, its step 109): `libs/Pipeline/` (`pipe:`), `Pipeline.xtlm`: `pipe:p_ipe<` splits on `\|>` outside brackets, braces, parentheses and strings; a stage after the first is a function (applied; a quoted lambda or name loses its quote, a train or `'+ r_/` stays), an expression naming `it` (a block of its own, `it := <previous>`: a lambda could not take a tuple) or an expression of earlier names (the oracle's own examples need it); ` : name` and ` : (Q, K, V)`; ` [N O]` emits `error[shape]: stage 2 (y): shape 4, expected 3`; fresh names from `[]G_ENSYM`; the expansion laid out one stage per line. Every reader rule uses all of its text or refuses it (`error[bad-pipeline]` naming the text): 14 refusal goldens, the Equations cases among them (text after a shape, two groups side by side, an unknown character). Tests first: the five oracle pipelines one-line and raw-string equal X_eTaL's `pipes-oracle.xtl` (`14 25 / 20 31`, the softmax, layer norm, `2 / 2`, `3.0 6.0 9.0 12.0`), 14 differential checks over every kind of stage, `pipe:t_ext<` for the round trip, the shape check failing with its message, 39 baselines with every expansion pinned; the demo mlp.xtl (XOR as a pipeline beside the same by hand); the libraries' site lists it under Networks. The design doc's `a_dd_axis` corrected to `a_ddAxis` (first letter underlined only) with an as-built section |
| 19 | gate-affected | DONE (2026-10-17, the user: the gate took about 30 minutes for any change): `scripts/affected.py` plans from the files changed since HEAD (tracked and untracked; `GATE_BASE` another base): a demo for its own files, a library for its own files plus every demo and library importing it (`u_se<`) and the libraries' site, `all` for the shared shell, the scripts, the justfile, the templates or a pin, nothing for docs alone; a self-test on known changes. `just gate` runs the plan (the cheap checks always: the pinned X_eTaL, the libraries' examples, asks, rebind, c_at chains, spelling, markdown), `just gate --full` everything as before (the tooling self-tests, shared/microscope, every demo, library and page; the plan asks for it by itself when it must). `build-pages.sh` and `doc-test.sh` take names; every step of 5 s or more prints its time and the end lists the slowest three; `GATE_PLAN` tries a path. A library's change now gates in minutes, not half an hour |
| 20 | pipeline-bias | DONE (2026-10-17; docs/pipeline-macro.md, decided with the user): `nn:a_ddAxis` (`(k, b) nn:a_ddAxis V`: b along axis k by an outer product and a permuting transpose; b's length must be the axis's extent, `error[shape]` naming both, a missing axis `error[axis]`) and `nn:b_ias` over it with the convention by rank (rows: the last axis, as `d_ense` keeps it; a channel-first picture: the first; a batch of them: the second); X_eTaL has no implicit broadcasting and the APL family disagrees on the axis, so it is named or follows the stated convention. Tests: NN's bias.xtl (9 checks against r_eshape/t_ranspose by hand), Pipeline's bias.xtl (the linear regression as `X \|> it '+ '* i_nner W \|> b nn:b_ias it`, equal to the written-out form) and bias-wrong.xtl (the error through a stage); doc examples; the pages of NN, Conv (a bias on a batch of maps names the second axis) and Pipeline |
| 21 | integrals | DONE (2026-10-18, the user: in parallel with the cloud lanes, for the blog; no peer repository touched): `libs/Integrals/` (`integ:`), `Integrals.xtlm`: `@ integ:i_ntegral< "integral tau:=a..b by trapezoid 64 \| body"` in X_eTaL's Saga 42 call shape (decisions 2 to 4: `integral` read like `sum`, the rule and grid as option words with the documented default trapezoid 64, simpson on an even grid, improper integrals refused naming the two ways out); the body X_eTaL over the grid vector bound to the variable (until Equations' grammar takes `integral`, decision 1: the library lifts whole); the expansion one definition per line with fresh names; `integ:t_ex<` the LaTeX from the same reading. Every reader rule uses all of its text or refuses it: 11 refusal goldens. Tests first: exact on constants and lines (trapezoid) and quadratics and cubics (simpson); the orders on e^x (error ratios 4 4 16 16); the box convolution converging to the triangle (first order at the jumps: the sample takes the body's value as written, so high by half a step, 0.25 then 0.25/16), the plan's oracle 1.0 by hand and by the macro; the LaTeX; every expansion pinned. Demos convolution.xtl (the 17-value strip beside the triangle, e - 1 by simpson 16) and gelu.xtl (GELU as x (1/2 + the integral of the Gaussian density), 0.8413 at 1, the tanh approximation off by at most 0.000473; the KL divergence of two Gaussians as an integral beside its closed form, 0.443147), both live on the libs site. The site gains a Mathematics group |
| 22 | repin-code-checks | DONE (2026-10-19, the user's go): XETAL_COMMIT f7848d00 (X_eTaL #146 code strings checked: every inside lexed, `## takes: code`, string-kind and code-check; #147 fmt keeps statement macro calls) and XETAL_LIBRARIES_COMMIT 0f82f93 (Equations on Parsing, `## uses: parsed`), Equations and Parsing linked; Flow's three refusals of text that does not lex now the lexer's (rebased); Flow keeps no `## takes:` line. The rebind checks (scripts/check-rebind.py, microscope's `rebound`) see two lambdas on one line as two scopes (Integrals' golden had been refused) |
| 23 | pipeline-demos | DONE (2026-10-19): `demos/pipelines/`: dense with its bias, the stable softmax, layer norm and one attention head as Flow pipelines, each beside its expansion and the same layer in plain X_eTaL, checked; a CNN layer four ways (rosetta.xtl): Equations' `e:e_q<`/`e:t_ex<` (the activation as `max(..., 0)`), a pipeline with a `[O Ho Wo]` check, its expansion, Conv's `c_onv`, all equal and checked against Rust loops in the page's tests; a live page, the equation drawn by KaTeX (vendored, `shared/katex/`); expansion goldens (test.sh); ask M18. Open with the user: a Blender column (the layer visualized or animated) |
| 24 | diffusion-rosetta | DONE (2026-10-20, the user: two screenshots of a latent diffusion sampler as a demo, the asks filed): `demos/pipelines/diffusion.xtl`, a second Rosetta Stone on the pipelines page. The paper's first form, X0 = D(zT + sum from T down to 1 of [a_t z_t + b_t eps_theta(z_t, t, E_text(prompt)) + sigma_t eps_rand] - z_t), is a recurrence (the term at t uses the z_t the term before made): one step is the summand as an Equations call over the batch and the latent, applied T times by `p_ower`, the networks X_eTaL functions outside the string (dense layers with made-up weights; the diffusion demo has the trained ones); the second form, D(zT + the sum of dz_t), a plain Equations sum over the differences collected as the steps run, telescoping to the same latents. The sampler as a Flow pipeline (`p_ower` as a stage, `$(z0, t0)`, the decoder's `[N K]` checked) and in plain X_eTaL; all three equal, checked, and the page's tests redo the arithmetic in Rust; both equations drawn (KaTeX; an example's `math` is a list of statements). What the grammar refused was said, not worked around silently: asks for X_eTaL-libraries' Equations in docs/xetal-asks.md's section for the libraries (E1 a descending range `t:=T..1` dies in the expansion, `error[domain]: a count cannot be -1`, instead of a refusal; E2 a recurrence word; E3 the caller's functions by name). Seen there: X_eTaL-libraries' Equations at its HEAD reads `integral` (this repository's Integrals is its oracle), so the next libraries re-pin lifts Integrals as planned |
| 25 | equation-palette | DONE (2026-10-20, the user: a palette of math symbols and parts to compose an equation or a tensor pipeline, for the WASM UI): `shared/microscope/palette.rs`, a composer component for the demo pages, on the pipelines page as its last choice: programs to start from (the dense layer with its bias as an equation with its LaTeX; the stable softmax as a pipeline), the palette (`palette.tsv`, data: the equation grammar's words, references, functions and Greek names; the pipeline's `|>`, `it`, labels, shapes, derived functions; the libraries' functions as stages; the imports) put in at the caret, the program in ASCII beside its rendered form, Run (the lines printed, those between `$$` drawn by KaTeX, the errors as X_eTaL places them in the text) and Expand. The input is the grammar, not LaTeX, which comes out of the same reading; the macro's whole-text rule is the lint of the notation. Tests natively: the groups and parts (every non-equation part lexes, every stage runs in a pipeline over fitting data), insertion at the caret, composing (the LaTeX taken out, the expansion, an error placed, a refusal naming its rule), the starters. Lint in the browser stays X_eTaL's (its step 158) |
| 26 | math-asks | DONE (2026-10-23, the user: an Activations library and what it needs; where mathematics lives): the asks filed before the work. `docs/xetal-asks.md` M19 (the libm builtins `t_anh`, `e_rf`, `s_qrt`, `e_xpm1`, `l_og1p`, by the rule that a builtin is what libm gives at full accuracy and a library cannot write as well; nothing leaves the base language); `docs/xetal-libraries-asks.md`, new, the asks for X_eTaL-libraries in a file of their own (E1 to E3 moved from xetal-asks.md; E4 a Trig library, E5 Special functions, E6 Equations' functions widened by tanh, erf, sigmoid, E7 a numeric `derivative x:=x0 by central h \| body` word in the shape of `integral`, E8 symbolic differentiation as a fourth reading of the tree, `e:d_iff<` and `e:d_tex<`, E9 indexed gradients by the Kronecker-delta rules, backpropagation from the loss); the Activations row in the libraries table and the steps below; the settled question |
| 27 | activations-lib | DONE (2026-10-23, the user's go; X_eTaL-libraries planned E4 to E9 as its Sagas 17 to 23 and X_eTaL M19 as its step 169 the same day, so the library carries its own tanh and erf meanwhile): `libs/Activations` (`act:`): 15 activations (ReLU, leaky, PReLU with a slope per column, ELU, SELU, sigmoid, tanh, the stable softplus, SiLU, Swish, GELU in the tanh form and exact through a private A&S 7.1.26 erf, Mish, hard sigmoid, hard swish), the four gated layers as `(W, V) act:s_wiglu x` (the activation on x W times x V), 15 derivatives in terms of the input; 20 doc examples run by the gate; 37 checks (the relations, the bounds and extremes, where the smooth ones dip, the two GELUs within 4.73e-4 and the exact one agreeing with the Integrals demo's quadrature, hard swish within 0.15 of SiLU, SELU's fixed point on 4,000 Box-Muller normals, the gates written out, every derivative against central differences within 1e-5); the pinned types; the curves demo (15 x 13 tables of values and slopes); the page with the comparison table (cost in transcendental calls, monotone, bounded, the dip, the slope's range, who uses it) and the papers; the site's Networks group. Planned: `libs/Activations` (`act:`) as the libraries table says: the functions, the gates, a derivative each; tests first: 16 property checks (values at known points from the papers, the smooth ones below zero where they dip, hard swish equal to swish within its bound, SELU's fixed point: mean 0 and variance 1 kept through a layer of standard-normal weights), every derivative against central differences, the exact GELU against the Integrals demo's integral (its tanh form off by at most 4.73e-4), softplus and Mish finite at 1e3 and -1e3; the pinned types; the page; a demo (one picture of every curve, drawn by X_eTaL as SVG) |
| 28 | activations-move | DONE (2026-10-27): NN's `r_elu`, `l_eaky`, `s_igmoid`, `t_anh` removed (its header, page, basics and checks c7 to c9, now Activations' own); every caller on `act:` with the import beside NN's (14 programs, Integrals' gelu demo importing only Activations; the pages, READMEs, the palette, microscope's tests), Flow's examples and refusal message say `act:r_elu`; Net's words are relu, sigmoid, tanh, softplus, silu, gelu, mish, selu, hardSigmoid, hardSwish (Activations) and softmax, logSoftmax (NN), the expansion calls both libraries (programs import both); the slopes of relu, tanh and sigmoid stay at the output, the smooth ones are Activations' derivatives of the dense layer computed again (`act:d_gelu A0 nn:d_ense W1`), checked against finite differences within 1e-7 for three new networks (gelu and mish, silu and softplus, selu and hard swish); the refusal tests use swiglu (a gated layer, not a word). Every CLI output unchanged; the expansion and error goldens re-blessed after review: names, the added import's line numbers, NN's error line (112 to 87), the wider word list. `bench/activations.xtl` (the fifteen on 65,536 Floats, 200 ms here) in bench-check; the per-function cost is the page's (row 29). Planned: NN loses `r_elu`, `l_eaky`, `s_igmoid`, `t_anh` (its page and plan row say where they went); every caller moves to `act:` (about 40 files, programs, pages, READMEs and the goldens that print the names: backprop, cnn-digits, cnn-backprop, diffusion, pipelines, train-live, world-model, Flow's demo, docs and refusal goldens, Integrals' gelu demo, Net and its tests, NN's own tests and xor demo, the palette's Stages group, the pages' heads; Learn, Sample and Attention use only softmax and stay); values unchanged, so only Net's expansion goldens (which print the names) are re-blessed on purpose; Net's activation words widened (`"2 16 gelu 3 softmax"`, the slope table taking the pre-activation for the smooth ones); `bench/activations.xtl` (ms per million elements, each function) in bench-check and docs/speed.md |
| 29 | activations-page | `demos/activations`, live: every function and its derivative on one axis, drawn from the numbers X_eTaL computes (SVG), sliders for alpha and beta, a table measured not asserted: transcendental calls per element (0 for relu, leaky, PReLU, hard sigmoid, hard swish; 1 exp for sigmoid, tanh, SiLU, ELU, SELU; 2 for softplus and GELU; 3 for Mish), monotone or not, bounded below, zero-centered, the slope's range (sigmoid's 0.25 at most: vanishing; ReLU's 0 for x < 0: dying), the bench's ms, and where each is used (GELU: BERT, GPT-2; SwiGLU: LLaMA, at 8/3 d hidden to keep the parameter count; Swish: EfficientNet; Mish: YOLO; hard swish: MobileNetV3; SELU: deep MLPs without a norm layer); all code shown |
| 30 | activation-compare | the same network, every activation: train-live's spiral net (2-4-3) trained with each activation in the browser, the loss curves side by side; at width 4 ReLU dies (units at zero: shown), sigmoid saturates, tanh, leaky, ELU and SELU train; the choice of activation for a small network made visible; the trainer's slope from `act:d_*`, so the step is one expression per activation |
| 31 | glu-ffn | the gates in a transformer's feed-forward layer: micro-gpt's FFN as GELU (`x W1 -> gelu -> W2`) beside SwiGLU (`(x W) * silu(x V) -> W2`, V the extra weights at 2/3 the hidden width, the same parameter count), the two trained offline by train/ from the same seed, their held-out loss and a generated name each on one page; the cost of the gate as a count of multiply-adds, measured |
| 32 | micro-bert | DONE (2026-10-27, the user: BERT as a one-line equation and a demo of it as a tensor pipeline; top-level, its twin micro-gpt): `demos/micro-bert`, live: a tiny BERT (2 layers, 16 wide, 2 heads of 8, a GELU feed-forward layer of 64, post-LN, 7,594 weights; no [CLS]/[SEP], segments, attention biases or 80/10/10 masking, said on its page) trained offline by train/ (micro-gpt's tape autograd with tanh for GELU, a fixed seed; each name with each letter masked in turn, 4 names a step, 6,000 steps, Adam, 3 minutes; `just micro-bert-data`): on 1,000 held-out names 35% of blanks right, 65% in its top three, against 17% for always a. micro-bert.xtl writes it four ways: BERT in one line as LaTeX (Equations has no composition, so by hand), one head's attention as Equations calls (scores, the softmax written out, the weighted values; GELU, the layer and its repeats outside the strings: E6, E3, E2), one encoder layer as a Flow pipeline (`$h`, `$a` for the residuals, `[n 16]` contracts) applied L times by `p_ower` carrying the layer number, and the layer written out; checked: the equations are head 1's, the pipeline is the written-out layer, the scores and every head's attention are the Rust model's within 1e-9; six blanks filled. The page: a name with a blank, the eight likeliest fills, each layer's heads (both ways), the model drawn by KaTeX, all code shown with the pipeline highlighted; 7 native tests (the blank as the Rust model scores it to 1e-9, heads not causal, the math, beating the guess); checked in Chrome |
| 33 | equation-pipelines | DONE (2026-10-27, the user: the pipeline demos lacked a math-like tensor pipeline of equations; a pipeline drawn as LaTeX; a BERT layer example): Flow: a stage that calls a macro (an Equations e:e_q< in a raw code string c#"..."#) counts the names in its strings too, `it` among them (the stage then gets the previous value as it), Equations still checking each; `flow:m_ath<` draws a pipeline as math by the same reading: an aligned display of a row a stage, an arrow before each stage after the data, each label after the stage it names (a triangle as X_eTaL draws $, with its shape), an equation drawn by its library's t_ex< on the same strings (a macro's expansion may call another macro: checked first), `it` under the name the value came with, code as \texttt. Tests first: Flow's equations.xtl (a dense layer and ReLU as equations, a softmax by row with code and equations mixed, both against the written-out forms and drawn), a refusal (an equation naming none of the pipeline's values), the doc example, every expansion pinned; the old refusal's wording rebased. The pipelines demo: equations.xtl, an attention head as a pipeline of equations (fan-out from the data's label, fan-in at the scores and the weighted values) checked against at:h_ead, and a BERT encoder layer with made-up weights checked against the layer written out, both drawn; the page's two new examples (the drawing panel, a sentence each, the BERT layer linking to Tiny BERT), a test that each drawing reads the text its pipeline computes. Asked by the user next: an equation as a paper writes it (inside out) rewritten as a pipeline of equations, with each part highlighted across the equation, the pipeline, the macro's input and output and a 3D view of each stage's tensors |
| 34 | bert-layer-math | DONE (2026-10-27, the user: the BERT layer drawn as math showed X_eTaL code as typed, not math; then: every example on the Pipelines page shown as math): the BERT layer rewritten with every stage an equation (one head of attention, the output projection and residual, layer norm as mean, variance and normalized, the dense layers with their bias rows, GELU written with exp as f / (1 + exp(-2u)), the second residual and layer norm), equal to the layer written out with the libraries; every other example gains a pipeline of equations checked equal to its own pipeline and drawn: the CNN layer (the convolution with its bias, ReLU), the diffusion sampler (the telescoped sum, the decoder), the dense layer, the softmax (exp(z_j - max_i z_i) over the sum), layer norm, the causal attention head (the mask multiplying each exponential); the page's panel is As math for all eight; a test that every example draws one pipeline of equations, every stage an equation and no code, from the text of a pipeline it computes; it caught the page taking eqpipe's expansion for pipe's (the name now matched at a line's start). Flow's page says a code stage is drawn as code |

### Combined with X_eTaL-gpu (2026-10-12)

The user's direction: the ML work tested, measured and shown on real
GPUs through X_eTaL-gpu (OpenCL; Apple silicon now, the Arch
machine's NVIDIA cards later). Only at the command line: the pages
run X_eTaL as WebAssembly, which cannot reach OpenCL. This repo
writes no GPU code; it gives each model as a batch X_eTaL program,
its weights in `data/` and its expected outputs, and X_eTaL-gpu runs
it on its devices (asks in [`gpu-asks.md`](gpu-asks.md)). The first
three need only operations its IR already has (inputs, constants,
elementwise maps with `max`, `exp`, `div`, reductions along an axis,
matrix products, `take`, `drop`, `ravel`, `table`, `select`):

| # | Workload | Here | On the GPU, it shows |
| - | -------- | ---- | -------------------- |
| 1 | world-model: 432 -> 64 ReLU -> 144 sigmoid, all 1,936 states of its world as one batch | step research: its program prints the number of next frames predicted exactly | the same Int on the evaluator and on every device; the batch's time, the device's and the bridge's (X_eTaL-gpu's saga 4 tiny-nn, made here) |
| 2 | the benchmarks: `bench/dense.xtl`, `inner-wide.xtl`, `softmax.xtl` | as they are (docs/speed.md) | a GPU column beside the evaluator's: M1 Max and M3 Pro, then each NVIDIA card |
| 3 | the learned denoiser: the same kind of network, 20 steps from noise, 256 pictures at once, the noise passed in | step research: a batch run with given noise, its pictures and a summary number | about 20 calls, not one: what each call's fixed cost does to a loop |
| 4 | cnn-digits over a batch: the first 1,000 MNIST test digits through the Conv library (every picture's windows, one product, the pooling) and the dense layer | step gpu-batches: `demos/cnn-digits/batch.xtl`, the reader one function of arrays, `data/test-digits.txt`; prints how many it reads right (971, the Rust model's count, checked) | the windows in the IR (GA3, landed): 6 million numbers and a product per batch; the count the same everywhere |
| 5 | micro-gpt over a batch of names: the first 1,000 names scored by the forward pass, a name per call under its causal mask | step gpu-batches: `demos/micro-gpt/batch.xtl`, `u:g_pt` as the demo has it, `data/names.txt` as token rows; the mean loss per token (2.2496) and the next letters right (2,098), the Rust model's (checked) | a thousand calls of a small forward pass, the token ids all that goes in (X_eTaL-gpu's saga 6 sweeps its width); transpose and the lookup in the IR (GA2, landed) |

Later, once the batch programs exist: ternary-net and moe-router
(top-k by masks, which X_eTaL-gpu's IR now has), the Attention
library, then cnn-backprop's training. The tolerance:
the same decision always; numbers within 1e-5 in f32 (Apple's OpenCL
has no 64-bit floats), within 1e-9 in f64 against the trainers' Rust
models on cards with fp64. The results come back as X_eTaL-gpu's
table, linked from docs/speed.md (step gpu-results).

Results (2026-10-13, step gpu-results; the tables linked from
[`speed.md`](speed.md), "On GPUs"): rows 1 and 2 ran on the Apple
M1 Max and the RTX 5060 Ti, with the same decisions and the numbers
within the tolerance (f32 on both; f64 not yet exercised). The
kernels take a fraction of a millisecond; compiling them on every
run and the text bridge are the cost, and a kept session with the
weights resident does not beat the host's BLAS on a model this
small. GA2 and GA3 landed there the same day, so the later
workloads above no longer wait on the IR; what they wait on here is
a batch program each (cnn-digits runs one digit, micro-gpt one name:
a batch over the MNIST test set and over many names would be new
programs, a candidate step), and on the bridge (GA4, GA5) for the
models whose libraries use `h:` and tuples. Row 3 is queued there.

Since 2026-10-13 (the user's note; step gpu-source) X_eTaL-gpu
builds the lowering itself, provisionally (its A4b: X_eTaL source
to its IR through the pinned X_eTaL's crates; `xetal-gpu run
FILE.xtl` evaluates a program's statements by the pinned xetal up
to the accelerated function; a macro `gp:a_ccel<` hands a
function's text to its extension), and its step models-from-source
runs this repository's world model and benchmarks from their own
`.xtl`, the generated twins deleted. So what a model needs here is
one function of arrays in the subset it lowers, the rest (loops,
tuples, choices, printing) left to the evaluator:
[`gpu-asks.md`](gpu-asks.md) lists the subset and where each of our
programs stands (the world model's `u:p_redict` and `u:e_xact` and
the benchmarks fit; the denoiser's `c_at_2`, `o_neHot` and `^` did
not: ask GA7, and the denoiser's network split out as `u:n_et`, one
function of arrays in the subset, step denoiser-split). The later
workloads above follow the same rule when their batch programs are
written.

Also queued there the same day (its step gpu-pipeline; the user:
throughput over response time): pinned f32 input buffers and
non-blocking transfers, the batch axis chunked and run on streams
with events from the IR's dependencies (chunk i uploads while chunk
i-1 computes and i-2 downloads), `explain` showing the stages, and
`--repeat` reporting sustained items per second, measured first on
the world model against OpenBLAS and Accelerate. What our two batch
workloads offer that: the world model is a stream of independent
batches (any number of states; its core generates the windows, and
the weights stay resident), the plain case for chunking; the
denoiser is ten dependent stages per batch (step t needs step t-1's
pictures), so within one batch only the chunks overlap, but a stream
of batches pipelines stage by stage (batch k at step t while batch
k+1 is at step t-1), the case for parallel stages; both keep their
decision (the counts; the same pictures) whatever the chunking. A
program here need not change for it: the batch is an input, and
how it is cut is the schedule's.

## Cross-cutting

- Move the X_eTaL pin (`just xetal-pin`) at the start of a saga, or
  when an ask in `docs/xetal-asks.md` has landed upstream; never in
  the middle of a step. `XETAL_COMMIT` changes in its own commit,
  with the baselines and `just bench-check` re-run.
- When an ask lands, remove the workaround in the step that moves the
  pin, and mark the ask landed.
- Tags: `v0.1.0` (2026-10-05, the user's request for the linked release of the X_eTaL repositories, as X_eTaL-games, X_eTaL-demos and X_eTaL-extensions have) marks the state after the history rewrite: four live demos, NN, X_eTaL abb8274. `v0.1.1` (2026-10-05) is the same repository built against X_eTaL v0.1.0 (512b3ee), 4 to 6 times faster. Versions are independent per repository; each tag's notes name the X_eTaL it is built against.
- Ports: 8435 is this repo's (the user's choice, 2026-10-04: one port
  per X_eTaL repo, so a demo from each can run at once): `just serve
  SLUG` and `just serve-pages` default to it; the headless checks pick
  a free port.
- The split: once ternary-net, moe-router and cnn-digits are pushed
  and live here (saga 1 steps 4 and 5), the user deletes them from
  X_eTaL-demos (the user's decision, 2026-10-03); X_eTaL-demos then
  links here.

## Settled questions

- `docs/research.txt`: none is needed here (the user, 2026-10-03);
  the sources are `../X_eTaL/docs/research3.txt` and `research4.txt`
  and X_eTaL-demos' research.txt.
- Live or recorded: A13. Libraries from X_eTaL-libraries: A14.
- Where mathematics lives (the user, 2026-10-23, planning Activations):
  a builtin of X_eTaL is what libm gives at full accuracy and a
  library cannot write as well (`s_in`, `c_os`, `a_tan`, `e_xp`,
  `l_og` today; `t_anh`, `e_rf`, `s_qrt`, `e_xpm1`, `l_og1p` asked,
  M19); what is derivable from them in closed form is a library in
  X_eTaL-libraries (Trig, E4), and an approximation of a function
  without a closed form is a library until its builtin exists
  (Special, E5), wrapping the builtin from then on. Nothing is
  duplicated, nothing leaves the base language; a breaking change
  would be allowed (no users yet) and is not needed. The ML
  activations are this repository's (Activations, `act:`), their
  mathematics X_eTaL-libraries', the notation Equations' (E6 to E9).

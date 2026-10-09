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
| embedding-explorer | 64-dimensional embeddings projected to a rotatable 3-D cloud by PCA | covariance, power iteration, transpose | X_eTaL-demos (deferred there) | 4 (post-launch) |
| gradient-descent | a tiny model trained in X_eTaL itself: the loss surface, the path, the update as one expression | outer product, reduce, iteration | new | 4 (post-launch) |
| micro-gpt | microgpt's forward pass (embedding, RMSNorm, attention, MLP) on weights trained offline; sample names | everything above | microgpt-mlpl (port) | 4 (post-launch) |
| net-macro | a network written as `"..." net:n_etwork<` and its expansion beside it; three networks on a spiral; any spec typed | `.xtlm` macros, `xetal expand` | new | 3 (live) |
| world-model | a ball's next frame predicted from the last three | recurrence, prediction | X_eTaL-demos (deferred) | 4 (post-launch) |
| diffusion | noise to image, step by step | tensor transforms, iteration | X_eTaL-demos (deferred) | 4 (post-launch) |

## The libraries

Ranked by how many demos need them. Aliases are recommendations; the
alias is the importer's choice.

| Library | Alias | What | Taken from | Saga |
| ------- | ----- | ---- | ---------- | ---- |
| NN | `nn:` | activations (ReLU, leaky ReLU, sigmoid, tanh), softmax and log-softmax by row, a dense layer, argmax by row, one-hot, cross-entropy, accuracy | ternary-net, moe-router, cnn-digits | 1 |
| Quant | `qz:` | FP16 rounding, INT8 symmetric quantize and dequantize, ternary (BitNet b1.58 absmean) weights, storage bits, quantization error | ternary-net | 4 (post-launch) |
| Conv | `cv:` | windows by rotation, 2-D convolution of several filters, max- and average-pooling by reshape, padding | cnn-digits, X_eTaL-demos image-pipeline | 4 (post-launch) |
| Attention | `at:` | scaled dot-product attention, causal mask, masked softmax, heads split and merged | research (attention) | 4 (post-launch; the attention demo keeps its definitions) |
| Norm | `nm:` | layer norm, RMSNorm, standardize columns, batch statistics | microgpt | 4 (post-launch) |
| Embed | `em:` | centering, covariance, power iteration for the top components, PCA projection, cosine similarity, nearest neighbors | research (embedding explorer) | 4 (post-launch) |
| Sample | `sm:` | temperature, top-k and top-p filtering, sampling from a distribution with `r_oll` | microgpt | 4 (post-launch) |
| Optim | `op:` | hand-derived gradients for linear and logistic regression and a dense layer, SGD and momentum steps, a training loop by `p_ower` | new | 4 (post-launch) |
| Net (`.xtlm`) | `net:` | `"784 128 relu 10 softmax" net:n_etwork<` -> a forward function built from `nn:` calls; `net:p_arams<`, `net:s_hapes<` | research3.txt | 3 |

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
| 4b | char-builtins | inserted: re-pin X_eTaL for `[]A` and `[]D` (step 090) and the doc tooling fixes; replace digit and alphabet literals |
| 5 | classic-ml | a `fit`/`pred` library after APLearn's list (k-means, kNN, PCA, logistic regression) and a k-means demo whose clusters move each iteration |

Left out for now: MENACE (it fits X_eTaL-games, and X_eTaL ships
TTTML, which learns tic-tac-toe by self-play); a U-Net (a tiny one on
synthetic shapes is a later stretch).

## Saga 5 -- post-launch

The follow-up stream after the launch (research4: material for after
the announcement), in this order:

| # | Step slug | Delivers |
| - | --------- | -------- |
| 1 | quant | the Quant library (`qz:`); ternary-net moved onto it, baselines unchanged |
| 2 | conv | Conv (`cv:`); cnn-digits moved onto it |
| 3 | attention-lib | Attention (`at:`); the attention demo keeps its lesson |
| 4 | norm, sample | Norm (`nm:`), Sample (`sm:`) |
| 5 | micro-gpt | microgpt inference in X_eTaL on offline-trained weights (port from microgpt-mlpl) |
| 6 | embed, embedding-explorer | Embed (`em:`); PCA from 64 dimensions to a rotatable 3-D cloud (PCA may come with saga 4's classic-ml) |
| 7 | optim | Optim (`op:`), what saga 4 leaves of it |
| 8 | lib-site | the libraries in the live site, as X_eTaL-libraries' site |
| 9 | world-model, diffusion | the deferred research demos (training speed, a learned denoiser) |

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

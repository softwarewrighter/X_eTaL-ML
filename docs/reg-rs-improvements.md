# reg-rs baselines: an audit and the improvements it asks for

An audit of this repository's reg-rs baselines (200 at 1ffdc20,
2026-11-02: the libraries' `libs/*/tests/` and the demos' `reg/`),
asked for by the user: do they test something, or are some contrived,
going through the motions, or of little value? This file lists what was
found and the fix for each item. Nothing here has been changed yet.

## How it was audited

- **Read review.** Every library test program and golden, every demo's
  `cli-`, `browser-` and `expand-` golden, and the pinned types were
  read; small results were recomputed by hand.
- **Mechanical checks.** Over all baselines: empty or tiny output,
  masked commands, FAIL lines, identical commands and outputs.
- **Mutation testing.** For each top-level definition of each library
  (`libs/<Name>/src`, up to 30 a file), one small change that still
  type-checks was made (`+` to `-`, `<` to `<=`, `m_ax` to `m_in`, a
  literal plus one, ...), and the library's baselines were run
  (`scripts/test-libs.sh <Name>`): a mutant no baseline notices
  "survives". 104 mutants were run; Activations was skipped because its
  baselines already fail here (below).
- Claims marked (verified) were rerun by hand.

## Verdict

The tests are mostly real. Each library has a `checks.xtl` of real
oracles: finite-difference gradients (Activations, Conv, Net,
backprop), independent Rust forward passes (cnn-digits, micro-gpt,
micro-bert, world-model, diffusion), convergence orders (Integrals),
hand-written equivalents (Flow, Net). Hand spot checks all matched
(NN's softmax and cross-entropy 0.16425, SELU(-1) = -1.1113, Norm's
-1.069, Quant's INT8 0.9094, the Adam step, PCA variances 40.61 and
0.333, Integrals' KL 0.443147, ...). The training demos really learn
(train-live loss 1.319 to 0.062; cnn-digits 971 of 1000). No baseline
pins a FAIL, NaN or unexpected error.

Mutation testing killed 96 of 104 mutants. The eight survivors and the
read review point at the same few gaps.

## 1. Bugs found

| Where | What | Fix |
| ----- | ---- | --- |
| `libs/Sample/src/Sample.xtl` `l:t_op` (verified) | ties above the k-th place count as one: `2 sm:t_op 0.4 0.4 0.2` keeps all three (`0.4 0.4 0.2`) where the top 2 is `0.5 0.5 0.0`. basics' tie is at the k-th place, so it cannot see this | add the case to `tests/checks.xtl`, then count tied values above the k-th correctly |
| Integrals' LaTeX, `tests/latex.out`, `expand-latex`, `demo-convolution`, `demo-gelu` (verified) | the body is put in math mode as written: `\int_{0}^{1} e_xp tau \, d\tau` renders `e` with a subscript `x` (KaTeX accepts it, so nothing failed); `p_i @` shows a literal `@` | set the body as `\texttt{...}` with escapes, as Flow and Equations do, and check the output with KaTeX |

## 2. Tests that no longer reach the rule they name

- `libs/Flow/tests/reject-bad-name`, `reject-label-alone`,
  `reject-label-apart` (and their `expand-` copies) are refused by the
  X_eTaL lexer since the pin at f7848d00 (`ambiguous-minus`,
  `bad-label`): Flow's own rules for a `$` that does not touch its name,
  a bare `$`, and a plain name outside a tuple have no test. Plain
  strings still reach Flow's reader (differential c14/c15), so rewrite
  the three with plain strings (`@ flow:p_ipe< "x $ m |> n_eg"`) and keep
  one `c"..."` case as the lexer-level test.
- Flow's "a shape [N O] goes after the stage's name" rule: mutating its
  condition (`0 < t_ally b` to `<=`) survived. `reject-shape-no-name`
  (`x |> n_eg [4]`) is refused earlier, by the stage rule, so the shape
  rule is never reached. It looks unreachable: a shape is read only
  after a `$`, and `x $ [4]` (in a plain string) is refused first by
  the rule that a `$` touches its name. Remove the dead rule, or find an
  input that reaches it and pin that.

## 3. Checks too weak to catch a broken function

| Where | Why it is weak | Fix |
| ----- | -------------- | --- |
| `NN.b_ias`, rank 2 (mutant survived) | every rank-2 test has a square result (2 x 2 with a bias of 2), so adding along the wrong axis passes the length check and the values | add a non-square case, e.g. a 2 x 3 result with a bias of 3 |
| `Learn.k_nn` self-exclusion, `Learn.k_step` empty cluster (mutants survived) | no test queries a point of the training set, or leaves a cluster empty | add both cases with hand-worked answers |
| Integrals' reader (`h:o_ption`'s empty grid; `h:o_utside` brackets; mutants survived) | no refusal of an empty step count; no `|` inside brackets in a body | add both |
| `libs/Optim/tests/checks.xtl` c3 | `a_damWith` is compared with `a_dam` only at t = 1 from m = v = 0, where the betas cancel; swapped betas pass | several steps with non-default betas, against hand-worked values |
| Norm `r_ms` gain | the gain is always 1 (basics, checks c5, the demo, micro-gpt) | pin `2.0 0.5 1.0 3.0 nm:r_ms m` |
| `libs/Learn/tests/checks.xtl` c10 | "the mean projects to the origin" holds for any projection; `p_roject` is pinned nowhere else | check the projection's variance along each axis against `ev` |
| `libs/Embed/tests/checks.xtl` c4 | calls `8 em:p_roject P` twice in one run: true by construction | drop it |
| `libs/Integrals/tests/boxes.xtl` c4 | plain arithmetic, no library call | drop it |
| Learn c12, c4; Conv c3; NN c14, c16 | shape or count only, or a perfect-guess zero | minor: basics covers the values |

## 4. Tautological: the expected value repeats the formula

- `libs/Attention/tests/checks.xtl` c5, c6 (the scores formula retyped).
- `libs/Activations/tests/checks.xtl` c13, c13b, c13c, c14 (the gated
  layers rewritten as the library writes them).
- `libs/Quant/tests/checks.xtl` c7 (`s_cale`'s own formula).
- `libs/Optim/tests/checks.xtl` c1, c2 (train-live's Adam, the same
  expression character for character).

Fix: replace each with values worked out by hand on a small case.
(Flow's and Net's comparisons with hand-written code are not
tautological: for a macro, the hand-written expansion is the oracle.)

## 5. Bare 1 and 0, and browser goldens of static text

- The demos' oracle checks print a bare `1` (cnn-digits batch,
  micro-gpt and its batch, world-model, diffusion, backprop,
  cnn-backprop), and `scripts/test-demos.sh` does not look for FAIL
  lines (`test-libs.sh` does). Re-recording would accept a `0`
  silently. Fix: use Check (`ok:` / `FAIL:`) and scan demo output for
  FAIL as test-libs does.
- Most `browser-*` goldens list only panel titles from `web/src/app.rs`
  (k-means, cnn-backprop, embedding-explorer, train-live, world-model;
  micro-bert, micro-gpt, diffusion and activations add one label), so a
  page computing wrong numbers still passes. Good ones show a computed
  result: backprop (`: the gradients agree.`), moe-router, net-macro,
  ternary-net, pipelines. Fix: one computed number per page in
  `web/browser.txt`.
- `demos/activations/reg/cli-activations` pins only the drawn files and
  each chart's minimum and maximum, through `| sed`, so the exit code is
  sed's.
- `demos/_template/reg/cli-__SLUG__` (output `55`) is a scaffold the
  tests skip.

## 6. Redundant

- 33 `expand-*` refusal goldens are byte-identical to their run twins:
  Flow 17, Integrals 11, Net 5. The refusal happens at expansion either
  way. Fix: in `scripts/test-libs.sh`, skip the `expand-` baseline when
  the program fails at expansion.
- The XOR forward pass is pinned three times (`libs/NN/tests/demo-xor`,
  `libs/Net/tests/demo-xor`, the first lines of `libs/Net/tests/basics`).
- Flow's `bias` and `bias-wrong` re-test NN's `b_ias` through one-stage
  pipelines (different programs, low marginal value).

## 7. Brittle: rebased without a behavior change

- **Float noise pinned to the last digit.** `libs/Activations/tests/basics`
  fails on Linux (verified: `-0.1588080093917233` against
  `-0.15880800939172324`, and three more), though it passes where it
  was recorded: the 17th digit depends on the platform's math library.
  Round printed values, or compare with `k:n_ear`.
- `libs/Flow/tests/bias-wrong.err` and `libs/NN/tests/bias-wrong.err`
  pin `../../NN/src/NN.xtl:87:31` and NN's source line: any edit above
  line 87 of NN rebases a Flow golden. Pin the message line only.
- `demos/pipelines/reg/expand-*` are 2,197 lines (100 KB), mostly the
  Equations library's expansion; Flow's `expand-*` pin gensym numbers
  (`g1:stage`). Every pin move rebases them. Keep one representative
  expansion per macro, or normalize `gN:` before comparing.
- `XETAL_BLESS=1` rebases every baseline of a library at once; the
  bare `1`s above make that riskier.

## 8. Thin coverage

Every `l:` and `m:` export appears in some test. Effectively untested:
Optim `a_damWith` with other betas, Norm's `r_ms` gain, Learn's
`p_roject`, Sample's top-k ties, Flow's `$` rules, Embed's `c_osines`
(one 4 x 2 pin), Quant's `c_ut` (doc example and ternary-net only),
Conv's `u_npool` with tied maxima, Integrals' `t_ex<` refusals.

## Library and demo verdicts

| Library | Verdict |
| ------- | ------- |
| Activations | strong: every derivative against central differences, closed forms, SELU's fixed point; four tautological checks |
| Attention | basics hand-checkable, real properties; c5/c6 tautological |
| Conv | strong: an independent per-picture formula, finite differences |
| Embed | good counts and neighbors; c4 vacuous |
| Flow | strong differential and oracle tests; three refusals now reach only the lexer, the shape-after-name rule unreached |
| Integrals | strong exactness and convergence orders; LaTeX malformed |
| Learn | good; `p_roject`, `k_nn` self-exclusion, empty clusters untested |
| NN | good, values hand-verified; rank-2 bias only square |
| Net | strong: finite differences, training equal to hand-written, hygiene |
| Norm | good; `r_ms` gain untested |
| Optim | convergence real; `a_damWith` check vacuous, c1/c2 tautological |
| Quant | good; c7 tautological |
| Sample | good draws; top-k ties broken and untested |

Demos: strong oracles in backprop, cnn-backprop, cnn-digits, micro-bert,
micro-gpt, world-model, pipelines; qualitative but meaningful in
attention, embedding-explorer, k-means; pinned without an oracle in
moe-router and ternary-net; weak in activations (spans only).

## Suggested order

1. The two bugs (section 1) and the unreached rules (section 2).
2. The weak checks the mutants found (section 3, first four rows).
3. Turn bare 1s into Check calls and scan demo output for FAIL
   (section 5).
4. Drop the 33 duplicate expansion goldens and normalize float noise
   and locations (sections 6, 7).
5. The rest as the files are next touched.

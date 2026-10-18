# The pipeline macro: what to build, as decided

A hand-off from the X_eTaL agent (2026-10-17) for the agent that builds
the pipeline macro here. The design was decided with the user in
X_eTaL (decision D155; `../X_eTaL/docs/spatial.md`, the "Pipelines"
section; the alternatives weighed in `../X_eTaL/docs/eq-rationale.md`,
"Pipelines: the choices"). This page says what the macro does, what it
refuses, what must be tested, and what to read first. Nothing below
needs a compiler change; it is a library on the language as it is.

## Why here

A pipeline writes ML data flow left to right (X_eTaL reads right to
left; a paper's equation reads inside out): the initial data, then
each transform in turn. Its stages are ML functions, `nn:d_ense`,
`nn:r_elu`, `nn:s_oftmax`, which live in this repository, so the macro
lives beside them and its demos can be the real ones. The equation
macro (`e:e_q<`, X_eTaL-libraries' `Equations`) is its companion: a
stage may be an equation.

## The call

```
Y := @ e:p_ipe< r"
      X
   |> W1 nn:d_ense it
   |> nn:r_elu
   |> W2 nn:d_ense it
   |> nn:s_oftmax"
```

One line works as well: `@ e:p_ipe< "X |> W1 nn:d_ense it |> nn:r_elu"`.
The raw string (`r"..."`, X_eTaL ST4, in X_eTaL since 823adbd3) is
what lets a pipeline be written one stage per line.

- **The left side is `@`.** The whole flow is in the right string;
  the first stage is the data, any X_eTaL expression.
- **Stages are separated by `|>`**, split only outside brackets,
  braces and strings. `|>` has no meaning in X_eTaL, so a stage never
  contains one.
- **A stage is ordinary X_eTaL**, type-checked like any code and copied
  into the expansion as written (so an error in it is reported inside
  the string, by the macro phase's own mapping). It is one of two
  things:
  - a **function** (a name, a train, a quoted lambda: `nn:r_elu`,
    `[f_ g_]`, `'{ x -> x * 2 }`), applied to the previous value;
  - an **expression naming the previous value `it`**: `W1 nn:d_ense it`,
    `it - m`, `m - it`. Noncommutative order is written, never guessed.
  `it` is an anaphor the macro declares (`## binds: it`, X_eTaL MC30),
  so a stage cannot bind an `it` of its own.
- **` : name`** at a stage's end (spaces around the colon, so it is
  never a namespace prefix) names the stage's value; later stages use
  the name directly, so fan-out and joins need nothing more:
  `z |> 'm_ax r_/ it : m |> e_xp z - m : e |> e / '+ r_/ e`. A tuple
  ` : (Q, K, V)` destructures. Names are local to the pipeline; each
  stage is evaluated once.
- **` [N O Ho Wo]`** after a name is an optional shape annotation, a
  contract: the macro emits a run-time check after the stage, failing
  as `error[shape]: stage 3 (conv): shape 2 8 6 6, expected 2 8 4 4`
  (the extents are the caller's values). A shape lint that proves or
  refutes the annotation before the program runs is X_eTaL's work
  (its step shape-lint, `../X_eTaL/docs/type-check-macros.md`
  approach 8); the run-time check stays as the backstop.
- **The value** of the call is the last stage's value.

## What the macro refuses (at compile time, at the argument)

Every rule of the reader must use all of its text or refuse it (the
rule X_eTaL-libraries' Equations learned the hard way: a reader that
uses part of its input and drops the rest gives silent wrong answers;
`../X_eTaL/docs/type-check-macros.md`, "The case that taught the
class"). Refuse, with `[]R_EJECT` and a message naming the text:

- an empty stage (`a |> |> b`), or an empty call;
- a stage that is a value without `it` after the first (`X |> 3`): it
  can be neither applied nor joined;
- `it` in the first stage (nothing precedes it);
- a name bound twice (` : m` twice, or ` : it`);
- a name after the colon that is not a plain name, or a tuple with a
  non-name part;
- a shape annotation on a stage without a name, or malformed brackets;
- anything else the grammar does not know, rather than passing it on.

Rank and shape mismatches that only the arrays can tell are run-time
errors from the annotation check, never silent (no implicit
broadcasting: adding a bias vector to every row is written out, see
the bias helper below).

## The expansion

Ordinary X_eTaL, laid out as a program (one statement per line; X_eTaL
823adbd3's `xetal expand` and the live demo's Expand also reflow long
expansions, D159): a block that binds each stage's value to a fresh
name from `[]G_ENSYM` (or the given name), one line per stage, the
annotation checks after their stages, and the last value as the
result. A stage that is a function `f` becomes `f <previous>`; a stage
with `it` becomes its text with `it` bound to the previous value (bind
`it` to a fresh name in the block, or write the stage as a lambda
applied to the previous value). Read `Equations.xtlm` in
X_eTaL-libraries for the working pattern: `h:s_plit` outside brackets,
`[]R_EJECT` at the argument with the place `"bad-pipeline right"`,
`[]G_ENSYM` for the block's own names, and the layout with newlines.

## The expected outputs (from X_eTaL's oracle)

`../X_eTaL/docs/spatial-poc/pipes-oracle.xtl` holds the hand-written
equivalents; each pipeline must print exactly what its block prints.
With `X := 2 3 r_eshape 1 2 3 4 5 6`, `W := 3 2 r_eshape 1 0 0 1 1 1`,
`b := 10 20`, `z := f_loat 1 2 3`, `x := f_loat 1 2 3 4`,
`W1 := 3 2 r_eshape 1 -1 0 1 -1 0`, `W2 := 2 1 r_eshape 1 2`:

| Pipeline (one line) | Expected |
| ------------------- | -------- |
| `X \|> it '+ '* i_nner W : P \|> P + (s_hape P) r_eshape b` | `14 25` / `20 31` |
| `z \|> 'm_ax r_/ it : m \|> e_xp z - m : e \|> e / '+ r_/ e` | `0.0900... 0.2447... 0.6652...` |
| `x \|> ('+ r_/ it) / f_loat t_ally it : mu \|> x - mu : c \|> ('+ r_/ c * c) / f_loat t_ally c : v \|> c / v ^ 0.5` | `-1.3416... -0.4472... 0.4472... 1.3416...` |
| `X \|> it '+ '* i_nner W1 \|> 0 m_ax it \|> it '+ '* i_nner W2` | `2` / `2` |
| `x : s \|> 2 * it \|> it + s` | `3.0 6.0 9.0 12.0` |

The residual shows why the direction matters: by hand, `2 * x + x` is
`2 * (x + x)`. Add the real ones with this repository's libraries: the
dense layer with its bias, the stable softmax, layer norm with
`nn:` functions, attention (Q, K, V as a tuple name).

## The bias helper (decided with the user)

X_eTaL has no implicit broadcasting, and the APL family agrees with it
(J and BQN match a vector against the leading axis, NumPy against the
trailing one: for a square matrix both give an answer, and they
differ). Adding a bias to every row is therefore written out today
(`(s_hape P) r_eshape b`). Build a checked helper, `a_ddAxis` (the
value, the axis, the vector; the vector's length must equal that axis's
extent, else an error naming both; this repository's names underline
the first letter only, as `o_neHot`, so not `a_dd_axis`), and wrap it as `nn:b_ias` with a
documented axis convention; the channel-first CNN tensor `[O Ho Wo]`
takes its bias on the first axis, the dense layer on the last. A
general rank operator is deferred until several uses justify it.

## Tests, first

- Every example above, one-line and raw-string form, equal to the
  oracle's output (goldens).
- Each refusal above as a rejection golden.
- An expansion golden per demo (`xetal expand`), so the generated code
  is reviewed and pinned.
- The reader's own class of error: a round-trip test (the parse
  written back and compared with the input) and a differential test
  (random short pipelines run through the macro and as plain X_eTaL).
- The run-time shape check: a test that fails it, with the message
  naming the stage, its name and both shapes.

## Naming

Plain words that read under any alias; no puns, nothing picked for
being short and clever (the user's rule, 2026-10-17, after `e:q_<`
became `e:e_q<`). `p_ipe<` is decided; name the helpers the same way.

## Pins and prerequisites

- X_eTaL at or past `823adbd3` (raw strings ST4, `[]G_ENSYM`, hygiene
  by choice MC30, names bound once M1, the laid-out `xetal expand`).
- X_eTaL-libraries at or past `ed91e6d` if a stage is an equation
  (`e:e_q<` with its whole-text reader).
- Read first: `../X_eTaL/docs/spatial.md` (Pipelines),
  `../X_eTaL/docs/eq-rationale.md` (Pipelines: the choices),
  `../X_eTaL/docs/type-check-macros.md` (what a macro's reader must
  guarantee), `../X_eTaL/docs/lang-choices.md` MC10 to MC30 (macros)
  and ST4 (raw strings), and `libs/Equations/src/Equations.xtlm` in
  X_eTaL-libraries as the model.

## Report back

What the language or the equation library should do differently goes
in `docs/xetal-asks.md` here, as every ask does; X_eTaL's plan keeps a
step for the pipeline macro (its 109, spatial-pipeline-macro) that
will be closed with a pointer to this repository's commits.

## As built (2026-10-17, step pipeline-macro)

`libs/Pipeline/` (`"pipe:" u_se< "Pipeline"`, so the call is
`@ pipe:p_ipe< "..."`; the `e:` of the examples above was the alias of
the spatial library the design sat in). What the building settled:

- A stage after the first may also be an **expression of names the
  pipeline bound before it** (`e / '+ r_/ e`, `P + (s_hape P) r_eshape b`):
  the oracle's own examples need it. A value stage (`X |> 3`) is still
  refused: not a function, no `it`, no earlier name.
- A quoted function or lambda stage (`'n_eg`, `'{ v -> v * 2.0 }`) is
  applied without its quote; a derived function (`'+ r_/`) and a train
  stay as written.
- An `it` stage becomes a block of its own, `name := { @ -> it := <previous>; <stage> } @`
  (a lambda `{ it -> ... }` cannot take a tuple, which a ` : (Q, K, V)`
  stage leaves as the previous value).
- A shape goes with one name; on a tuple it is refused (each part would
  have its own).
- `pipe:t_ext<` writes the pipeline back from the same reading, for the
  round-trip test; the differential test enumerates every kind of stage
  and their orders against plain X_eTaL.
- The names: `Pipeline`, `pipe:`, `p_ipe<`, `t_ext<`; the bias helper is
  `nn:a_ddAxis` and `nn:b_ias` (step pipeline-bias, the same day): the
  linear regression is `X |> it '+ '* i_nner W |> b nn:b_ias it`.
- On the libraries' site a pipeline is drawn as typed: it is a string
  to the renderer (ask M17). X_eTaL answered with its step 111, code
  strings (`c"..."`, `cr"..."`): at the re-pin the pipelines are written
  `cr"..."` and render as code; the macro is unchanged (the same Char
  vector). Its step 110's `## uses: parsed` line will declare the reader.

# Flow

ML data flow written left to right, `|>` between stages. One macro
call turns the pipeline into ordinary X_eTaL, one line per stage,
shown by `xetal expand`; intermediates are named, and a shape a stage
promises is checked when the program runs. A stage may be an equation
(X_eTaL-libraries' `e:e_q<` is just a value).

```
"flow:" u_se< "Flow"
```

Put the library's `src/` on `XETAL_PATH` (`just path` prints every
library's; see the [README](../../../README.md)); `flow:` is the
recommended alias. Designed in X_eTaL (its `docs/spatial.md`,
"Pipelines", decision D155) and built here
([`docs/pipeline-macro.md`](../../../docs/pipeline-macro.md)).

## The macros

| Macro | What |
| ----- | ---- |
| `@ flow:p_ipe< "X \|> W1 nn:d_ense it \|> nn:r_elu"` | the value of the pipeline: the data, then each stage in turn |
| `@ flow:t_ext< "..."` | the pipeline written back from the same reading, a string (blanks squeezed) |

The left side is `@`; the whole flow is in the right string. A raw
string (`r"..."`, X_eTaL ST4) puts one stage per line:

```
Y := @ flow:p_ipe< r"
   X
|> it nn:d_ense W1
|> nn:r_elu
|> it nn:d_ense W2
|> nn:s_oftmax"
```

- **Stages** are separated by `|>`, split only outside brackets,
  braces, parentheses and strings (`|>` means nothing in X_eTaL, so a
  stage never holds one). The first stage is the data, any X_eTaL
  expression.
- **A stage is ordinary X_eTaL**, type-checked like any code and
  copied into the expansion as written, so an error in it is reported
  inside the string. After the first, a stage is one of:
  - a **function**, applied to the previous value: a name
    (`nn:r_elu`, `t_ally`), a train (`[n_eg r_ev]`), a derived
    function (`'+ r_/`), or a quoted function or lambda (`'n_eg`,
    `'{ v -> v * 2.0 }`; the quote passes it, the macro applies it);
  - an **expression naming the previous value `it`**: `it nn:d_ense W1`,
    `it - m`, `m - it`. Noncommutative order is written, never guessed.
    `it` is an anaphor the macro declares (`## binds: it`, X_eTaL
    MC30), so a stage cannot bind an `it` of its own;
  - an **expression of names the pipeline bound before it**:
    `e / '+ r_/ e`, `P + (s_hape P) r_eshape b`.
  Anything else (`3`, `X`) is refused: it can be neither applied nor
  joined.
- **`$name`** at a stage's end, the `$` touching the name (X_eTaL's
  label, drawn with a small right triangle, U+25B9, as `|>` is drawn U+25B7), names the
  stage's value; `$ m` and a `$` with nothing after it are refused.
  Later stages use
  the name directly, so fan-out and joins need nothing more:
  `@ flow:p_ipe< "z |> 'm_ax r_/ it $m |> e_xp z - m $e |> e / '+ r_/ e"`. A tuple
  `$(Q, K, V)` takes a tuple apart; the previous value for the next
  stage is then the tuple (`(p, q) := it`). Names are plain names
  (letters and digits), local to the pipeline, each bound once; each
  stage is evaluated once.
- **` [N O Ho Wo]`** after a name is a shape the value must have, a
  contract: the macro writes a check after the stage, which fails as
  `error[shape]: stage 2 (y): shape 4, expected 3` with the caller's
  values for the extents (names or numbers). A shape lint that proves
  or refutes the annotation before the program runs is X_eTaL's work
  (its step shape-lint); the run-time check stays as the backstop.
- **The value** of the call is the last stage's.

## What is refused

Every rule of the reader uses all of its text or refuses it (the rule
X_eTaL-libraries' Equations learned the hard way: a reader that drops
part of its input gives silent wrong answers). The call fails when the
program is compiled, `error[bad-pipeline]` at the string, naming the
text:

- an empty call, or an empty stage (`a |> |> b`);
- a stage that is a value (`X |> 3`): not a function, no `it`, no
  earlier name;
- `it` in the first stage (nothing precedes it);
- a name bound twice, or `it` as a name;
- after the colon anything but a plain name or a tuple of two or more
  plain names (`m-1`, `(Q, 2)`, `(a, b) (c, d)`);
- a shape without a name before it, on a tuple, with unclosed brackets
  (`[4 |> n_eg`: the `|>` inside is not a separator), with anything
  after it (`[4] extra`), or with an extent that is not a name or a
  number (`[4;]`).

Rank and shape mismatches only the arrays can tell are run-time
errors from the annotation check, never silent. X_eTaL has no implicit
broadcasting: adding a bias vector to every row is written out
(`P + (s_hape P) r_eshape b`), or is the stage `b nn:b_ias it` (NN's
helper, the axis by convention or named, `(2, b) nn:a_ddAxis it`,
checked for length: `../tests/bias.xtl`).

## The expansion

Ordinary X_eTaL, laid out as a program: a block binding each stage's
value to a fresh name (`[]G_ENSYM`: `g1:stage`, ...) or the name
given, the shape checks after their stages, the last value as the
result. A function stage `f` becomes `f <previous>`; a stage naming
`it` becomes a block of its own with `it` bound to the previous value;
a stage of earlier names is copied as written. From `../demos/mlp.xtl`
(`just expand-lib Flow mlp`):

```
Y := ({ @ ->
  g1:stage := X
  g2:stage := { @ ->
    it := g1:stage
    it nn:d_ense W1
  } @
  g3:stage := nn:r_elu g2:stage
  g4:stage := { @ ->
    it := g3:stage
    it nn:d_ense W2
  } @
  g5:stage := nn:s_oftmax g4:stage
  g5:stage
} @)
```

On the libraries' site the pipeline itself is drawn as typed: it is a
string to the renderer, which decorates the expansion beside it (ask
M17 in `docs/xetal-asks.md`).

The expansion names no library: the stages carry their own calls
under the aliases the caller chose (a macro's text cannot name the
importer's alias, ask M11), and the checks use built-ins only.

## Examples

The five of X_eTaL's `docs/spatial.md`, each equal to its hand-written
equivalent in `docs/spatial-poc/pipes-oracle.xtl` there and
`../tests/oracle.xtl` here. With `X := 2 3 r_eshape 1 2 3 4 5 6`,
`W := 3 2 r_eshape 1 0 0 1 1 1`, `b := 10 20`, `z := f_loat 1 2 3`,
`x := f_loat 1 2 3 4`, `W1 := 3 2 r_eshape 1 -1 0 1 -1 0`,
`W2 := 2 1 r_eshape 1 2`:

```
      @ flow:p_ipe< "X |> it '+ '* i_nner W $P |> P + (s_hape P) r_eshape b"
14 25
20 31
      @ flow:p_ipe< "z |> 'm_ax r_/ it $m |> e_xp z - m $e |> e / '+ r_/ e"
0.09003057317038046 0.24472847105479764 0.6652409557748218
      @ flow:p_ipe< "x |> ('+ r_/ it) / f_loat t_ally it $mu |> x - mu $c |> ('+ r_/ c * c) / f_loat t_ally c $v |> c / v ^ 0.5"
-1.3416407864998738 -0.4472135954999579 0.4472135954999579 1.3416407864998738
      @ flow:p_ipe< "X |> it '+ '* i_nner W1 |> 0 m_ax it |> it '+ '* i_nner W2"
2
2
      @ flow:p_ipe< "x $s |> 2 * it |> it + s"
3.0 6.0 9.0 12.0
```

The residual shows why the direction matters: by hand, `2 * x + x` is
`2 * (x + x)` (right to left), not the residual.

A shape contract that fails (`../tests/shape-fails.xtl`, `n := 3`):

```
      @ flow:p_ipe< "x |> 2 * it $y [n] |> y + 1"
error[shape]: stage 2 (y): shape 4, expected 3
```

Text the grammar cannot use is refused (`../tests/reject-trailing.xtl`,
`../tests/reject-two-groups.xtl`):

```
      @ flow:p_ipe< "x $m [4] extra |> n_eg"
error[bad-pipeline]: a shape is [extents] at the stage's end, nothing after it: [4] extra
      @ flow:p_ipe< "x $(a, b) (c, d) |> n_eg"
error[bad-pipeline]: after $ comes a plain name, or a tuple of names (Q, K, V): (a, b) (c, d)
```

The pipeline read back (`../tests/roundtrip.xtl`): what the macro
read is what was written, blanks squeezed.

```
      @ flow:t_ext< "  x   $s |>  2 * it   |>  it + s "
x $s |> 2 * it |> it + s
```

## Tests

- `oracle.xtl`: the five examples, one-line and one stage per line,
  against the hand-written code.
- `differential.xtl`: every kind of stage and their orders (a function,
  `it` on either side of a dyad, an earlier name, a tuple taken apart
  and used as `it`, a quoted lambda, a quoted name, a train, a matrix,
  a shape that holds, the data alone) against plain X_eTaL.
- `roundtrip.xtl`: `flow:t_ext<` gives the pipeline back, `|>` inside
  brackets and strings included.
- `reject-*.xtl`: one golden per refusal above, the Equations cases
  among them (text after a stage, two groups side by side, an unknown
  character).
- `shape-fails.xtl`: the run-time check, with its message.
- `expand-*.rgt`: every program's expansion, pinned.

## Demos

- [`demos/mlp.xtl`](../demos/mlp.xtl): a two-layer network on XOR and
  a stable softmax as pipelines, each beside the same by hand; a shape
  contract that holds.

## Provenance

The pipeline notation was decided in X_eTaL from the user's proposal
(`docs/xetal-ml-pipelines-design.md` there; the alternatives in its
`docs/eq-rationale.md`, "Pipelines: the choices"): F#'s and Elixir's
`|>`, an anaphoric `it` as in Kotlin and Groovy, a label for
intermediates (first ` : name`, now `$name`, X_eTaL S14). The reader
follows X_eTaL-libraries' Equations: every
rule uses all of its text or refuses it, `[]R_EJECT` at the argument,
fresh names from `[]G_ENSYM`, the expansion laid out as a program.

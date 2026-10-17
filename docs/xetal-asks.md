# Asks for X_eTaL

Features the ML demos and libraries need that X_eTaL does not have
yet, and bugs they uncovered. This repo does not change X_eTaL: each
ask is filed here (and taken to `../X_eTaL`), the demo or library uses
the workaround noted below or waits, and the workaround is removed
when the ask lands in the X_eTaL commit this repo pins (`XETAL_COMMIT`).

Each entry: status, kind (feature, bug or speed), which demos or
libraries need it, why, a minimal repro or example, and the
workaround in use. The status is X_eTaL's own word for it, from its
ledger (`docs/asks.md` and `asks.toml` in X_eTaL, read from the pinned
copy in `work/xetal`): **landed**, **partly landed**, **planned**,
**declined** (with X_eTaL's reason: the workaround here is then
permanent unless that decision changes), **new** (taken to X_eTaL, not
yet triaged), or here only **not yet taken** (filed here, still to be
added to X_eTaL's ledger: see the last section). `scripts/check-asks.py`
(in the gate, `just check-asks`) checks that every ask here is in the
pinned ledger with the same status, or is marked not yet taken, and
that the ledger has none of ours this file lacks. Asks first filed by
a sibling repo are copied here with this repo's users named, so this
list stands on its own.

| # | Status | Kind | Ask | Demos, libraries | Workaround |
| - | ------ | ---- | --- | ---------------- | ---------- |
| M1 | landed | feature | `.xtlm` macro libraries: in X_eTaL v0.1.0 (`lib/Macros.xtlm`, `Combinators.xtlm`, the system macros of `System.xtlm`) | Net (`net:n_etwork<`), the net-macro demo | none left: saga 3 can start |
| M2 | landed | feature | `xetal expand FILE`: the program after macro expansion, in X_eTaL v0.1.0 | Net, net-macro | none left |
| M3 | landed | speed | `'+ '* i_nner` (matrix product) and `t_able`: fixed in X_eTaL v0.1.0; here the dense benchmark went from 804 ms to 145 ms and the CNN program from 1.5 s to 0.3 s (docs/speed.md) | ternary-net, cnn-digits, NN (`nn:d_ense`) | none needed; the pages keep their small maps |
| M4 | declined | feature | Grade per row (top-k along an axis): `g_rade_2 M` still grades the columns as items. X_eTaL declined it as `g_rade_2` (an axis subscript moves axis 2 to the front, so the columns are the items); a per-row grade, a rank form, is to decide with the user | moe-router, Learn (`k_nn`), Sample (`t_op`) | the row maximum as a mask, taken out, then the maximum again |
| M5 | planned | feature | `xetal-play`: arrays in and out without text, or a session kept between runs (v0.1.0 adds an interactive run that waits for typed lines, not arrays) | every page | programs written with literal arrays or data files; printed `r_avel` lines parsed |
| M6 | partly landed | speed | Whole-array arithmetic: fast enough for every demo here since abb8274; vector kernels are planned (X_eTaL's Saga 22) | all | none |
| M7 | landed | bug | Long Int literal strands read in quadratic time: fixed | moe-router (word numbers) | removed |
| M8 | landed | feature | Float literals with an exponent (`1.5e-5`) | NN (`1e-12`), attention (`1e9`) | none |
| M9 | landed | bug | A Bool bound to a name in arithmetic: works in X_eTaL v0.1.0 (`a := 1 2 > 0` then `1 * a` gives `1 1`) | masks | none left (no demo here carried the workaround) |
| M10 | declined | feature | `e_ach` returning an array per item (`'{ i -> i * 1 2 3 } e_ach 1 2` is refused: e_ach needs a single value from each call) X_eTaL declined it (B14: `e_ach` stays scalar-only; `m_ap` boxes each result); a mix to turn the boxes back into an array is to decide with the user | cnn-digits (the ten digits, one number per call), a per-row grade | one `e_ach` per number wanted |
| M11 | new | feature | A macro's text cannot name a library by the importer's alias: `m:f_` in an expansion is not rewritten to the alias the caller chose, and an expansion may not import (MC23), so a macro library cannot call its own `.xtl` half, or another library, without fixing the alias (X_eTaL-libraries X14) | Net (its networks call NN) | the expansion says `nn:`; the page tells the reader to import NN under that alias |
| M12 | declined | feature | Shapes in types: a matrix's type is `Float`, so code can be well-typed and shape-wrong (a spec saying 4 outputs against a 3-column weight array ran and answered); `r_eshape` repeats or cuts data silently. Asked in three sizes: a shape ascription built-in now, rank in types next, sized types as research X_eTaL declined it (T7: array types are rank-erased, shapes are checked at run time; T4: no annotations in v0) | Net (its models check at load instead), every demo | the Net macros write run-time checks that stop with an error naming the layer |
| M13 | landed | feature | A state of several arrays for `p_ower`: tuples with destructuring, then records (first asked by X_eTaL-demos; on X_eTaL's wish list as named records). Tuples through `p_ower` landed in X_eTaL 96060b5 and patterns in d284a8c, pinned here 2026-10-08: train-live's and Net's training state is a tuple taken apart by name (step tuple-state). Boxes already carry arrays of different shapes and one element type, by position; tuples would carry mixed types, by name | train-live (Adam's state), net-train-macro | none left |
| M14 | new | bug | `xetal_play::run` collects a run's `[]S_HOW` pictures in state shared across threads (`xetal_store::take_shown()`), so two runs at once take each other's pictures: a run can come back with none, or with another's | backprop, train-live, net-macro (their Rust tests, which cargo runs in parallel; a page runs one program at a time) | `microscope::run` runs one program at a time behind a lock |
| M15 | new | feature | `xetal doc --out` names every site "X_eTaL doc" (the page titles, the sidebar's heading): a sibling's site cannot say whose it is; asked: a `--title` option | the cross-reference (pages/doc) | none: the catalog, the cards and the pages' footers say what the site is |
| M16 | new | speed | The matrix product runs at about 85 ns a multiply-add: `(8 9 r_eshape ...) '+ '* i_nner 9 33800 r_eshape ...` and `(8 33800 r_eshape ...) '+ '* i_nner o_\ 9 33800 r_eshape ...` (2.4 million multiply-adds each, one with many short sums, one with few long ones) both take about 0.2 s, printed results included; the outer product `t_able` of 270,400 results takes 30 ms | cnn-backprop (the convolution and the filters' gradient are most of a step) | 600 digits and 20 a step, not more |
## Details

### M1, M2: macro libraries and seeing an expansion (landed)

X_eTaL v0.1.0 has `.xtlm` macro libraries (MC10 to MC30: a macro is a
function from the source text left and right of its call to new
source; hooks such as `[]R_EJECT` report a bad argument at the call;
hygiene is by choice: a macro's own names come from `[]G_ENSYM`, MC30)
and `xetal expand FILE`. `libs/Net` is built on
them: `"2 2 relu 2 softmax" net:n_etwork< "w1 w2"` becomes
`{ x -> nn:s_oftmax (nn:r_elu x nn:d_ense w1) nn:d_ense w2 }`, and its
tests pin every expansion.

### M11: naming a library from a macro's text

```
# T.xtl:   l:d_ouble := { x -> x * 2 }
# T.xtlm:  m:t_wice< := { @ e -> "m:d_ouble (" c_at e c_at ")" }
"zz:" u_se< "T"
@ zz:t_wice< "3 + 4"
error[unknown-namespace]: no library is imported as m: here
```

The macro's `m:` is rewritten at the call (`zz:t_wice<`) but not in the
text it gives, and the text may not import, so the only way to call a
library function from an expansion is to write a fixed alias and ask
the caller to import under it. Net writes `nn:`. Ask: rewrite `m:` in
a macro's text to the importer's alias (for the library's own `.xtl`
half), or let a macro name a library it imports (a hook giving the
caller's alias for it). X_eTaL-libraries filed the same as its X14.

### M3: matrix product speed (landed)

The networks are matrix products. At abb8274 four products of a
1024 x 16 by a 16 x 16 matrix (about a million multiply-adds) took
0.8 s, about 770 ns per multiply-add, and `t_able` about 620 ns per
cell: a regression from 06d39fa that X_eTaL-demos filed and X_eTaL's
Saga 30 fixed. At X_eTaL v0.1.0 the same program takes 145 ms and
every ML workload here runs 4 to 6 times faster (`just bench-check`,
docs/speed.md).

### M4: grade per row

```
$ xetal eval -e "M := 2 4 r_eshape 0.1 0.5 0.3 0.9 0.7 0.2 0.8 0.1
g_rade_2 M"
1 3 2 4
```

grades the columns as items; a top-k per row needs each row graded
(`'g_rade e_ach_1 M` is refused: e_ach needs a single value from each
call). Ask: a grade along an axis per row, so top-k is `k t_ake_2` of
it. Workaround in moe-router: the largest of each row as a mask, taken
out, then the largest again.

### M12: shapes in types

```
a1 := 3 3 r_eshape 0.5
u:f_wd := "2 4 softmax" net:n_etwork< "a1"    # the spec says 4 outputs
u:f_wd 1 2 r_eshape 1.0                        # runs: three numbers
2 3 r_eshape 1 2                               # 1 2 1 / 2 1 2: data repeated to fit
```

Inference sees `Float -> Float` and has nothing to object to: a
matrix's type carries no shape, and `r_eshape` makes any data fit.
The Net library closes the gap by writing run-time checks into the
code its macros produce (a file's count at load, `net:s_hapes<`),
which is the best a library can do. Asked, in rising size:

1. A shape ascription built-in (`x :> 3 4`, say): checks at run time,
   stops with an error that names the expected and actual shapes, and
   gives a future checker a place to hang a type. Cheap, and what the
   generated code wants today.
2. Rank in types (vector, matrix, cube), the first typed step.
3. Sized types (sizes as literals and variables with linear
   arithmetic, data-dependent sizes existential with a coercion, as
   Futhark's): the real fix, and a large change (every inferred type
   changes, programs that reshape from computed vectors need
   coercions, broadcasting needs typing rules). Research, with this
   library and the ML demos as the motivating case.

### M14: pictures from runs at once

Two threads each run `"p:" u_se< "Plot"` then `c := p:l_ine! 1.0 2.0
3.0` with `xetal_play::run`: now and then one run returns two
pictures and the other none (seen in backprop's web tests, 1 run in 2
with cargo's parallel tests). The pictures should belong to the run,
as its output does. Workaround: one program at a time, behind a lock
in `shared/microscope/src/run.rs`.

### M15: a title for a doc site

`xetal doc --out DIR FILES` writes every page's title as "NAME -
X_eTaL doc" and heads the sidebar "X_eTaL doc", so this repository's
cross-reference (pages/doc, live at /doc) says it is X_eTaL's. Asked:
a `--title TEXT` option (the sidebar's heading and the titles'
suffix). Repro: `xetal doc --out d lib.xtl && grep -o "<title>[^<]*" d/index.html`
gives "index - X_eTaL doc". Workaround: none needed to work; the
catalog, the demo cards and every page's footer link it as "the
cross-reference".

### M16: the matrix product's speed

At the pinned X_eTaL a matrix product costs about 85 ns a
multiply-add, whatever its shape: `(8 9 r_eshape ...) '+ '* i_nner 9
33800 r_eshape ...` (270,400 results of 9 terms) and `(8 33800
r_eshape ...) '+ '* i_nner o_\ 9 33800 r_eshape ...` (72 results of
33,800 terms) are both about 0.2 s, 2.4 million multiply-adds each.
CNN training's step on 20 digits is about 0.4 s natively and 1.6 s in
the browser, mostly two such products; the demo uses 600 digits, not
MNIST's 60,000. (M3 brought it from about 700 ns at abb8274.) Workaround: smaller data.

### To take to X_eTaL

M15 and M16 are filed here but not yet in X_eTaL's ledger. Entries in
its `docs/asks.toml` form, ready to add there (this repository does
not change X_eTaL):

```
[[ask]]
id = "M15"
repo = "X_eTaL-ML"
ask = "a title for a doc site: `xetal doc --out` names every site \"X_eTaL doc\""
state = "new"
here = ""

[[ask]]
id = "M16"
repo = "X_eTaL-ML"
ask = "the matrix product at about 85 ns a multiply-add, whatever its shape"
state = "new"
here = ""
```

### M5, M10

M5 is recorded with its repro in
`../X_eTaL-demos/docs/xetal-asks.md`; X_eTaL plans it with its host
bindings (Saga 23). M10 is declined: `e_ach` stays scalar-only, and
`m_ap` returns each result boxed; whether a mix turns those boxes back
into an array is for the user to decide in X_eTaL. With it the ten
digits of cnn-digits would be one `m_ap` giving a 10 x 10 matrix of
probabilities; today they are one `e_ach` per number wanted.

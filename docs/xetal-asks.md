# Asks for X_eTaL

Features the ML demos and libraries need that X_eTaL does not have
yet, and bugs they uncovered. This repo does not change X_eTaL: each
ask is filed here (and taken to `../X_eTaL`), the demo or library uses
the workaround noted below or waits, and the workaround is removed
when the ask lands in the X_eTaL commit this repo pins (`XETAL_COMMIT`).

Each entry: status (open, filed, landed, dropped), kind (feature, bug
or speed), which demos or libraries need it, why, a minimal repro or
example, and the workaround in use. Asks first filed by a sibling
repo are copied here with this repo's users named, so this list
stands on its own; every one was re-run against this repo's pinned
X_eTaL (v0.1.0, 512b3ee) on 2026-10-05.

| # | Status | Kind | Ask | Demos, libraries | Workaround |
| - | ------ | ---- | --- | ---------------- | ---------- |
| M1 | landed | feature | `.xtlm` macro libraries: in X_eTaL v0.1.0 (`lib/Macros.xtlm`, `Combinators.xtlm`, the system macros of `System.xtlm`) | Net (`net:n_etwork<`), the net-macro demo | none left: saga 3 can start |
| M2 | landed | feature | `xetal expand FILE`: the program after macro expansion, in X_eTaL v0.1.0 | Net, net-macro | none left |
| M3 | landed | speed | `'+ '* i_nner` (matrix product) and `t_able`: fixed in X_eTaL v0.1.0; here the dense benchmark went from 804 ms to 145 ms and the CNN program from 1.5 s to 0.3 s (docs/speed.md) | ternary-net, cnn-digits, NN (`nn:d_ense`) | none needed; the pages keep their small maps |
| M4 | open | feature | Grade per row (top-k along an axis): `g_rade_2 M` still grades the columns as items | moe-router | the row maximum as a mask, taken out, then the maximum again |
| M5 | open | feature | `xetal-play`: arrays in and out without text, or a session kept between runs (v0.1.0 adds an interactive run that waits for typed lines, not arrays) | every page | programs written with literal arrays or data files; printed `r_avel` lines parsed |
| M6 | landed | speed | Whole-array arithmetic: fast enough for every demo here since abb8274 | all | none |
| M7 | landed | bug | Long Int literal strands read in quadratic time: fixed | moe-router (word numbers) | removed |
| M8 | landed | feature | Float literals with an exponent (`1.5e-5`) | NN (`1e-12`), attention (`1e9`) | none |
| M9 | landed | bug | A Bool bound to a name in arithmetic: works in X_eTaL v0.1.0 (`a := 1 2 > 0` then `1 * a` gives `1 1`) | masks | none left (no demo here carried the workaround) |
| M10 | open | feature | `e_ach` returning an array per item (`'{ i -> i * 1 2 3 } e_ach 1 2` is refused: e_ach needs a single value from each call) | cnn-digits (the ten digits, one number per call), a per-row grade | one `e_ach` per number wanted |
| M11 | filed | feature | A macro's text cannot name a library by the importer's alias: `m:f_` in an expansion is not rewritten to the alias the caller chose, and an expansion may not import (MC23), so a macro library cannot call its own `.xtl` half, or another library, without fixing the alias (X_eTaL-libraries X14) | Net (its networks call NN) | the expansion says `nn:`; the page tells the reader to import NN under that alias |
## Details

### M1, M2: macro libraries and seeing an expansion (landed)

X_eTaL v0.1.0 has `.xtlm` macro libraries (MC10 to MC30: a macro is a
function from the source text left and right of its call to new
source; hooks such as `[]R_EJECT` report a bad argument at the call;
hygiene is automatic) and `xetal expand FILE`. `libs/Net` is built on
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

### M5, M10

M5 is recorded with its repro in
`../X_eTaL-demos/docs/xetal-asks.md`. M10: X_eTaL's own message says
nested results come later; with it the ten digits would be one
`e_ach` giving a 10 x 10 matrix of probabilities, and a per-row grade
(M4) could be written with `e_ach` too.

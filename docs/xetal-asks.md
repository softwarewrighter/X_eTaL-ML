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
## Details

### M1: `.xtlm` macro libraries

X_eTaL decided (MC10 to MC13) and planned (its Saga 19) user macro
libraries: a `.xtlm` file defines `m:name<` macros, each a function
from the source text at the call's left and right to new source; a
`Name.xtl` and `Name.xtlm` in one directory load together under one
alias. The ML macro library this repo plans (plan A10, saga 4) is
`Net.xtlm`:

```
"net:" u_se< "Net"
f := "784 128 relu 10 softmax" net:n_etwork< "W1 b1 W2 b2"
```

expanding into a definition built from `nn:` calls, then type-checked
like any code. It shows a macro adding notation for a domain, not
only control flow (research3.txt). Workaround: none; the library and
its demo wait.

### M2: seeing an expansion

A macro whose expansion cannot be seen is magic. `xetal expand FILE`
(or `--expand`) printing the program after expansion lets the
net-macro demo show the layer list and the code it became side by
side, and lets its tests pin the expansion.

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

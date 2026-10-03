# Asks for X_eTaL

Features the ML demos and libraries need that X_eTaL does not have
yet, and bugs they uncovered. This repo does not change X_eTaL: each
ask is filed here (and taken to `../X_eTaL`), the demo or library uses
the workaround noted below or waits, and the workaround is removed
when the ask lands in a vendored release (`vendor/xetal/VENDORED`).

Each entry: status (open, filed, landed, dropped), kind (feature, bug
or speed), which demos or libraries need it, why, a minimal repro or
example, and the workaround in use. Asks first filed by a sibling
repo are copied here with this repo's users named, so this list
stands on its own; every one was re-run against this repo's vendored
X_eTaL (abb8274) on 2026-10-03.

| # | Status | Kind | Ask | Demos, libraries | Workaround |
| - | ------ | ---- | --- | ---------------- | ---------- |
| M1 | filed | feature | `.xtlm` macro libraries: `m:n_ame<` macros, `(String, String) -> String`, imported with `u_se<` (X_eTaL decisions MC10 to MC13, its Saga 19; X_eTaL-libraries X1) | Net (`net:n_etwork<`), the net-macro demo | none: they wait (plan A10, saga 4) |
| M2 | filed | feature | `xetal expand FILE`: the source after macro expansion (X_eTaL-libraries X2) | Net, net-macro | none: waits with M1 |
| M3 | open | speed | `'+ '* i_nner` (matrix product): about 700 ns per multiply-add at abb8274, worse than at 06d39fa (370 ns); `t_able` regressed too (X_eTaL-demos measured 2.7x) | ternary-net, cnn-digits, NN (`nn:d_ense`) | small maps; pages rerun only what changed |
| M4 | open | feature | Grade per row (top-k along an axis): `g_rade_2 M` still grades the columns as items | moe-router, Sample (top-k) | the row maximum as a mask, taken out, then the maximum again |
| M5 | open | feature | `xetal-play`: arrays in and out without text, or a session kept between runs (its API is still `run`, `run_to`, `check`, notebooks; upstream Saga 23) | every page | programs written with literal arrays; printed `r_avel` lines parsed |
| M6 | open | speed | Whole-array arithmetic: much faster at abb8274 (about 8x on `x + x * x`, X_eTaL-demos' measurement); vector kernels still planned upstream (Saga 22) | cnn-digits, training demos | none needed for the current demos |
| M7 | landed | bug | Long Int literal strands read in quadratic time: fixed (8000 Ints in 14 ms at abb8274) | moe-router (word numbers) | removed: the page passes Ints |
| M8 | landed | feature | Float literals with an exponent (`1.5e-5`): read at abb8274 | pages writing small Floats | none left: `microscope::run::lit` writes plain decimals, which still read |
| M9 | open | bug | A Bool bound to a name cannot be used in arithmetic (`a := 1 2 > 0` then `1 * a` or `f_loat a`: type mismatch at abb8274) | masks in NN and Attention | bind masks as Floats |

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

### M3: matrix product speed

The networks are matrix products. Four products of a 1024 x 16 by a
16 x 16 matrix (about a million multiply-adds) take 0.75 s at abb8274
(release, natively):

```
x := (1024 c_at 16) r_eshape 0.1 0.2 -0.3 0.5 0.7
w := (16 c_at 16) r_eshape 0.3 -0.1 0.2 0.0 0.4
d := x '+ '* i_nner w        # four times
```

About 700 ns per multiply-add, while elementwise arithmetic is now
tens of nanoseconds per element: a dedicated kernel for `'+ '* i_nner`
on Floats would make the 1.58-bit network's map finer and the CNN page
interactive. X_eTaL-demos filed the regression from 06d39fa (`t_able`
2.7x and `i_nner` 1.5x slower, after the higher-order built-ins became
steps of the machine, upstream D50) with its measurements.

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

### M5, M6, M9

Recorded with their repros in `../X_eTaL-demos/docs/xetal-asks.md`;
re-run here with the results in the table above.

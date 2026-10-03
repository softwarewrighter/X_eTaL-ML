# Asks for X_eTaL

Features the ML demos and libraries need that X_eTaL does not have
yet, and bugs they uncovered. This repo does not change X_eTaL: each
ask is filed here (and taken to `../X_eTaL`), the demo or library uses
the workaround noted below or waits, and the workaround is removed
when the ask lands in a vendored release (`vendor/xetal/VENDORED`).

Each entry: status (open, filed, landed, dropped, re-check), kind
(feature, bug or speed), which demos or libraries need it, why, a
minimal repro or example, and the workaround in use. Asks first filed
by a sibling repo are copied here with this repo's users named, so
this list stands on its own; `re-check` means inherited and not yet
verified against this repo's vendored X_eTaL (done in saga 1 step 4).

| # | Status | Kind | Ask | Demos, libraries | Workaround |
| - | ------ | ---- | --- | ---------------- | ---------- |
| M1 | filed | feature | `.xtlm` macro libraries: `m:n_ame<` macros, `(String, String) -> String`, imported with `u_se<` (X_eTaL decisions MC10 to MC13, its Saga 19; X_eTaL-libraries X1) | Net (`net:n_etwork<`), the net-macro demo | none: they wait (plan A10, saga 4) |
| M2 | filed | feature | `xetal expand FILE`: the source after macro expansion (X_eTaL-libraries X2) | Net, net-macro | none: waits with M1 |
| M3 | re-check | speed | `'+ '* i_nner` (matrix product) about 370 ns per multiply-add, slower than broadcast-and-reduce (X_eTaL-demos) | ternary-net, cnn-digits, NN (`nn:d_ense`) | small maps; pages rerun only what changed |
| M4 | re-check | feature | Grade per row (top-k along an axis) (X_eTaL-demos) | moe-router, Sample (top-k) | the row maximum as a mask, taken out, then the maximum again |
| M5 | re-check | feature | `xetal-play`: arrays in and out without text, or a session kept between runs (X_eTaL-demos; upstream Saga 23) | every page | programs written with literal arrays; printed `r_avel` lines parsed |
| M6 | re-check | speed | Whole-array arithmetic about 50 ns per element per operation (X_eTaL-demos; upstream Saga 22) | cnn-digits (a forward pass), training demos | small models and inputs |
| M7 | re-check | bug | Long Int literal strands read in quadratic time (X_eTaL-demos D1; fixed upstream 2026-10-02) | weights written as literals | Floats |
| M8 | re-check | feature | Float literals with an exponent (X_eTaL-demos D4; landed upstream 2026-10-02 as S8) | weights and small learning rates written as literals | the shortest plain decimal |
| M9 | re-check | bug | A Bool bound to a name cannot be used in arithmetic (X_eTaL-demos) | masks in NN and Attention | bind masks as Floats |

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

### M3 to M9

Inherited from `../X_eTaL-demos/docs/xetal-asks.md`, where their
repros and measurements are recorded. Each is re-run against this
repo's vendored X_eTaL when the demos are taken over (saga 1 step 4)
and then given its own section here, or marked landed.

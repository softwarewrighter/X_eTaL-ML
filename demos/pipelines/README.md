# Pipelines

Layers written as data flow. [Flow](../../libs/Flow/docs/README.md)
pipelines read left to right, `|>` between the stages: each stage takes
the one before as `it`, and `$name` labels a stage's value for later
stages. One macro call becomes ordinary X_eTaL, a line per stage, before
the program is compiled; `xetal expand` shows it.

Live: [Pipelines](https://softwarewrighter.github.io/X_eTaL-ML/pipelines/)
(pick an example and see the pipeline, what it became and the same
layer in plain X_eTaL side by side, checked to agree in your browser).

## Run it

From a clone of this repository (Rust, git and `just`):

```bash
just xetal                         # fetch and build the pinned X_eTaL (once)
just run pipelines                 # four layers, each pipeline checked against plain X_eTaL
just run pipelines rosetta.xtl     # a CNN layer four ways, the feature maps
just expand pipelines              # what each pipeline became
just expand pipelines rosetta.xtl
just serve pipelines               # the web app
just test-demo pipelines           # its CLI, expansion and browser baselines, the web app's tests
```

## The programs

`pipelines.xtl` writes four layers as pipelines on this repository's
libraries ([NN](../../libs/NN/docs/README.md),
[Norm](../../libs/Norm/docs/README.md),
[Attention](../../libs/Attention/docs/README.md)), each followed by the
same layer in plain X_eTaL and a check that they agree:

```
dense := @ flow:p_ipe< c"X |> it '+ '* i_nner W |> b nn:b_ias it"
soft := @ flow:p_ipe< c"z |> 'm_ax r_/ it $m |> e_xp z - m $e |> e / '+ r_/ e"
norm := @ flow:p_ipe< c"X |> nm:s_tandard |> it * (s_hape it) r_eshape g |> it + (s_hape it) r_eshape c"
```

Attention takes its rows apart with a tuple label, `$(Q, K, V)`.
The pipelines are code strings (`c"..."`): editors and pages draw them
as X_eTaL, `|>` and `$` as triangles; the attention one is the verbatim
form `c#"..."#`, which keeps the transpose `o_\` as written.

`rosetta.xtl` is one CNN layer four ways, a Rosetta Stone:

1. As a paper prints it: `Y[o,h,w] = max((sum r, s of P[h+r,w+s] F[o,r,s]) + B[o], 0)`,
   read by X_eTaL-libraries' Equations (`e:e_q<` its value, `e:t_ex<`
   its LaTeX, from one reading of the same text).
2. As a pipeline: the picture's windows, every filter times every
   window in one matrix product, the bias along the filters' axis, the
   maps shaped `[O Ho Wo]` (a shape the pipeline checks), the activation.
3. As its expansion: what `just expand pipelines rosetta.xtl` shows.
4. As plain X_eTaL: [Conv](../../libs/Conv/docs/README.md)'s `c_onv`.

All give the same maps, which the program checks; the page also checks
them against the arithmetic done directly (its tests).

## What it shows about X_eTaL

- A macro library can give a notation (data flow) without hiding
  anything: the expansion is ordinary, type-checked code.
- A code string is X_eTaL inside a string: drawn as code, lexed with the
  file, its errors placed inside it.
- No implicit broadcasting: a bias is added along an axis the library
  names (`nn:b_ias`, `nn:a_ddAxis`), and a shape a stage promises is
  checked when the program runs.

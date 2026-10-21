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
just run pipelines diffusion.xtl   # a diffusion sampler four ways, the pictures
just expand pipelines              # what each pipeline became
just expand pipelines rosetta.xtl
just expand pipelines diffusion.xtl
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

`diffusion.xtl` is a latent diffusion sampler four ways, from the two
forms a paper prints (bold z a latent, D the decoder, eps_theta the
denoiser conditioned on the text embedding of the prompt):

```
X0 = D(zT + sum_{t=T}^{1} [a_t z_t + b_t eps_theta(z_t, t, E_text(prompt)) + sigma_t eps_rand] - z_t)
X0 = D(z0) = D(zT + sum_{t=1}^{T} dz_t)
```

1. As a paper prints it. The first sum is a recurrence (its term at t
   uses the z_t the term before made), so it is one step, the paper's
   summand as one Equations call over the batch and the latent,
   `Zn[n,k] = a[t-1] * Z[n,k] + b[t-1] * eps[n,k] + sigma[t-1] * noise[t-1,n,k]`,
   applied T times by `p_ower`; the networks (`eps_theta`, `D`) are
   X_eTaL functions called outside the string, since an equation's
   functions are its six (`exp`, `log`, `sqrt`, `abs`, `max`, `min`).
   The second form is a plain sum of the steps' differences,
   `z0[n,k] = zT[n,k] + (sum t:=1..T | dz[t-1,n,k])`, collected as the
   steps run; it telescopes to the same latents. Both are drawn as
   math on the page.
2. As a pipeline: the latents of noise, the T steps (`p_ower` over the
   latents and the level), the decoder with the pictures' shape `[N K]`
   checked.
3. As its expansion: what `just expand pipelines diffusion.xtl` shows.
4. As plain X_eTaL: the step without the macro, the sampler as the
   [diffusion](../diffusion/README.md) demo runs it on trained weights
   (here the networks are dense layers with made-up weights, the
   schedule three steps, the last without noise, as DDIM).

All three decode the same pictures, which the program checks; the page
checks them against the arithmetic done directly (its tests).

## Compose your own

The page's last choice is a composer: a program to start from (an
equation with its LaTeX, or a pipeline), a palette of the notations'
parts put in at the caret (the equation grammar's `sum i:=0..N-1 | `,
`A[i,j]`, its functions and Greek names; the pipeline's `|>`, `it`,
`$name`, `$(a, b)`, `[N K]`; this repository's libraries as stages;
the imports), the program typed in ASCII beside its rendered form,
then Run and Expand: what it printed (a line between `$$` drawn as
math), any error placed in the text as X_eTaL places it, and what the
macros became. The input is the grammar, not LaTeX: the LaTeX comes
out of the same reading, and a notation with a part the grammar does
not use is refused at the call rather than read as something else,
which is the lint of the notation. The parts are data,
`shared/microscope/palette.tsv`, each of them run by the tests.

## What it shows about X_eTaL

- A macro library can give a notation (data flow) without hiding
  anything: the expansion is ordinary, type-checked code.
- A code string is X_eTaL inside a string: drawn as code, lexed with the
  file, its errors placed inside it.
- No implicit broadcasting: a bias is added along an axis the library
  names (`nn:b_ias`, `nn:a_ddAxis`), and a shape a stage promises is
  checked when the program runs.
- What an equation's grammar refuses is said, not guessed: a function
  it does not have, a sum that is really a recurrence (`t:=T..1` with
  a term using the one before) stay outside the string, as X_eTaL
  functions and `p_ower`; the asks for a recurrence word and the
  caller's functions are in [docs/xetal-asks.md](../../docs/xetal-asks.md).

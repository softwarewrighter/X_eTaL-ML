# LoRA explorer

Low-rank adaptation (LoRA) of one 8 x 8 projection, computed by X_eTaL
with this repository's [LoRA](../../libs/LoRA/docs/README.md) library.
A frozen weight matrix W is adapted by two small trainable ones, A
(r x 8) and B (8 x r): the layer is y = W x + s B (A x), s = alpha / r.
The adapted layer is a pipeline of equations (Flow and
X_eTaL-libraries' Equations), drawn as math, a row a stage; the update
s B A has rank r; merged into W it gives the same layer; and the same
rank on a large mixture-of-experts model costs what the fine-tuning
research (`docs/fine-tune-moe-research.txt`) works out.

Live: [LoRA explorer](https://softwarewrighter.github.io/X_eTaL-ML/lora-explorer/)
(move the rank slider from 1 to 8; click a row of the math or a
stage button: on the right its numbers and, in 3D, the stages it reads
and what it gives, dragged to turn, as the Pipelines demo shows them;
under the buttons what it computes from what in 2D, the matrices too;
change the large model's sizes).

The arrays here are made up (a fixed ripple), to show the shapes and
the arithmetic. Training them, LoRA beside LoRA-FA, and the rank sweep
against the best possible come next (`docs/fine-tune-moe-demo-plan.md`).

## Run it

From a clone of this repository (Rust, git and `just`):

```bash
just xetal                     # fetch and build the pinned X_eTaL (once)
just run lora-explorer         # the layer's output, the update's rank, the counts
just serve lora-explorer       # the web app at http://127.0.0.1:8435/
just test-demo lora-explorer   # its CLI and browser baselines and the web app's tests
```

## The program

`lora-explorer.xtl` sets the rank and the sizes, makes W, A, B and x,
then writes the adapted layer as a pipeline whose stages are
equations:

```
y := @ flow:p_ipe< c#"
   X $x
|> "h[k] | r" e:e_q< "sum j:=0..7 | A[k,j] * x[j]" $h
|> "u[i] | 8" e:e_q< "sum k:=0..q | B[i,k] * h[k]" $u
|> "y[i] | 8" e:e_q< "(sum j:=0..7 | W[i,j] * x[j]) + s * u[i]""#
```

The same text given to `flow:m_ath<` is the LaTeX the page draws.
Then the update (`lo:d_elta`), the merged matrix (`lo:m_erge`), the
update's rank (`lo:r_ank`, Gram-Schmidt on the rows), the check that
the merged layer's output is the factored one's, and the counts:

| At rank r | 8 x 8 here | A SwiGLU expert, d 4096, f 14336 |
| --------- | ---------- | -------------------------------- |
| numbers trained | r (8 + 8) = 16 r, against 64 | 3 r (d + f): 884,736 at r = 16 |
| bytes in BF16 | | 1.69 MiB an expert at r = 16, 108 MiB for 64, 2.53 GiB for 24 layers |

At rank 4 an adapter on this small matrix trains as many numbers as
the matrix has, and at 8 twice as many: LoRA saves only where r is much
smaller than the matrix.

## What it shows about X_eTaL

- A layer's equations are its code: the pipeline of equations runs,
  and the same text is drawn as math.
- A tuple is one argument: an adapter is `(W, A, B, s)`, taken apart
  by the library's pattern `{ (W, A, B, s) x -> ... }`.
- Right-to-left evaluation keeps the factors apart by itself: `B '+ '*
  i_nner A '+ '* i_nner x` computes A x first.

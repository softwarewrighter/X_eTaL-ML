# LoRA explorer

Low-rank adaptation (LoRA) of one 8 x 8 projection, computed by X_eTaL
with this repository's [LoRA](../../libs/LoRA/docs/README.md) library.
A frozen weight matrix W is adapted by two small trainable ones, A
(r x 8) and B (8 x r): the layer is y = W x + s B (A x), s = alpha / r.
The adapted layer is a pipeline of equations (Flow and
X_eTaL-libraries' Equations), drawn as math, a row a stage; the update
s B A has rank r; merged into W it gives the same layer; and the same
rank on a large mixture-of-experts model costs what the fine-tuning
research (`docs/fine-tune-moe-research.txt`) works out. A backward step
is a second pipeline of equations, the gradients of A, B and x. Then
LoRA and LoRA-FA (A frozen) are trained side by side to fit a known
update, against the best any update of the rank can do (the truncated
SVD), and swept over every rank. VeRA freezes A and B too, random and
shared, and trains two vectors: its layer and gradients are pipelines
of equations as well, and it is trained and swept beside them.

Live: [LoRA explorer](https://softwarewrighter.github.io/X_eTaL-ML/lora-explorer/)
(move the rank slider from 1 to 8; click a row of the math or a
stage button: on the right its numbers and, in 3D, the stages it reads
and what it gives, dragged to turn, as the Pipelines demo shows them;
under the buttons what it computes from what in 2D, the matrices too;
change the large model's sizes; switch the math to the backward step
and click its stages; watch the two loss curves against the best
possible at the slider's rank; switch the math to VeRA's layer or its
backward step; the rank sweep runs once).

W, A, B and x are made up (a fixed ripple), to show the shapes and the
arithmetic; the known update T to learn is too (8 x 8, of rank 4).

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

A backward step, with g = y - (W + T) x the loss's gradient at the
output, is a second pipeline; its dx is checked against the backward
function the library's macro writes from a spec, `"u:f_orward
u:b_ackward" lo:a_dapter< "W A B s | r r | train A B"` (the macro's
gradients are checked against `lo:g_rads` and finite differences in
the library's tests):

```
back := @ flow:p_ipe< c#"
   G $g
|> "b[k] | r" e:e_q< "sum i:=0..7 | B[i,k] * g[i]" $b
|> "dA[k,j] | r 8" e:e_q< "s * b[k] * X[j]" $dA
|> "dB[i,k] | 8 r" e:e_q< "s * g[i] * h[k]" $dB
|> "dx[j] | 8" e:e_q< "(sum i:=0..7 | W[i,j] * g[i]) + s * (sum k:=0..q | A[k,j] * b[k])""#
```

Training: the inputs are the 8 unit vectors, so the loss (`lo:l_oss`)
is half the squared distance between s B A and T. Both methods start
as LoRA does, B at 0, from the same seeded A (`lo:s_eeded`), and take
300 Adam steps (`lo:a_dam`, `lo:a_damFA`). The best any update of rank
r can do is half the sum of T's squared singular values past the r-th
(Eckart-Young); they are the eigenvalues of T' T, found by power
iteration with deflation. At rank 3:

| | loss after 300 steps | trains |
| - | -------------------- | ------ |
| best possible (truncated SVD) | 0.014665 | |
| LoRA | 0.014665 | 48 numbers |
| LoRA-FA | 1.443761 | 24 numbers |

LoRA reaches the optimum; LoRA-FA cannot, since its update s B A keeps
its rows in the frozen A's row space whatever B learns. The rank
sweep after `# -- end of the explorer at rank r` trains both at every
rank 1 to 8: LoRA follows the optimum (0 from rank 4, T's rank),
LoRA-FA needs rank 8 to get near it. The page runs the program up to
that marker at each setting and the whole of it once, for the sweep.

W in 4 bits, as QLoRA keeps its frozen base (the
[Quant](../../libs/Quant/docs/README.md) library): W's 64 numbers in
blocks of 8 to 64 (the page's choice; QLoRA's is 64), each block's
largest size its scale. Uniform 4 bits round to -7 .. 7 steps of the
scale / 7 (`(4, qk) qz:u_niform W`); NF4 takes the nearest of 16
levels at the normal distribution's quantiles (`qk qz:n_f4 W`). At
blocks of 16:

| RMSE | uniform 4 bits | NF4 |
| ---- | -------------- | --- |
| on W, a sine | 0.0361 | 0.0504 |
| on a normal 8 x 8 matrix | 0.1020 | 0.0953 |

NF4 spends its levels where normal weights are, so it wins on normal
numbers and loses on W's, which pile up near the ends. A weight takes
4 bits and its share of a 32-bit scale, 4.5 bits at QLoRA's block of
64, 4.129 with the scales double-quantized; a 4096 x 14336 projection
takes 224 MiB in FP32, 112 in FP16, 31.5 in NF4 and 28.9
double-quantized.

QLoRA: the same adapter trained the same way on the 4-bit base
(`qk lo:q_base W`, computed on dequantized: `lo:d_equant`). W~ is W
plus the 4-bit error, so the adapter has to learn the known update
less that error; the best rank r can do for it comes from the singular
values of `T - (W~ - W)`. At rank 3, blocks of 16:

| | loss after 300 steps | best possible |
| - | -------------------- | ------------- |
| LoRA on W | 0.014665 | 0.014665 |
| QLoRA on W~ | 0.032227 | 0.032226 |

Merging the trained adapter into a 4-bit base (`lo:m_ergeQ`)
quantizes W~ + s B A again: it misses by up to 0.18, and the merged
layer's output the adapter's by up to 0.40.

VeRA (Kopiczko, Blankevoort and Asano, 2024; the research's lines
3612-4464): A0 (r x 8) and B0 (8 x r) frozen, from a seed
(`1 lo:v_era (W, r)`, LoRA-FA's seed, so A0 is its A) and shared by
every 8 x 8 layer; two vectors trained, d (r numbers) and b (8),
`y = W x + b (B0 (d (A0 x)))`. Its forward pass and its backward step
(c = B0' (b g) once, then dd = c (A0 x), db = g (B0 (d (A0 x))) and
dx) are pipelines drawn as math, checked against the functions
`lo:a_dapter< "W A0 B0 dv bv | r r | train dv bv"` writes. Trained on
the same T, steps and rate from its start (b at 0), at the rank on the
slider and in the sweep:

| rank | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| ---- | - | - | - | - | - | - | - | - |
| numbers VeRA trains | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
| VeRA's loss | 1.738 | 1.733 | 1.724 | 1.584 | 1.668 | 1.584 | 1.677 | 1.335 |
| LoRA-FA's | 1.738 | 1.650 | 1.444 | 1.164 | 0.869 | 0.554 | 0.410 | 0.026 |
| LoRA's | 1.142 | 0.410 | 0.015 | 0 | 0 | 0 | 0 | 0 |

At rank 1 VeRA is LoRA-FA: its update is any 8-vector times the seeded
row of A. After that 8 + r numbers on fixed random directions learn
little of an update of 64 numbers, and a larger rank does not always
help (each rank draws other projections). What VeRA saves shows on a
large model: on the research's expert (d 4096, f 14336) at rank 3 it
trains 32,777 numbers to LoRA's 165,888, and the projections, 110,592
numbers, are stored once for every expert of every layer (or not at
all: the seed).

## What it shows about X_eTaL

- A layer's equations are its code: the pipeline of equations runs,
  and the same text is drawn as math.
- A tuple is one argument: an adapter is `(W, A, B, s)`, taken apart
  by the library's pattern `{ (W, A, B, s) x -> ... }`.
- Right-to-left evaluation keeps the factors apart by itself: `B '+ '*
  i_nner A '+ '* i_nner x` computes A x first.
- Gradients are equations too: the backward step is written, run and
  drawn the same way as the forward pass, and checked against code.
- Training is `p_ower` on a tuple state: `(adapter, moments, t)` for
  Adam, the loss appended every 10 steps.

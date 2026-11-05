# LoRA

Low-rank adapters (LoRA): a frozen weight matrix W adapted by a
trainable low-rank update, `y = W x + s B (A x)`, kept in its factored
form; the update merged into the weights to deploy it; the numbers an
adapter trains, their bytes and their cost, and the rank of a matrix to
check that an update is of rank r at most. Training: the gradients of
A, B and the inputs, steps of gradient descent and Adam (through the
Optim library), and LoRA-FA, which freezes A (regenerated from a seed)
and trains B alone.

```
"lo:" u_se< "LoRA"
```

Put each library's `src/` directory on `XETAL_PATH` (`just path`
prints them; see the [README](../../../README.md)); `lo:` is the
recommended alias. The demos and live pages of this repository find
the libraries without setup.

## Conventions

- W is m x n (m outputs, n inputs), A is r x n, B is m x r, the rank r
  much smaller than m and n; s is the scale, alpha / r
  (`alpha lo:s_caling r`), so that changing r keeps the update's size;
  rsLoRA's alpha / sqrt r (`alpha lo:s_calingRs r`) keeps the update's
  learning from slowing as r grows. The adapter takes either.
- An adapter is the tuple `(W, A, B, s)`.
- Inputs are columns: x is one input of n numbers, or an n x N matrix
  of N inputs; W x is `W '+ '* i_nner x`.
- The forward pass computes A x (r numbers per input), then B of that:
  r (m + n) multiply-adds per input. It never makes the m x n matrix
  B A, which costs r m n; that is made only by `lo:m_erge`, to deploy
  one task's adapter as a single matrix.
- LoRA starts with B at 0, so the adapted layer starts as the base.
- Training minimizes the squared error, half the sum of the squared
  differences (`lo:l_oss`), so the loss's gradient at the output, g,
  is the output less the one wanted. With z = A x, the gradients are
  dL/dA = s (B' g) x', dL/dB = s g z' and dL/dx = W' g + s A' (B' g)
  (`'` is the transpose, `o_\` in X_eTaL); B' g is made once. For the
  gradients, x is an n x N matrix (one input: an n x 1 one).
- A step takes the adapter and the inputs with the outputs wanted,
  `(adapter, (x, y))`, and gives back the adapter moved (`p_ower`
  repeats it). Adam's state is `(adapter, moments, t)`: `lo:a_damStart`
  makes it, the moments at 0 and the step count t at 1.
- LoRA-FA (A frozen) skips A's gradient but still sends x's back
  through A. Its update s B A has every row in A's row space, whatever
  B learns: the same rank as LoRA, fewer updates within reach. A is
  `seed lo:s_eeded r n`, regenerated from a whole number, so a
  checkpoint holds B alone. On a SwiGLU expert it trains r (2f + d) of
  LoRA's r (3d + 3f) numbers (`lo:e_xpertFA`); Adam keeps its two
  averages for those alone.

## Functions

| Function | Type | What |
| -------- | ---- | ---- |
| `alpha lo:s_caling r` | `Num a => Float -> a -> Float` | the scale s, alpha / r |
| `alpha lo:s_calingRs r` | `Num a => Float -> a -> Float` | rsLoRA's rank-stabilized scale, alpha / sqrt r |
| `(W, A, B, s) lo:f_orward x` | `Num a => (a, a, a, a) -> a -> a` | the adapted layer, W x + s B (A x), the factors kept apart |
| `lo:d_elta (A, B, s)` | `Num a => (a, a, a) -> a` | the update, s B A: an m x n matrix of rank r at most |
| `lo:m_erge (W, A, B, s)` | `Num a => (a, a, a, a) -> a` | the adapter folded into the weights, W + s B A |
| `(W, A, B, s) lo:l_oss (x, y)` | `(Float, Float, Float, Float) -> (Float, Float) -> Float` | the squared error on inputs x against outputs y, half the sum of the squared differences |
| `(W, A, B, s) lo:g_rads (x, g)` | `Num a => (a, a, a, a) -> (a, a) -> (a, a, a)` | the gradients (dA, dB, dx) for inputs x and the output's gradient g |
| `lr lo:s_gd (adapter, (x, y))` | `Num a => a -> ((a, a, a, a), (a, a)) -> (a, a, a, a)` | a step of gradient descent at the rate lr: A and B moved (Optim's `s_gd`) |
| `lo:a_damStart adapter` | `(Any a, Any b, Num c) => (a, Float, Float, b) -> ((a, Float, Float, b), (Float, Float, Float, Float), c)` | Adam's start: the adapter, A's and B's running averages at 0, the step 1 |
| `lr lo:a_dam (state, (x, y))` | `Float -> (((Float, Float, Float, Float), (Float, Float, Float, Float), Float), (Float, Float)) -> ((Float, Float, Float, Float), (Float, Float, Float, Float), Float)` | Adam's step (Optim's `a_dam`): the next state |
| `seed lo:s_eeded r n` | `Num a => a -> Int -> Float` | an r x n matrix regenerated from a seed, sin(seed k k) / sqrt n: LoRA-FA's frozen A |
| `(W, A, B, s) lo:g_radsFA (x, g)` | `Num a => (a, a, a, a) -> (a, a) -> (a, a)` | LoRA-FA's gradients (dB, dx): A's skipped, x's still through A |
| `lr lo:s_gdFA (adapter, (x, y))` | `Num a => a -> ((a, a, a, a), (a, a)) -> (a, a, a, a)` | LoRA-FA's step of gradient descent: B alone moves |
| `lo:a_damStartFA adapter` | `(Any a, Any b, Any c, Num d) => (a, b, Float, c) -> ((a, b, Float, c), (Float, Float), d)` | LoRA-FA's Adam start: B's two averages only |
| `lr lo:a_damFA (state, (x, y))` | `Float -> (((Float, Float, Float, Float), (Float, Float), Float), (Float, Float)) -> ((Float, Float, Float, Float), (Float, Float), Float)` | LoRA-FA's Adam step: B alone moves |
| `lo:r_ank M` | `Num a => Float -> a` | the number of independent rows of M, by Gram-Schmidt (a row whose remainder is under 1e-9 of the largest row does not count) |
| `r lo:p_arams m n` | `Num a => a -> a -> a` | the numbers a rank-r adapter on an m x n matrix trains, r (m + n) |
| `r lo:e_xpert d f` | `Num a => a -> a -> a` | the numbers rank-r adapters train on a SwiGLU expert's three matrices (gate, up, down), 3 r (d + f) |
| `r lo:e_xpertFA d f` | `Num a => a -> a -> a` | the same expert under LoRA-FA: trained r (2f + d), then frozen r (2d + f) |
| `b lo:b_ytes n` | `Num a => a -> a -> Float` | the bytes n numbers take at b bits each (16 for BF16, 32 for FP32) |
| `r lo:c_osts m n` | `Num a => a -> a -> a` | multiply-adds: B (A x) for one input, r (m + n), then making B A, r m n |

## Examples

From `../tests/basics.xtl` and the doc comments:

```
      (2 2 r_eshape 1.0 0.0 0.0 1.0, 1 2 r_eshape 1.0 1.0, 2 1 r_eshape 1.0 -1.0, 0.5) lo:f_orward 2.0 4.0
5.0 1.0
      lo:m_erge (2 2 r_eshape 1.0 0.0 0.0 1.0, 1 2 r_eshape 1.0 1.0, 2 1 r_eshape 1.0 -1.0, 0.5)
 1.5 0.5
-0.5 0.5
      lo:r_ank 3 3 r_eshape 1.0 2.0 3.0 2.0 4.0 6.0 0.0 1.0 1.0
2
      16 lo:e_xpert 4096 14336
884736
      8 lo:c_osts 64 64
1024 32768
      (2 2 r_eshape 1.0 0.0 0.0 1.0, 1 2 r_eshape 1.0 1.0, 2 1 r_eshape 1.0 -1.0, 0.5) lo:g_rads (2 1 r_eshape 2.0 4.0, 2 1 r_eshape 1.0 -1.0)
(1: 2.0 4.0
, 2:
  3.0
 -3.0
, 3:
 2.0
 0.0
)
      0.1 lo:s_gd ((2 2 r_eshape 1.0 0.0 0.0 1.0, 1 2 r_eshape 1.0 1.0, 2 1 r_eshape 1.0 -1.0, 0.5), (2 1 r_eshape 2.0 4.0, 2 1 r_eshape 4.0 2.0))
(1:
 1.0 0.0
 0.0 1.0
, 2: 0.8 0.6
, 3:
  0.7
 -0.7
, 4: 0.5)
      16 lo:e_xpertFA 4096 14336
524288 360448
      16.0 lo:s_calingRs 4
8.0
```

In the gradient example, W is the identity, A = [1 1], B = [1 -1]',
s = 0.5 and x = [2 4]': z = A x is 6 and B' g is 2, so dA = 0.5 * 2 *
x' = [2 4], dB = 0.5 * 6 * g = [3 -3]' and dx = g + 0.5 * A' * 2 = [2
0]'.

`../tests/checks.xtl` checks, on an 8 x 6 base with 5 inputs:

- the merge identity, `W x + s B (A x) = (W + s B A) x`, and the
  forward pass as W x plus the update times x;
- the update's rank is r for r = 1 to 5 (A and B in general position),
  the base's is 6, an outer product's is 1;
- with B at 0 the layer is the base;
- the scale is alpha / r, and rsLoRA's is alpha / sqrt r (the same at
  rank 1, twice alpha / r at rank 4);
- the worked numbers of the fine-tuning research
  (`docs/fine-tune-moe-research.txt`, a SwiGLU expert with d 4096 and
  f 14336 at rank 16): 884,736 numbers per expert, 1.69 MiB in BF16,
  108 MiB for 64 experts, 2.53 GiB for 24 such layers;
- an 8 x 8 adapter at rank 3 trains 48 numbers, and the factored and
  merged costs.

`../tests/gradients.xtl` checks, on the same 8 x 6 base at rank 3:

- dL/dA, dL/dB and dL/dx against central finite differences
  ((L(w + e) - L(w - e)) / 2e, e = 1e-5, within 1e-6), and their
  shapes;
- LoRA-FA's dB and dx are LoRA's;
- a step of gradient descent lowers the loss; 200 steps of each of
  LoRA, LoRA-FA and their Adam lower it from LoRA's start, B at 0;
- LoRA-FA leaves A as it was, and its learned update lies in A's row
  space (A with the update's rows under it is still of rank 3), while
  LoRA's leaves it;
- a seed regenerates the same A, another seed another, of rank r;
- LoRA-FA on the research's SwiGLU expert (d 4096, f 14336, rank 16):
  524,288 numbers trained and 360,448 regenerated, LoRA's 884,736
  together; Adam's two FP32 averages take 4 MiB against LoRA's 6.75.

The research's table (`docs/fine-tune-moe-research.txt`, about line
1000) gives LoRA-FA 688,128 trained and 196,608 frozen (a 22.2% cut);
its own formulas, r (2f + d) and r (2d + f), give 524,288 and 360,448
(a 40.7% cut): the table took the down projection's A as r x d, where
that matrix is d x f and its A is r x f. The checks follow the
formulas.

The test matrices are the sine of a square (`s_in c * i * i`), not of
a line: rows of `s_in c * i` all lie in a plane, since sin(a + b) is a
sum of sin a and cos a, so every such matrix would be of rank 2.

## Demos

- [`demos/sizes.xtl`](../demos/sizes.xtl): an 8 x 8 projection adapted
  at rank 3 (48 numbers trained, against 64; the factored and merged
  outputs agree; the update's rank is 3), then the adapters of one
  SwiGLU expert (d 4096, f 14336) at ranks 8 to 64: the numbers, and
  the MiB in BF16 for one expert and for 64.

## Provenance

Written for X_eTaL-ML after the LoRA paper (Hu et al., "LoRA: Low-Rank
Adaptation of Large Language Models", 2021) and the questions and
answers in `docs/fine-tune-moe-research.txt` (the factored form, the
merge, the counts and their worked example), and LoRA-FA (Zhang et
al., "LoRA-FA: Memory-efficient Low-rank Adaptation for Large Language
Models Fine-tuning", 2023). The gradients are derived by hand (the
research's lines 2755-2947) and checked by finite differences. QLoRA
and VeRA come in later steps (`docs/fine-tune-moe-demo-plan.md`).

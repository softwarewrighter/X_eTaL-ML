# LoRA

Low-rank adapters (LoRA): a frozen weight matrix W adapted by a
trainable low-rank update, `y = W x + s B (A x)`, kept in its factored
form; the update merged into the weights to deploy it; the numbers an
adapter trains, their bytes and their cost, and the rank of a matrix to
check that an update is of rank r at most.

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
  (`alpha lo:s_caling r`), so that changing r keeps the update's size.
- An adapter is the tuple `(W, A, B, s)`.
- Inputs are columns: x is one input of n numbers, or an n x N matrix
  of N inputs; W x is `W '+ '* i_nner x`.
- The forward pass computes A x (r numbers per input), then B of that:
  r (m + n) multiply-adds per input. It never makes the m x n matrix
  B A, which costs r m n; that is made only by `lo:m_erge`, to deploy
  one task's adapter as a single matrix.
- LoRA starts with B at 0, so the adapted layer starts as the base.

## Functions

| Function | Type | What |
| -------- | ---- | ---- |
| `alpha lo:s_caling r` | `Num a => Float -> a -> Float` | the scale s, alpha / r |
| `(W, A, B, s) lo:f_orward x` | `Num a => (a, a, a, a) -> a -> a` | the adapted layer, W x + s B (A x), the factors kept apart |
| `lo:d_elta (A, B, s)` | `Num a => (a, a, a) -> a` | the update, s B A: an m x n matrix of rank r at most |
| `lo:m_erge (W, A, B, s)` | `Num a => (a, a, a, a) -> a` | the adapter folded into the weights, W + s B A |
| `lo:r_ank M` | `Num a => Float -> a` | the number of independent rows of M, by Gram-Schmidt (a row whose remainder is under 1e-9 of the largest row does not count) |
| `r lo:p_arams m n` | `Num a => a -> a -> a` | the numbers a rank-r adapter on an m x n matrix trains, r (m + n) |
| `r lo:e_xpert d f` | `Num a => a -> a -> a` | the numbers rank-r adapters train on a SwiGLU expert's three matrices (gate, up, down), 3 r (d + f) |
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
```

`../tests/checks.xtl` checks, on an 8 x 6 base with 5 inputs:

- the merge identity, `W x + s B (A x) = (W + s B A) x`, and the
  forward pass as W x plus the update times x;
- the update's rank is r for r = 1 to 5 (A and B in general position),
  the base's is 6, an outer product's is 1;
- with B at 0 the layer is the base;
- the scale is alpha / r;
- the worked numbers of the fine-tuning research
  (`docs/fine-tune-moe-research.txt`, a SwiGLU expert with d 4096 and
  f 14336 at rank 16): 884,736 numbers per expert, 1.69 MiB in BF16,
  108 MiB for 64 experts, 2.53 GiB for 24 such layers;
- an 8 x 8 adapter at rank 3 trains 48 numbers, and the factored and
  merged costs.

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
merge, the counts and their worked example). The gradients, LoRA-FA,
QLoRA and VeRA come in later steps (`docs/fine-tune-moe-demo-plan.md`).

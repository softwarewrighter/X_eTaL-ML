# LoRA

Low-rank adapters (LoRA): a frozen weight matrix W adapted by a
trainable low-rank update, `y = W x + s B (A x)`, kept in its factored
form; the update merged into the weights to deploy it; the numbers an
adapter trains, their bytes and their cost, and the rank of a matrix to
check that an update is of rank r at most. Training: the gradients of
A, B and the inputs, steps of gradient descent and Adam (through the
Optim library), and LoRA-FA, which freezes A (regenerated from a seed)
and trains B alone. QLoRA: the base stored in 4 bits (NF4, through
the Quant library), the layer computed on it dequantized, the merge
into a 4-bit base that is not exact, and QLoRA's bytes. VeRA: A and B
frozen too, random, regenerated from a seed and shared by every matrix
of the shape, and two vectors trained, `y = W x + b (B0 (d (A0 x)))`.

```
"lo:" u_se< "LoRA"
```

One import reaches the functions (`LoRA.xtl`) and the macro
(`LoRA.xtlm`, `lo:a_dapter<`). Put each library's `src/` directory on `XETAL_PATH` (`just path`
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
- QLoRA keeps W in NF4 (`k lo:q_base W`: codes, a scale per block of
  k, W's shape; QLoRA's k is 64) and computes with it dequantized, W~
  (`lo:d_equant`), the adapter in full precision: `(Q, A, B, s)
  lo:f_orwardQ x` is W~ x + s B (A x). It trains as LoRA does, with W~
  for W (`lr lo:s_gd ((lo:d_equant Q, A, B, s), (x, y))`), and since W~
  is W plus the 4-bit error E, what the adapter has to learn is the
  update less E. Merging into a 4-bit base (`k lo:m_ergeQ`) quantizes
  W~ + s B A again: not exact, the merged numbers fall between NF4's
  levels.
- LoRA-FA (A frozen) skips A's gradient but still sends x's back
  through A. Its update s B A has every row in A's row space, whatever
  B learns: the same rank as LoRA, fewer updates within reach. A is
  `seed lo:s_eeded r n`, regenerated from a whole number, so a
  checkpoint holds B alone. On a SwiGLU expert it trains r (2f + d) of
  LoRA's r (3d + 3f) numbers (`lo:e_xpertFA`); Adam keeps its two
  averages for those alone.
- VeRA's adapter is `(W, A0, B0, d, b)`: A0 (r x n) and B0 (m x r)
  frozen, from `seed lo:p_rojections m n r`, d (r numbers) and b (m)
  trained; `seed lo:v_era (W, r)` starts it as the paper does, d at
  0.1 and b at 0, so the layer starts as the base. Its update diag(b)
  B0 diag(d) A0 needs no diagonal matrix and no broadcasting: diag(b)
  B0 diag(d) is `B0 * b '* t_able d`, item by item. It trains m + r
  numbers per matrix (`lo:p_aramsV`), 2 f + d + 3 r per SwiGLU expert
  (`lo:e_xpertV`); the projections are stored once for every matrix of
  the shape, or not at all (the seed).

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
| `seed lo:p_rojections m n r` | `Num a => a -> Int -> (Float, Float)` | VeRA's frozen projections (A0, B0) from a seed, shared by every matrix of the shape |
| `seed lo:v_era (W, r)` | `Num a => a -> (b, Int) -> (b, Float, Float, Float, Float)` | a VeRA adapter (W, A0, B0, d, b) as the paper starts it: d at 0.1, b at 0 |
| `lo:d_eltaV (A0, B0, d, b)` | `Num a => (a, a, a, a) -> a` | VeRA's update, diag(b) B0 diag(d) A0 |
| `(W, A0, B0, d, b) lo:f_orwardV x` | `Num a => (a, a, a, a, a) -> a -> a` | the VeRA layer, W x + b (B0 (d (A0 x))), in factored form |
| `lo:m_ergeV adapter` | `Num a => (a, a, a, a, a) -> a` | the VeRA adapter folded into the weights |
| `adapter lo:l_ossV (x, y)` | `(Float, Float, Float, Float, Float) -> (Float, Float) -> Float` | the VeRA layer's squared error |
| `adapter lo:g_radsV (x, g)` | `Num a => (a, a, a, a, a) -> (a, a) -> (a, a, a)` | VeRA's gradients (dd, db, dx); the projections get none |
| `lr lo:s_gdV (adapter, (x, y))` | `Num a => a -> ((a, a, a, a, a), (a, a)) -> (a, a, a, a, a)` | VeRA's step of gradient descent: d and b move |
| `lo:a_damStartV adapter` | `(Any a, Any b, Any c, Num d) => (a, b, c, Float, Float) -> ((a, b, c, Float, Float), (Float, Float, Float, Float), d)` | VeRA's Adam start |
| `lr lo:a_damV (state, (x, y))` | `Float -> (((Float, Float, Float, Float, Float), (Float, Float, Float, Float), Float), (Float, Float)) -> ((Float, Float, Float, Float, Float), (Float, Float, Float, Float), Float)` | VeRA's Adam step: d and b move |
| `k lo:q_base W` | `Int -> Float -> (Int, Float, Int)` | the base as QLoRA stores it: NF4 codes, a scale per block, W's shape |
| `lo:d_equant Q` | `(Int, Float, Int) -> Float` | the stored base back to numbers, W~, in W's shape |
| `(Q, A, B, s) lo:f_orwardQ x` | `((Int, Float, Int), Float, Float, Float) -> Float -> Float` | QLoRA's layer, W~ x + s B (A x), the base dequantized when it runs |
| `k lo:m_ergeQ (Q, A, B, s)` | `Int -> ((Int, Float, Int), Float, Float, Float) -> (Int, Float, Int)` | the adapter merged into a 4-bit base: W~ + s B A quantized again (not exact) |
| `(r, k) lo:q_bytes m n` | `(Num a, Num b) => (a, b) -> a -> Float` | QLoRA's bytes: the base in NF4, the adapter in BF16, Adam's two FP32 averages |
| `lo:r_ank M` | `Num a => Float -> a` | the number of independent rows of M, by Gram-Schmidt (a row whose remainder is under 1e-9 of the largest row does not count) |
| `r lo:p_arams m n` | `Num a => a -> a -> a` | the numbers a rank-r adapter on an m x n matrix trains, r (m + n) |
| `r lo:e_xpert d f` | `Num a => a -> a -> a` | the numbers rank-r adapters train on a SwiGLU expert's three matrices (gate, up, down), 3 r (d + f) |
| `r lo:e_xpertFA d f` | `Num a => a -> a -> a` | the same expert under LoRA-FA: trained r (2f + d), then frozen r (2d + f) |
| `r lo:p_aramsV m n` | `Num a => a -> a -> a` | the numbers VeRA trains on an m x n matrix, m + r |
| `r lo:e_xpertV d f` | `Num a => a -> a -> a` | the numbers VeRA trains on a SwiGLU expert, 2 f + d + 3 r (its two pairs of projections, 2 r (d + f), shared) |
| `b lo:b_ytes n` | `Num a => a -> a -> Float` | the bytes n numbers take at b bits each (16 for BF16, 32 for FP32) |
| `r lo:c_osts m n` | `Num a => a -> a -> a` | multiply-adds: B (A x) for one input, r (m + n), then making B A, r m n |

## The macro

`"u:f_wd u:b_wd" lo:a_dapter< "W A B s | r 3 | train A B"` writes an
adapted layer from a spec, before the program is compiled (`xetal
expand` shows it): a check that A is r x n and B m x r (the program
stops naming the spec when they are not), the forward function under
the first name on the left and the backward function under the second.
The spec's three fields, between bars:

- the arrays: `W A B s` (the frozen base, the factors, the scale), or
  for VeRA `W A B d b` (the base, the frozen projections, the two
  vectors), or `W` alone for `W A B s`, or `W A B d b` when d and b
  train;
- the rank: `r` and a number or a name;
- what trains: `train A B` is LoRA, the backward function giving `(dA,
  dB, dx)`; `train B` is LoRA-FA (A frozen), giving `(dB, dx)`;
  `train d b` (of five arrays) is VeRA, the forward function W x + b
  (B (d (A x))) and the backward one giving `(dd, db, dx)`. Nothing
  else trains here.

The forward function takes the inputs (columns) and gives W x + s B (A
x); the backward one takes `(x, g)`, the inputs and the loss's gradient
at the output. The expansion is the library's formulas written out for
the names given, with no call to the library; its lambdas' own names
are fresh (`[]G_ENSYM`), so arrays named x, g or b are not captured.
From `../tests/basics.xtl`, `"u:f_wd u:b_wd" lo:a_dapter< "W A B s | r
2 | train A B"` becomes:

```
_ := { v ->
  v ? 0
  []P_ANIC "lo:a_dapter<: A must be 2 x n and B m x 2 for \"W A B s | r 2 | train A B\""
} (2 = f_irst s_hape A) & 2 = 2 s_elect s_hape B
u:f_wd := { g3:x -> (W '+ '* i_nner g3:x) + s * B '+ '* i_nner A '+ '* i_nner g3:x }
u:b_wd := { (g3:x, g2:g) ->
  g1:b := (o_\ B) '+ '* i_nner g2:g
  (s * g1:b '+ '* i_nner o_\ g3:x, s * g2:g '+ '* i_nner o_\ A '+ '* i_nner g3:x,
    ((o_\ W) '+ '* i_nner g2:g) + s * (o_\ A) '+ '* i_nner g1:b)
}
```

`../tests/adapter.xtl` checks, on the 8 x 6 base at rank 3, that the
forward function is `lo:f_orward`, the backward functions are
`lo:g_rads` and `lo:g_radsFA` and the central finite differences, that
the rank may be a name and arrays named x, g and b are not captured,
and that a step with the gradients is `lo:s_gd`'s; for VeRA, that its
functions are `lo:f_orwardV` and `lo:g_radsV` and the finite
differences, and that `W | r 3 | train d b` is VeRA on `W A B d b`.
`bad-spec`, `bad-train`, `bad-vera` and `bad-names` pin the refusals, `bad-rank` the run-time
check. The LoRA explorer's backward step, and its VeRA layer forward and
back, are checked against the functions the macro writes.

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
      lo:d_equant 2 lo:q_base 2 2 r_eshape 1.0 -4.0 2.0 0.5
0.9844492077827454               -4.0
               2.0 0.4922246038913727
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

`../tests/qlora.xtl` checks, on an 8 x 8 base in blocks of 16 at
rank 3:

- the stored base comes back as Quant's NF4 of W, in W's shape, with
  a scale per block, and it is not W;
- QLoRA's layer is LoRA's on W~, and its gradients on W~ are the
  central finite differences of its loss;
- trained on a known rank-4 update T (the unit vectors as inputs, the
  outputs (W + T) x wanted), QLoRA reaches the best rank 3 allows for
  T less the 4-bit error (0.032227 against 0.032226), LoRA on W the
  best for T (0.014665): the 4-bit base leaves the adapter more to
  learn;
- merged into a 4-bit base, the update is not exact (off by up to
  0.18 here) but within half NF4's widest gap times each block's
  absmax, and the merged layer's output moves from the adapter's;
- a 4096 x 14336 projection at rank 16 in blocks of 64: the base 31.5
  MiB in NF4 (112 in FP16), the adapter 0.56 MiB in BF16, Adam's
  averages 2.25 MiB.

`../tests/vera.xtl` checks, on the 8 x 6 base at rank 3:

- the factored VeRA layer is the merged matrix times x, and
  `B0 * b '* t_able d` is diag(b) B0 diag(d) written entry by entry;
  the update is of rank 3;
- `lo:v_era` starts as the base, with the seed's projections, d at
  0.1 and b at 0;
- dL/dd, dL/db and dL/dx against central finite differences (within
  1e-6), dx is LoRA's with A0 and the scaled B0, and the shapes;
- descent and Adam lower the loss and leave the projections as they
  were; a seed regenerates them, another seed gives others;
- the research's MoE projection (n 4096, m 14336, rank 16, 64
  experts): 14,352 numbers an expert, 918,528 for all, 20.5 times
  fewer than expert LoRA, 1.75 MiB in BF16, the shared projections
  0.56 MiB.

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
after Dettmers et al., "QLoRA: Efficient Finetuning of Quantized
LLMs", 2023 (its formats in the Quant library). VeRA after Kopiczko,
Blankevoort and Asano, "VeRA: Vector-based Random Matrix Adaptation",
2024 (b at 0 and d at 0.1 to start, as there), its gradients after the
research's lines 4130-4250, checked by finite differences; the counts
after its lines 3780-3975.

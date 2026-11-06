# Quant

Weights in fewer bits: float formats (FP32, FP16, bfloat16) as their
rounding, symmetric integers (INT8, INT4), ternary weights (-1, 0, +1:
1.58 bits) with their one scale, a ternary layer computed by
additions alone, and QLoRA's 4-bit formats: blockwise uniform 4 bits,
NF4 (16 levels at the normal's quantiles, its codebook derived here),
and double quantization of the block scales, with the bits each takes.

```
"qz:" u_se< "Quant"
```

Put each library's `src/` directory on `XETAL_PATH` (`just path`
prints them; see the [README](../../../README.md)); `qz:` is the
recommended alias. The demos and live pages of this repository find
the libraries without setup.

## Conventions

- Every format is simulated: the numbers stay Float, rounded as the
  format would store them, so a network runs unchanged on them.
- A step or a threshold is one number for the whole array, or an array
  shaped as the weights: per layer, per row, as the caller chooses
  (the 1.58-bit network demo passes one per layer).
- Ternary codes are Floats -1.0, 0.0, 1.0, so codes times a scale are
  weights.
- A block is k consecutive numbers of the raveled array (`qz:b_locks`;
  the count must be a multiple of k), each with its own scale, its
  largest size (absmax). Blockwise 4 bits divide by the scale over 7,
  round and clip to -7 .. 7; NF4 divides by the scale and takes the
  nearest of its 16 levels (codes 1 .. 16). Both keep 0 and each
  block's largest size exactly. QLoRA's block is 64 numbers.
- NF4's levels sit at the standard normal's quantiles (`qz:c_odebook`,
  through `qz:q_uantile`, the inverse of `qz:p_hi`), so each is as
  likely as the next for normal weights, which pretrained weights
  roughly are. On normal weights its error is below uniform 4 bits';
  on uniform weights it is above (checked).
- Double quantization stores the block scales themselves in 8 bits
  (their mean taken off, blocks of k2 = 256 scales, the last made up
  with zeros), as QLoRA does; QLoRA's second level is an 8-bit float,
  8-bit integers here, the same bytes.

## Functions

| Function | Type | What |
| -------- | ---- | ---- |
| `b qz:f_loats x` | `Float -> Float -> Float` | each number rounded to b bits after its leading one (23 FP32, 10 FP16, 7 bfloat16) |
| `s qz:s_nap x` | `Float -> Float -> Float` | each number rounded to the nearest multiple of the step s |
| `b qz:i_nt x` | `Num a => a -> Float -> Float` | symmetric b-bit integers for the whole array: its largest size is 2^(b-1) - 1 steps |
| `s qz:c_ut x` | `Num a => a -> a -> Float` | -1, 0 or +1: +1 where a number passes s, -1 where it is below -s |
| `t qz:t_ernary x` | `Float -> Float -> Float` | `c_ut` at t times the array's mean size |
| `x qz:s_cale q` | `Float -> Float -> Float` | the mean size of the numbers the codes q keep: the one multiply a ternary layer needs |
| `h qz:a_dds q` | `Float -> Float -> Float` | a ternary layer without multiplying: for each column of q, the inputs under +1 summed, minus those under -1 |
| `k qz:b_locks x` | `Int -> a -> a` | x's numbers as rows of k, a block each |
| `k qz:a_bsmax x` | `Num a => Int -> a -> a` | each block's largest size, its scale |
| `(b, k) qz:u_niformCodes x` | `Num a => (a, Int) -> Float -> (Float, Float)` | blockwise symmetric b-bit codes (-7 .. 7 for 4 bits, a row per block) and the scales |
| `(b, k) qz:u_niform x` | `Num a => (a, Int) -> Float -> Float` | x through blockwise b-bit integers and back, shaped as x |
| `qz:p_hi x` | `Float -> Float` | the standard normal's cumulative distribution, erf by its series |
| `qz:q_uantile p` | `Float -> Float` | its inverse, by 60 halvings |
| `qz:c_odebook offset` | `Float -> Float` | NF4's 16 levels from the normal's quantiles (QLoRA's offset 0.9677083) |
| `k qz:n_f4Codes x` | `Int -> Float -> (Int, Float)` | each number's nearest NF4 level (1 .. 16) after its block's scale, and the scales |
| `k qz:n_f4 x` | `Int -> Float -> Float` | x through NF4 and back, shaped as x |
| `k2 qz:d_ouble s` | `Int -> Float -> Float` | block scales through 8 bits (blocks of k2, their mean off) and back |
| `k2 qz:n_f4Double (k, x)` | `Int -> (Int, Float) -> Float` | NF4 with its scales double-quantized |
| `(k, k2) qz:b_its 4` | `(Num a, Num b) => (a, a) -> b -> Float` | bits a weight takes: 4 plus its scale's share; then double-quantized (4.5 and 4.129 at 64 and 256) |

## Examples

From `../tests/basics.xtl`:

```
      0.25 qz:s_nap 0.1 0.2 0.9
0.0 0.25 1.0
      4 qz:i_nt 0.875 -0.5 0.1
0.875 -0.5 0.125
      0.5 qz:c_ut 0.9 0.2 -0.7 -0.1
1.0 0.0 -1.0 0.0
      1.0 2.0 3.0 qz:a_dds 3 2 r_eshape 1.0 0.0 -1.0 1.0 0.0 -1.0
-1.0 -1.0
      (4, 2) qz:u_niform 1.0 -4.0 2.0 0.5
1.1428571428571428 -4.0 2.0 0.5714285714285714
      2 qz:n_f4 1.0 -4.0 2.0 0.5
0.9844492077827454 -4.0 2.0 0.4922246038913727
      (64, 256) qz:b_its 4
4.5 4.12890625
```

In blocks of 2, 1.0 and -4.0 share the scale 4: uniform 4 bits round
1.0 / (4 / 7) = 1.75 up to 2 steps, 1.14; NF4 takes the level nearest
0.25, 0.2461, times 4, 0.984. The block's largest, -4.0, is exact in
both.

`../tests/nf4.xtl` checks, on 2,048 weights in blocks of 64 (uniform
ones by a fixed ripple, normal ones from them by Box-Muller):

- the codebook derived from the normal's quantiles is QLoRA's
  published one to 1e-6 (theirs is float32), 16 levels from -1 to 1
  with 0 among them; Phi of the quantile is p; Phi(1) and the 97.5%
  quantile are the tables';
- every uniform 4-bit error is within half its block's step, every NF4
  error within half the widest gap between levels times the block's
  absmax; both keep each block's largest size and 0 exactly;
- on the normal weights NF4's RMSE is 0.122 against uniform 4 bits'
  0.148; on the uniform weights 0.096 against 0.077: NF4 is better
  where the weights are normal, worse where they are not;
- double-quantized scales stay within half an 8-bit step of their
  spread, and NF4 with them is about as close as without;
- 4.5 bits a weight, 4.129 double-quantized (QLoRA's paper counts
  4.127, leaving out the mean); a 4096 x 14336 projection takes 224
  MiB in FP32, 112 in FP16, 31.5 in NF4, 28.9 double-quantized.

`../tests/checks.xtl` checks the properties on 200 numbers: FP32's
rounding moves a number by at most 2^-24 of itself and FP16's by
2^-11; `s_nap` by at most half a step; INT8 keeps at most 255 levels
and is off by at most half its step; ternary codes are -1, 0, +1; the
scale is the mean size of the kept numbers; `a_dds` equals the
matrix product with the ternary weights. The `## >>` examples are run
by `just doc-test`.

## Demos

- [`demos/formats.xtl`](../demos/formats.xtl): one 16 x 16 matrix
  stored five ways, and how far each moves a layer's output (FP16
  0.0003 to ternary 0.56); the ternary layer by additions equals it
  by multiplying.
- [`../../../demos/ternary-net`](../../../demos/ternary-net/README.md):
  a classifier with FP32, FP16, INT8 and ternary weights, a step and a
  threshold per layer.

## Provenance

Written for X_eTaL-ML from the 1.58-bit network demo's formats (where
they began), generalized to any array and any per-group scale. The
formats are the standard ones (IEEE 754 binary32 and binary16,
bfloat16; symmetric integer quantization); the ternary scheme is
BitNet b1.58's (Ma et al., 2024: weights to -1, 0, +1 at a threshold
from the mean size, one scale per layer). The 4-bit formats are
QLoRA's (Dettmers et al., "QLoRA: Efficient Finetuning of Quantized
LLMs", 2023: blockwise absmax, the NF4 data type, double
quantization), reimplemented here; the NF4 codebook is derived as
bitsandbytes' `create_normal_map` derives it and checked against the
paper's appendix.

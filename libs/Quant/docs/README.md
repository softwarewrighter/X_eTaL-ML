# Quant

Weights in fewer bits: float formats (FP32, FP16, bfloat16) as their
rounding, symmetric integers (INT8, INT4), ternary weights (-1, 0, +1:
1.58 bits) with their one scale, and a ternary layer computed by
additions alone.

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
```

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
from the mean size, one scale per layer).

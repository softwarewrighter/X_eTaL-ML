# Norm

Normalizing each row: standardizing, layer norm (with a gain and a
bias) and RMS norm (with a gain), the normalizations a transformer
puts around its attention and feed-forward layers.

```
"nm:" u_se< "Norm"
```

Put each library's `src/` directory on `XETAL_PATH` (`just path`
prints them; see the [README](../../../README.md)); `nm:` is the
recommended alias. The demos and live pages of this repository find
the libraries without setup.

## Conventions

- Rows are examples or tokens; each row is normalized along its own
  numbers.
- A gain and a bias have one number per column; layer norm takes them
  as a pair, `(g, b) nm:l_ayer m`.
- eps (0.00001) is added to each row's variance, so a constant row
  gives 0, not a division by 0; results are a hair from the exact ones.

## Functions

| Function | Type | What |
| -------- | ---- | ---- |
| `nm:s_tandard m` | `Float -> Float` | each row less its mean, over its standard deviation |
| `(g, b) nm:l_ayer m` | `(Float, Float) -> Float -> Float` | layer norm: each row standardized, times the gain, plus the bias |
| `g nm:r_ms m` | `Float -> Float -> Float` | RMS norm: each row over the square root of its mean square, times the gain |

## Examples

From the library's `## >>` examples (run by `just doc-test`):

```
      (2.0 2.0, 1.0 1.0) nm:l_ayer 1 2 r_eshape 3.0 5.0
-0.9999900000749995 2.9999900000749995
```

`../tests/checks.xtl` checks the properties on five rows: standardized
rows have mean 0 and variance 1; a row's scale and shift do not
matter; layer norm with gain 1 and bias 0 is standardizing; RMS-normed
rows have mean square 1; a constant row becomes 0.

## Demos

- [`demos/rows.xtl`](../demos/rows.xtl): two rows on very different
  scales made alike, and where eps shows.

## Provenance

Written for X_eTaL-ML for the tiny GPT. Layer norm is Ba, Kiros and
Hinton's (2016); RMS norm Zhang and Sennrich's (2019).

# Conv

Convolution for a batch of pictures: every 3 x 3 window as rows, the
filters over all of them as one matrix product, 2 x 2 max-pooling,
and the backward passes a network needs to learn its filters.

```
"cv:" u_se< "Conv"
```

Put each library's `src/` directory on `XETAL_PATH` (`just path`
prints them; see the [README](../../../README.md)); `cv:` is the
recommended alias. The demos and live pages of this repository find
the libraries without setup.

## Conventions

- A batch is n pictures of h x w: `n h w r_eshape ...`; one picture
  is a batch of one.
- Filters K are f rows of 9 (each 3 x 3 filter row by row) with
  biases B (f); a convolution's maps are n x f x (h - 2) x (w - 2)
  (valid: no padding), pooled by 2 x 2 blocks to n x f x (h - 2)/2 x
  (w - 2)/2 (even sides).
- The filters and their biases travel as a pair: `(K, B) cv:c_onv x`.
  A bias added to maps afterward names its axis: `(2, b) nn:a_ddAxis V`
  on a batch of maps (n x f x h x w: the second axis), which is also
  what `b nn:b_ias V` does for a rank-4 array (NN).

## Functions

| Function | Type | What |
| -------- | ---- | ---- |
| `cv:w_indows x` | `a -> a` | every 3 x 3 window of every picture as 9 rows, a column per picture and place (the picture's nine shifted copies) |
| `(K, B) cv:c_onv x` | `Num a => (a, a) -> a -> a` | each filter over every window, plus its bias: one matrix product for the batch |
| `cv:p_ool a` | `Num a => a -> a` | the largest of each 2 x 2 block of the maps |
| `a cv:u_npool d` | `Num a => a -> Float -> Float` | the pooled maps' gradient d sent back through the pooling of a: each block's to its first largest item |
| `V cv:g_rads D` | `Num a => a -> a -> (a, a)` | the filters' and biases' gradients from the windows V and the maps' gradient D |

## Examples

From `../tests/basics.xtl`, two 4 x 4 pictures and two filters (a
vertical edge, a dot), biases 0.5 and -1:

```
      s_hape cv:w_indows x
9 8
      c := (K, B) cv:c_onv x
      s_hape c
2 2 2 2
```

`../tests/checks.xtl` checks the properties on three 8 x 8 pictures
and four filters: the convolution equals the one-picture formula of
the [Tiny CNN](../../../demos/cnn-digits/README.md) demo, picture by
picture; pooling keeps the maps' largest value and halves each side;
unpooling keeps the gradient's total and puts one value in each
block; the filters' and the biases' gradients match central finite
differences. The `## >>` examples are run by `just doc-test`.

## Demos

- [`demos/edges.xtl`](../demos/edges.xtl): a square through two edge
  filters, ReLU and pooling, drawn.
- [`../../../demos/cnn-backprop`](../../../demos/cnn-backprop/README.md):
  a CNN trained in the browser on 600 digits, forward and backward
  through Conv.

## Limits

- 3 x 3 filters, one input channel, valid convolution, 2 x 2 pooling:
  the Tiny CNN's shapes. Wider filters, padding and several input
  channels would each be a few more lines of the same kind.
- The convolution is a matrix product of about 9 x f multiply-adds a
  place, at X_eTaL's speed for it (ask M16): a step on 20 digits of
  28 x 28 takes about 0.4 s at the command line.

## Provenance

Written for X_eTaL-ML from the Tiny CNN demo's convolution and the CNN
training demo's backward pass, generalized to any batch and picture
size. The method is the standard im2col convolution and max-pooling's
gradient routed to the maximum; after Shinkarov's "Convolutional
neural networks in APL" (2019), reimplemented.

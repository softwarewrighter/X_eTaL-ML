# NN

Neural-network building blocks: activations, softmax by row, dense
layers, one-hot labels, loss and accuracy. Each is one short array
expression over a whole batch.

```
"nn:" u_se< "NN"
```

Put each library's `src/` directory on `XETAL_PATH` (`just path`
prints them; see the [README](../../../README.md)); `nn:` is the
recommended alias. The demos and live pages of this repository find
the libraries without setup.

## Conventions

- A batch is a matrix, one example per row.
- A layer is one array: the weights W (one row per input, one column
  per output) with the bias b as one more row at the bottom, so
  `x nn:d_ense wb` is x W + b in two arguments.
- Activations work item by item on any shape. Softmax, log-softmax
  and argmax work along the last axis of any array: a vector is one
  example, a matrix one example per row, a rank-3 array a batch of
  sequences.
- Labels count from 1 (`nn:o_neHot`, `nn:a_rgmax`).
- Everything is Float: `nn:r_elu` of an Int array is a type error;
  write `nn:r_elu f_loat x`.

## Functions

| Function | Type | What |
| -------- | ---- | ---- |
| `nn:r_elu x` | `Float -> Float` | max(x, 0) |
| `a nn:l_eaky x` | `Float -> Float -> Float` | x where positive, a times x elsewhere |
| `nn:s_igmoid x` | `Num a => a -> Float` | 1 / (1 + e^-x) |
| `nn:t_anh x` | `Float -> Float` | the hyperbolic tangent, 2 sigmoid(2x) - 1 |
| `nn:s_oftmax a` | `Num a => a -> Float` | each row into probabilities (largest taken off first, so nothing overflows) |
| `nn:l_ogSoftmax a` | `Float -> Float` | the log of each row's softmax, computed stably |
| `x nn:d_ense wb` | `Num a => a -> a -> a` | the layer x W + b, W and b stacked in wb |
| `nn:a_rgmax a` | `Num a => a -> Int` | each row's position of its largest item (the first on a tie) |
| `k nn:o_neHot y` | `Int -> Int -> Float` | labels 1..k as rows with 1.0 at the label |
| `y nn:c_rossEntropy p` | `Float -> Float -> Float` | mean over rows of -sum(y log p); p clamped at 1e-12 |
| `y nn:m_se p` | `Float -> Float -> Float` | mean squared error over every item |
| `y nn:a_ccuracy p` | `Num a => Int -> a -> Float` | the fraction of rows whose argmax is the label |

## Examples

From `../tests/basics.xtl`, with
`m := 2 3 r_eshape 1.0 2.0 3.0 -1.0 0.0 5.0`:

```
      nn:r_elu m
1.0 2.0 3.0
0.0 0.0 5.0
      0.1 nn:l_eaky m
 1.0 2.0 3.0
-0.1 0.0 5.0
      nn:s_igmoid 0.0 2.0
0.5 0.8807970779778823
      nn:s_oftmax 1.0 2.0 3.0
0.09003057317038046 0.24472847105479764 0.6652409557748218
      nn:s_oftmax m
 0.09003057317038046  0.24472847105479764 0.6652409557748218
0.002456114904450954 0.006676412513376452 0.9908674725821728
      nn:a_rgmax m
3 3
      3 nn:o_neHot 2 1 3
0.0 1.0 0.0
1.0 0.0 0.0
0.0 0.0 1.0
```

A dense layer: rows 1 2 and 3 4 times W (1 0 and 0 2) plus b (10 20):

```
      x := 2 2 r_eshape 1.0 2.0 3.0 4.0
      wb := 3 2 r_eshape 1.0 0.0 0.0 2.0 10.0 20.0
      x nn:d_ense wb
11.0 24.0
13.0 28.0
```

Loss and accuracy:

```
      (2 nn:o_neHot 1 2) nn:c_rossEntropy 2 2 r_eshape 0.9 0.1 0.2 0.8
0.164252033486018
      3 2 nn:a_ccuracy m
0.5
```

`../tests/checks.xtl` checks the properties: softmax rows sum to 1,
ignore a shift and work along the last axis of any rank; log-softmax
is the log of softmax; sigmoid(-x) = 1 - sigmoid(x); tanh is odd;
argmax inverts one-hot; dense is x W + b.

## Demos

- [`demos/xor.xtl`](../demos/xor.xtl): XOR, which no single layer can
  learn, by a two-layer network with hand-set weights: the batch of
  four points through `nn:d_ense`, `nn:r_elu`, `nn:d_ense`,
  `nn:s_oftmax`, then `nn:a_rgmax`, `nn:a_ccuracy` and
  `nn:c_rossEntropy`.
- [`../../../demos/cnn-digits`](../../../demos/cnn-digits/README.md):
  a convolutional network on MNIST digits uses `nn:r_elu`,
  `nn:d_ense`, `nn:s_oftmax` and `nn:a_rgmax`.

## Provenance

Written for X_eTaL-ML from the definitions in its demos (softmax and
top-2 in moe-router, ReLU, dense and softmax in cnn-digits, taken
over from X_eTaL-demos), generalized to any rank. The functions are
the textbook ones (Goodfellow, Bengio and Courville, Deep Learning,
chapters 6.2 and 6.3); the stable softmax and log-softmax subtract
the row maximum first, the usual practice.

# Learn

Classic machine learning, after APLearn's list: k-means, k nearest
neighbors, PCA and logistic regression, each a fit and a predict of a
few array expressions, every iterative fit a step repeated by
`p_ower`.

```
"ml:" u_se< "Learn"
```

Put each library's `src/` directory on `XETAL_PATH` (`just path`
prints them; see the [README](../../../README.md)); `ml:` is the
recommended alias. Learn imports [NN](../../NN/docs/README.md) (for
softmax, one-hot and argmax). The demos and live pages of this
repository find the libraries without setup.

## Conventions

- The data X is a matrix, one example per row; a model's points (the
  k-means centers) are rows too.
- Labels count from 1 (as NN's), and a tie goes to the smallest.
- A model with several parts is a tuple: `(mu, V, ev) := 2 ml:p_ca X`;
  settings come as a tuple on the left, `(k, n) ml:k_means X`.
- Nothing draws on chance: k-means starts farthest-first, PCA's power
  iteration and logistic regression's descent start from fixed
  values, so a fit gives the same answer every run.

## Functions

| Function | Type | What |
| -------- | ---- | ---- |
| `C ml:a_ssign X` | `Float -> Float -> Int` | the nearest of the centers C (one per row) for each row of X |
| `X ml:k_step C` | `Float -> Float -> Float` | one step of k-means (Lloyd's): each center to the mean of the rows nearest it; a center no row is nearest stays |
| `C ml:i_nertia X` | `Float -> Float -> Float` | the sum of each row's squared distance to its nearest center |
| `k ml:f_arthest X` | `Int -> Float -> Float` | k starting centers without chance: the first row, then each time the row farthest from those so far |
| `(k, n) ml:k_means X` | `(Int, Int) -> Float -> Float` | k centers after n steps from the farthest-first start |
| `(k, X, y) ml:k_nn Q` | `(Int, Float, Int) -> Float -> Int` | for each row of Q, the label most of its k nearest rows of X have |
| `n ml:p_ca X` | `Int -> Float -> (Float, Float, Float)` | the mean, the first n principal directions (a unit column each) and the variance along each: `(mu, V, ev)` |
| `(mu, V, ev) ml:p_roject X` | `(Num a, Any b) => (a, a, b) -> a -> a` | each row of X in the principal directions |
| `(n, lr) ml:l_ogistic (X, y)` | `(Int, Float) -> (Float, Int) -> Float` | a softmax's weights (a row per column of X, the bias last; a column per label) after n steps of gradient descent on the cross-entropy at rate lr |
| `W ml:p_redict X` | `Num a => a -> a -> Int` | the label the weights W give each row of X |

## Examples

From `../tests/basics.xtl`, with six points in two groups,
`X := 6 2 r_eshape 0.0 0.0 1.0 0.0 0.0 1.0 9.0 9.0 10.0 9.0 9.0 10.0`
and `y := 1 1 1 2 2 2`:

```
      C0 := 2 ml:f_arthest X
      C0
 0.0 0.0
10.0 9.0
      C1 := X ml:k_step C0
      C1
0.3333333333333333 0.3333333333333333
 9.333333333333334  9.333333333333334
      C1 ml:a_ssign X
1 1 1 2 2 2
      (3, X, y) ml:k_nn 2 2 r_eshape 2.0 2.0 8.0 8.0
1 2
      (mu, V, ev) := 2 ml:p_ca X
      V
0.7071067811865475 -0.7071067811865476
0.7071067811865476  0.7071067811865476
      W := (100, 0.5) ml:l_ogistic (X, y)
      W ml:p_redict X
1 1 1 2 2 2
```

The first principal direction is the diagonal the two groups lie
along; its variance is 40.6 of the total 40.9.

`../tests/checks.xtl` checks the properties on three blobs and on
points along a known line: k-means finds the blobs' centers and a step
never raises its inertia; farthest-first starts in three different
blobs; with k = 1 every training point gets its own label; PCA's
directions are unit length and at right angles (to 1e-9), the first
is the line the points lie along, and the mean projects to the
origin; logistic regression reads the three blobs right. The
library's `## >>` examples are run by `just doc-test`.

## Demos

- [`demos/blobs.xtl`](../demos/blobs.xtl): the four models on three
  blobs of 20 points.
- [`../../../demos/k-means`](../../../demos/k-means/README.md): k-means
  live, the centers moving step by step.

## Limits

- kNN finds the k nearest by taking the nearest and masking it, k
  times: X_eTaL has no grade along a row yet (ask M4 in
  [`docs/xetal-asks.md`](../../../docs/xetal-asks.md)).
- PCA uses power iteration (200 steps a direction) with deflation:
  accurate for directions whose variances differ; two nearly equal
  variances give a direction somewhere in their plane.
- Distances use |a|^2 + |b|^2 - 2 a.b, one matrix product for all
  pairs; it rounds to about 1e-12 of the squared sizes (an inertia of
  exactly 8/3 prints as 2.666666666666819).

## Provenance

Written for X_eTaL-ML after the model list of APLearn (an APL machine
learning library on the APL Wiki's machine-learning page); the
algorithms are the textbook ones (Lloyd's k-means; Gonzalez's
farthest-first start; principal components by power iteration with
Hotelling's deflation; multinomial logistic regression by gradient
descent), reimplemented, not ported.

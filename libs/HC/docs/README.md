# HC

Hyper-Connections and mHC. A residual stream widened into S streams,
mixed by small tables; mHC projects the table that carries the state
onto the doubly stochastic matrices (Sinkhorn-Knopp), so the gain
stays bounded at any depth. The library has the projection and its
gradient, one layer and its backward pass in two halves around the
block, the start the mHC-poc Python uses, the two gain measures by
depth, and the paper's input-dependent tables with their gradient.

```
"hc:" u_se< "HC"
```

Put each library's `src/` directory on `XETAL_PATH` (`just path`
prints them; see the [README](../../../README.md)); `hc:` is the
recommended alias.

## Conventions

- A stream state X is S x N, a stream a row: N is the width, or B
  times the width for a batch (each example's numbers side by side).
- H_res is S x S and H_post a list of S. H_pre is either an S x S
  table, whose block input is the mean over streams of H_pre X (the
  static form of the mHC-poc Python), or a list of S, whose block
  input is H_pre X (the paper's eq. 3, and what `hc:d_ynamic` gives).
- The block `f` takes and gives a list of N. A block of a batch
  reshapes it to B x D itself.
- The layer's next state is `Hres X + Hpost' F`, where F = f m and m
  is the block's input.
- Backward: `hc:b_ackTop` goes from the gradient G at the next state
  to the gradient at the block's output (dF). The caller carries dF
  back through its own block to dm. `hc:b_ackBottom` goes from dm to
  H_pre and X. The gradient by X is dXres + dXpre, plus
  `hc:d_ynamicBack`'s part when the tables depend on X.

## Functions

| Function | Type | What |
| -------- | ---- | ---- |
| `n hc:s_inkhorn M` | `Int -> Float -> Float` | n rounds of Sinkhorn-Knopp on a positive square M, rows then columns normalized (the paper's eq. 9): a doubly stochastic matrix |
| `(n, M) hc:s_inkhornBack G` | `(Int, Float) -> Float -> Float` | the gradient by M, given G, the gradient by `n hc:s_inkhorn M`: the rounds run again and undone from the last |
| `('f, Hpre, Hpost, Hres) hc:l_ayer X` | `(Float -> Float, Float, Float, Float) -> Float -> (Float, Float, Float)` | one layer: (the next state, the block's input m, the block's output F) |
| `(Hpost, Hres) hc:b_ackTop (X, F, G)` | `Num a => (a, a) -> (a, a, a) -> (a, a, a, a)` | (dXres, dF, dHres, dHpost) from G: `Hres' G`, `Hpost G`, `G X'`, `G F` |
| `Hpre hc:b_ackBottom (X, dm)` | `Float -> (Float, Float) -> (Float, Float)` | (dXpre, dHpre) from dm, the gradient by the block's input |
| `hc:s_tart S` | `Int -> Float` | the Python's starting logits, `2 I - 1`; HC's H_res is their exponential, mHC's `20 hc:s_inkhorn` of that |
| `hc:g_ainProxy Hs` | `Box Float -> Float` | the Python's proxy: log10 of the largest absolute entry of the product of the chain Hs (boxed tables, the first layer's first) |
| `hc:a_maxGain Hs` | `Box Float -> (Float, Float)` | the paper's Amax Gain Magnitude of the product: (the largest absolute row sum, the largest absolute column sum), forward and backward |
| `(alpha, phi, b) hc:d_ynamic X` | `(Float, Float, (Float, Float, Float)) -> Float -> (Float, Float, Float)` | the paper's tables from X (eq. 7 and 8): X flattened and scaled to an RMS of 1, projected by phi (S C x (2 S + S S)), gated by alpha (three numbers), biased by b = (bpre, bpost, bres); H_pre = sigmoid, H_post = 2 sigmoid, H_res = 20 rounds of Sinkhorn-Knopp of the exponential |
| `(alpha, phi, b) hc:d_ynamicBack (X, dHpre, dHpost, dHres)` | `(Float, Float, (Float, Float, Float)) -> (Float, Float, Float, Float) -> (Float, Float, (Float, Float, Float), Float)` | (dalpha, dphi, (dbpre, dbpost, dbres), dX) through the tables; dX is the part through the tables only |

## Examples

From `../tests/basics.xtl`:

```
      20 hc:s_inkhorn e_xp hc:s_tart 4
 0.7112345942275938 0.09625513525746872 0.09625513525746872 0.09625513525746872
0.09625513525746872  0.7112345942275938 0.09625513525746872 0.09625513525746872
0.09625513525746872 0.09625513525746872  0.7112345942275938 0.09625513525746872
0.09625513525746872 0.09625513525746872 0.09625513525746872  0.7112345942275938
      hc:g_ainProxy 12 r_eshape e_nclose e_xp hc:s_tart 4
6.389115385966659
      hc:a_maxGain 12 r_eshape e_nclose 20 hc:s_inkhorn e_xp hc:s_tart 4
(0.9999999999999999, 1.0)
```

A layer with its gradients; the block doubles its input:

```
u:t_wice := { m -> 2.0 * m }
(X2, m, F) := ('u:t_wice, Hpre, Hpost, Hres) hc:l_ayer X
(dXres, dF, dHres, dHpost) := (Hpost, Hres) hc:b_ackTop (X, F, G)
(dXpre, dHpre) := Hpre hc:b_ackBottom (X, 2.0 * dF)
dX := dXres + dXpre
```

## Checked

`../tests/checks.xtl`, 22 checks:

- **Sinkhorn-Knopp:** from `exp(2 I - 1)`, rows and columns sum to 1,
  with 0.711 on the diagonal and 0.096 off it.
- **Central differences:** each of these agrees with the central
  differences of a probe, to 1e-6 relative:
  - `hc:s_inkhornBack`, at 5 and at 20 rounds;
  - the layer's gradients by X, H_pre (a table and a list), H_post
    and H_res;
  - `hc:d_ynamicBack`, by alpha, phi, the three biases and X;
  - a whole input-dependent layer by X, its three paths added.
- **The gains at the start:** the numbers of X_eTaL-fine-tuning's
  `docs/mhc-plan.md`.
  - HC's proxy is 6.39, 13.37 and 27.35 at 12, 24 and 48 layers.
  - mHC's proxy is log10 1/4 at 48 layers.
  - HC's Amax gain over 12 layers is (e + 3/e)^12 both ways; mHC's
    is 1.

## Demos

[`demos/depth.xtl`](../demos/depth.xtl) is the gain by depth at the
start, for 1 to 64 layers. Each table is a product of L copies of
H_res. Its columns are:
- L;
- HC's proxy and its Amax gain (log10), which grow as L log10(e +
  3/e), 0.58 a layer;
- mHC's proxy, which falls to log10 1/4 = -0.602 as the streams are
  averaged;
- mHC's Amax gain, which stays at 1.

## Provenance

- **Asked for by** X_eTaL-fine-tuning (ask FM13, its
  `docs/xetal-ML-asks.md`). Its depth track wrote all of this out in
  its programs, and the signatures here are FM13's. One difference: the
  block takes and gives a list of N, not a B x D matrix, since the
  layer cannot know D.
- **The papers:**
  - Hyper-Connections: Zhu et al., arXiv 2409.19606.
  - mHC: Xie et al., DeepSeek-AI, arXiv 2512.24880. The Sinkhorn
    projection is its eq. 9, the layer its eq. 3, the input-dependent
    tables its eq. 7 and 8, and the Amax gain its section 3.1.
- **The static form, the start `2 I - 1` and the proxy** are the
  mHC-poc Python's (github.com/softwarewrighter/mHC-poc, the same
  author).
- **The input-dependent tables** follow X_eTaL-fine-tuning's
  `dynamic-mixing.xtl`. Their RMS scaling has no epsilon, as there.

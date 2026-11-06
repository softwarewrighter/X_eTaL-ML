# MoE

Mixture-of-experts layers: a router scores each token against the
experts, each token keeps its k most likely ones, and the layer's
output is their outputs weighted by the gates, each token through its
chosen experts only. An expert is an adapted projection, the LoRA
library's W x + s B (A x). The gradients with the selection fixed, and
the routing statistics: how often each expert is chosen, the routing's
entropy, the load-balancing loss.

```
"moe:" u_se< "MoE"
```

Put each library's `src/` directory on `XETAL_PATH` (`just path`
prints them; see the [README](../../../README.md)); `moe:` is the
recommended alias. The demos and live pages of this repository find
the libraries without setup.

## Conventions

- Tokens are columns, as in the LoRA library: x is n x N, N tokens of
  n numbers.
- The router R is E x n: `R moe:s_cores x` is E x N, a column of
  scores per token; `moe:s_oftmax` turns each column into
  probabilities over the experts.
- `k moe:t_opk P` keeps each token's k most likely experts and
  renormalizes their probabilities over those k (as Mixtral does): the
  gates G, E x N, 0 where an expert is not chosen. A tie keeps every
  expert at the tie.
- The experts are stacked, `(Ws, As, Bs, s)`: Ws E x m x n, As E x r
  x n, Bs E x m x r, one scale s; expert e (from 1) is `(e s_elect Ws,
  e s_elect As, e s_elect Bs, s)`, an adapter of the LoRA library.
- The layer is y = sum over the chosen e of G_e (W_e x + s B_e (A_e
  x)). `moe:c_ombine` runs each expert only on the tokens that chose
  it (picked out as columns by a one-hot matrix and put back);
  `moe:d_ense` runs every expert on every token, the reference.
- The top-k is not differentiable: `moe:g_rads` gives the gradients
  with each token's choice held fixed, as training frameworks do. The
  gates are then a softmax over each token's chosen experts, so for a
  chosen e, dL/dS_e = G_e (h_e - sum_i G_i h_i), h_e = g . y_e, the
  loss's gradient by that gate; the router's gradient is dS x'. An
  expert's gradients are the LoRA library's with g times its gates.
- Every function takes its data on the right, so each is a Flow stage:
  `@ flow:p_ipe< c"x |> R moe:s_cores it |> moe:s_oftmax it |> 2
  moe:t_opk it |> (ex, it) moe:c_ombine x"` is the layer.

## Functions

| Function | Type | What |
| -------- | ---- | ---- |
| `R moe:s_cores x` | `Num a => a -> a -> a` | each token's score for each expert, R x |
| `moe:s_oftmax S` | `Num a => a -> Float` | each column into probabilities over the experts |
| `k moe:m_ask P` | `Int -> Float -> Float` | 1 at each token's k most likely experts |
| `k moe:t_opk P` | `Int -> Float -> Float` | the gates: the k most likely, renormalized over them |
| `experts moe:a_pply (e, x)` | `Num a => (a, a, a, a) -> (Int, a) -> a` | expert e on the tokens x |
| `(experts, G) moe:c_ombine x` | `((Float, Float, Float, Float), Float) -> Float -> Float` | the layer for the gates G, each expert on its tokens only |
| `(experts, G) moe:d_ense x` | `((Float, Float, Float, Float), Float) -> Float -> Float` | the same with every expert on every token: the reference |
| `(R, experts, k) moe:l_ayer x` | `(Float, (Float, Float, Float, Float), Int) -> Float -> Float` | the whole layer: scores, probabilities, top-k gates, the sparse combine |
| `(R, experts, k) moe:g_rads (x, g)` | `(Float, (Float, Float, Float, Float), Int) -> (Float, Float) -> (Float, Float, Float, Float)` | (dR, dAs, dBs, dx), the selection fixed |
| `moe:f_requency G` | `Float -> Float` | the fraction of tokens choosing each expert (they sum to k) |
| `moe:e_ntropy P` | `Float -> Float` | the mean entropy of the tokens' probabilities, in nats (log E when the router has no preference) |
| `(P, G) moe:b_alance k` | `(Float, Float) -> Int -> Float` | the Switch Transformer's load-balancing loss, E sum_e f_e P_e: 1 when even, E when one expert takes all |

## Examples

From `../tests/basics.xtl` and the doc comments:

```
      2 moe:t_opk 3 1 r_eshape 0.5 0.3 0.2
              0.625
0.37499999999999994
                0.0
      moe:f_requency 2 4 r_eshape 1.0 0.5 0.0 1.0 0.0 0.5 1.0 0.0
0.75 0.5
```

`../tests/checks.xtl` checks, on 4 experts (6 inputs to 5 outputs, rank
2) and 7 tokens at top 2:

- each token's probabilities sum to 1; each keeps 2 experts whose
  gates sum to 1, each its probability over the two's sum;
- the sparse combine is the dense reference, and the layer is the
  combine of its gates; an expert is LoRA's layer; with every expert
  chosen (top 4) the layer is the probability-weighted sum of all;
- the router's gradient, every expert's dA and dB and dx are the
  central finite differences of half the squared error (a small bump
  keeps every choice), in their arrays' shapes; experts no token chose
  (a router biased against them) get no gradient, nor their router
  rows;
- the frequencies sum to k; the entropy is between 0 and log E, log E
  for even routing and 0 for certain; the balance is 1 when even and
  E when one expert takes everything;
- the layer as a Flow pipeline of the library's stages is the layer.

## Demos

- [`demos/routing.xtl`](../demos/routing.xtl): 8 experts (8 x 8
  projections at rank 2) and 32 tokens at top 2: each expert's share
  of the tokens, the routing's entropy (1.37 nats, against log 8 =
  2.08 for no preference) and balance (1.04), the sparse layer equal
  to the dense one, and 64 expert passes against 256.

## Provenance

Written for X_eTaL-ML from the mixture-of-experts equations of
`docs/fine-tune-moe-research.txt` (lines 3180-3274: the top-k router,
the weighted combine of adapted experts), with the gates of Mixtral
(Jiang et al., 2024: softmax over the top k) and the load-balancing
loss of the Switch Transformer (Fedus et al., 2021). The router
gradient with the selection fixed is derived here and checked by
finite differences. The repository's `demos/moe-router` demo routes
words the same way, with its own code, before this library.

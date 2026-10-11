# Attention

Attention as three array expressions: every query against every key
(S = Q K^T / sqrt d), each row of scores into weights under a mask
(softmax), the weighted sum of the values (Y = A V).

```
"at:" u_se< "Attention"
```

Put each library's `src/` directory on `XETAL_PATH` (`just path`
prints them; see the [README](../../../README.md)); `at:` is the
recommended alias. Attention imports [NN](../../NN/docs/README.md)
(for softmax). The demos and live pages of this repository find the
libraries without setup.

## Conventions

- Tokens X are a matrix, a row per token.
- A head's projections Wq, Wk and Wv have a row per feature of X: Wq
  and Wk give queries and keys of d numbers, Wv the values.
- A mask is n x n: 1 where a token may look at another, 0 where not.
- The projections travel together as a tuple, `(Wq, Wk, Wv)`; the
  tokens and their mask as another, `(X, M)`.

## Functions

| Function | Type | What |
| -------- | ---- | ---- |
| `(Wq, Wk) at:s_cores X` | `(Float, Float) -> Float -> Float` | every query against every key over the square root of their length: n x n |
| `M at:w_eights S` | `Float -> Float -> Float` | each row of scores into weights summing to 1 where the mask allows; 0 where not |
| `at:c_ausal n` | `Int -> Float` | the causal mask: each token sees itself and those before it |
| `at:a_ll n` | `Int -> Float` | every token sees every token |
| `(Wq, Wk, Wv) at:h_ead (X, M)` | `(Float, Float, Float) -> (Float, Float) -> Float` | one head: the masked weights of the scores times the values X Wv |

## Examples

From the library's `## >>` examples (run by `just doc-test`):

```
      at:c_ausal 3
1.0 0.0 0.0
1.0 1.0 0.0
1.0 1.0 1.0
      (at:c_ausal 2) at:w_eights 2 2 r_eshape 0.0 5.0 1.0 1.0
1.0 0.0
0.5 0.5
```

`../tests/checks.xtl` checks the properties on six tokens: each row
of weights sums to 1, with or without the causal mask; a masked place
gets 0 and the first token sees only itself; the scores are Q K^T over
the square root of the keys' length; a head is the weights times the
values; reordering the tokens reorders a head's output the same way.

## Demos

- [`demos/previous.xtl`](../demos/previous.xtl): a head that looks one
  token back: each token takes on its predecessor's value.
- [`../../../demos/attention`](../../../demos/attention/README.md): one
  hand-set head on a sentence: "tired" finds the animal, "wide" the
  street.

## Provenance

Written for X_eTaL-ML from the attention demo's two lines,
generalized to any head width. The method is scaled dot-product
attention (Vaswani et al., "Attention is all you need", 2017), with
masking by a large negative score before the softmax.

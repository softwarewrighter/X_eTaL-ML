# Sample

Choosing the next token from a model's scores: the greedy choice,
probabilities at a temperature, only the top k kept, and a draw by
uniform numbers the caller gives.

```
"sm:" u_se< "Sample"
```

Put each library's `src/` directory on `XETAL_PATH` (`just path`
prints them; see the [README](../../../README.md)); `sm:` is the
recommended alias. Sample imports [NN](../../NN/docs/README.md) (for
softmax and argmax). The demos and live pages of this repository find
the libraries without setup.

## Conventions

- Scores (logits) and probabilities are a row per position, a column
  per token; tokens count from 1.
- The library draws no chance: `d_raw` takes the uniform numbers (one
  per row, in 0 .. 1), so the caller chooses their source (`r_oll!`
  under a seed, or evenly spread numbers in a test) and the same
  numbers give the same tokens.

## Functions

| Function | Type | What |
| -------- | ---- | ---- |
| `sm:g_reedy s` | `Num a => a -> Int` | each row's most likely token |
| `t sm:p_robs s` | `Num a => a -> a -> Float` | softmax of the scores over the temperature t |
| `k sm:t_op p` | `Int -> Float -> Float` | each row's k largest probabilities kept and renormalized, the rest 0 (ties at the k-th all kept) |
| `u sm:d_raw p` | `(Num a, Num b, Truthy b) => a -> a -> b` | a token per row: the first whose running sum passes that row's uniform number |

## Examples

From `../tests/basics.xtl`:

```
      sm:g_reedy 2 3 r_eshape 0.1 2.0 0.5 3.0 0.0 1.0
2 1
      2 sm:t_op 0.1 0.5 0.15 0.25
0.0 0.6666666666666666 0.0 0.3333333333333333
      0.1 0.7 sm:d_raw 2 3 r_eshape 0.2 0.3 0.5 0.2 0.3 0.5
1 3
```

`../tests/checks.xtl` checks the properties on scores over ten
tokens: probabilities sum to 1; a low temperature is the greedy
choice and a high one nearly even; top-1 is the greedy token; top-3
keeps three a row summing to 1; evenly spread uniform numbers draw
each token its share exactly.

## Demos

- [`demos/next.xtl`](../demos/next.xtl): a model's scores over six
  words: greedy, three temperatures, top-2, twenty draws.

## Limits

- `t_op` finds each row's k-th largest by taking the largest and
  masking it, k times: X_eTaL has no grade along a row yet (ask M4 in
  [`docs/xetal-asks.md`](../../../docs/xetal-asks.md)).

## Provenance

Written for X_eTaL-ML for the tiny GPT; the methods are the usual
ones for sampling from a language model (temperature; top-k as in
Fan, Lewis and Dauphin, 2018).

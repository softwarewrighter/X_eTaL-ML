# Net

A network written as one line. `Net` is a macro library (`.xtlm`): its
macros read a spec such as `"784 128 relu 10 softmax"` when the
program is compiled and write ordinary X_eTaL in its place, which is
then type-checked like any code. `xetal expand` shows what was
written, so the notation hides nothing.

```
"nn:"  u_se< "NN"
"net:" u_se< "Net"
```

Put each library's `src/` directory on `XETAL_PATH` (`just path`
prints them; see the [README](../../../README.md)); `net:` is the
recommended alias. Import [NN](../../NN/docs/README.md) as `nn:`: the
code the macros write calls NN under that alias (see Limits).

## A spec

A spec is a list of words, the input's size first:

| Word | Means |
| ---- | ----- |
| the first number | how many inputs an example has |
| a later number | a dense layer with that many outputs (`nn:d_ense`), using the next weight array |
| `relu`, `sigmoid`, `tanh`, `softmax`, `logSoftmax` | that function of NN, applied to the layer before |

A dense layer's weights are one array, as NN keeps them: from m inputs
to n outputs it is m + 1 by n, the bias its last row.

## Macros

| Macro | Type | What |
| ----- | ---- | ---- |
| `"spec" net:n_etwork< "w1 w2 ..."` | `Char -> Char -> Char` | a forward function for the spec's network, using the weight arrays named, one per dense layer in order |
| `"spec" net:p_arams< @` | `Char -> Unit -> Char` | how many numbers the network has to learn, worked out when the program is compiled |
| `"spec" net:s_hapes< "w1 w2 ..."` | `Char -> Char -> Char` | whether the weight arrays have the shapes the spec says: 1 or 0 |

(The types are the macros' own: text in, text out. What a call gives
is the code written, shown below.)

## Examples

From `../tests/basics.xtl`, with `w1` and `w2` each 3 by 2 and `x` the
four points of XOR:

```
      u:x_or := "2 2 relu 2 softmax" net:n_etwork< "w1 w2"
      nn:a_rgmax u:x_or x
1 2 2 1
      "2 2 relu 2 softmax" net:p_arams< @
12
      "784 128 relu 10 softmax" net:p_arams< @
101770
      "2 2 relu 2 softmax" net:s_hapes< "w1 w2"
1
      "2 3 relu 2 softmax" net:s_hapes< "w1 w2"
0
```

What those lines became (`xetal expand`, recorded in
`../tests/expand-basics.out`):

```
u:x_or := ({ g1:x -> nn:s_oftmax (nn:r_elu g1:x nn:d_ense w1) nn:d_ense w2 })
12
101770
((s_hape w1) m_atch 3 2) & ((s_hape w2) m_atch 3 2)
((s_hape w1) m_atch 3 3) & ((s_hape w2) m_atch 4 2)
```

The network is the function you would write by hand (`g1:x` is its
parameter, renamed by X_eTaL so it cannot clash with a name of yours).
`../tests/checks.xtl` checks that for relu, sigmoid and tanh networks
the macro's function gives the same numbers as the hand-written one.

A wrong spec is refused when the program is compiled, at the call:

```
error[bad-macro-argument]: not a size or an activation (relu, sigmoid, tanh, softmax, logSoftmax): gelu at 146..156
error[bad-macro-argument]: name one weight array per dense layer: 2 for this spec, 1 given at 190..193
```

## Demos

- [`demos/xor.xtl`](../demos/xor.xtl): XOR with the network as one
  line (`just demo-lib Net xor`), and what the macros wrote
  (`just expand-lib Net xor`).

## Limits

- The written code calls NN as `nn:`, so NN must be imported under
  that alias: a macro's text cannot ask which alias the caller chose,
  and may not import a library itself (ask M11 in
  [`docs/xetal-asks.md`](../../../docs/xetal-asks.md)).
- Dense layers and NN's activations only: a convolution or attention
  layer is written by hand, as the demos do.
- `net:s_hapes<` checks when the program runs (shapes are not part of
  X_eTaL's types); the spec's own mistakes are caught when it is
  compiled.

## Provenance

Written for X_eTaL-ML. The idea of a network as a layer list is as in
Keras's `Sequential` and PyTorch's `nn.Sequential`; here it is a
macro, so the layer list becomes plain code before the program runs.
The word-splitting helpers follow X_eTaL's own `Combinators.xtlm`.

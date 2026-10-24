# Activations

Activation functions, a derivative for each, and the gated layers
(GLU, ReGLU, GEGLU, SwiGLU). Every activation is one short array
expression that works item by item on a whole batch.

```
"act:" u_se< "Activations"
```

Put each library's `src/` directory on `XETAL_PATH` (`just path`
prints them; see the [README](../../../README.md)); `act:` is the
recommended alias. The demos and live pages of this repository find
the libraries without setup.

## Conventions

- An activation works item by item on a Float array of any shape: a
  vector, a batch (one example per row), a batch of pictures.
- A setting comes first: `0.1 act:l_eaky x`, `1.5 act:s_wish x`,
  `1.0 act:e_lu x`. PReLU's slopes are one per column, the last axis
  (as a bias is), and its vector must be as long as that axis.
- Each function f has its derivative `d_f`, written in terms of f's
  **input** (`act:d_gelu z` for `act:g_elu z`), not its output. X_eTaL
  has no automatic differentiation: a training step multiplies the
  error coming back by `d_f` of the layer's pre-activation. (ReLU,
  tanh and sigmoid can also be written at the output, `1 - A * A`, as
  the backprop demos do; the smooth ones cannot.)
- A gated layer takes its two weight arrays as a tuple,
  `(W, V) act:s_wiglu x`: the activation on `x W`, times the linear
  `x V` (Shazeer 2020's order). `x` is a batch, W and V both inputs by
  hidden; no bias (LLaMA has none; add one with `nn:b_ias`).
- Everything is Float: `act:r_elu` of an Int array is a type error;
  write `act:r_elu f_loat x`.
- tanh and erf are built here from `e_xp`, since X_eTaL has no
  `t_anh` or `e_rf` builtin ([docs/xetal-asks.md](../../../docs/xetal-asks.md)
  M19): tanh as `2 sigmoid(2x) - 1`, which does not overflow at either
  end but loses digits near 0 (about 1e-6 relative at 1e-10); erf by
  Abramowitz and Stegun 7.1.26, within 1.5e-7 (so `d_geluErf 0.0` is
  0.5000000005, not 0.5). Both are replaced the day the builtins, or
  X_eTaL-libraries' Trig and Special
  ([docs/xetal-libraries-asks.md](../../../docs/xetal-libraries-asks.md)
  E4, E5), land.

## Functions

### Activations

| Function | Type | What |
| -------- | ---- | ---- |
| `act:r_elu x` | `Float -> Float` | max(x, 0) |
| `a act:l_eaky x` | `Float -> Float -> Float` | leaky ReLU: x where positive, a x elsewhere |
| `a act:p_relu x` | `Float -> Float -> Float` | PReLU: leaky ReLU with a slope per column (a's length is x's last extent) |
| `a act:e_lu x` | `Float -> Float -> Float` | ELU: x where positive, a (e^x - 1) elsewhere; bounded below by -a |
| `act:s_elu x` | `Float -> Float` | SELU: 1.0507 times an ELU with alpha 1.6733, the self-normalizing pair |
| `act:s_igmoid x` | `Float -> Float` | 1 / (1 + e^-x) |
| `act:t_anh x` | `Float -> Float` | the hyperbolic tangent (see the note above) |
| `act:s_oftplus x` | `Float -> Float` | log(1 + e^x), as max(x, 0) + log(1 + e^-abs x): no overflow |
| `act:s_ilu x` | `Float -> Float` | x sigmoid(x); Swish with beta 1 |
| `b act:s_wish x` | `Float -> Float -> Float` | x sigmoid(b x) |
| `act:g_elu x` | `Float -> Float` | GELU, the tanh form: x/2 (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3))) |
| `act:g_eluErf x` | `Float -> Float` | the exact GELU, x Phi(x) = x/2 (1 + erf(x / sqrt 2)) |
| `act:m_ish x` | `Float -> Float` | x tanh(softplus(x)) |
| `act:h_ardSigmoid x` | `Float -> Float` | min(max(x + 3, 0), 6) / 6: three straight lines |
| `act:h_ardSwish x` | `Float -> Float` | x hardSigmoid(x) |

### The gated layers

| Function | Type | What |
| -------- | ---- | ---- |
| `(W, V) act:g_lu x` | `(Float, Float) -> Float -> Float` | sigmoid(x W) * (x V) |
| `(W, V) act:r_eglu x` | `(Float, Float) -> Float -> Float` | relu(x W) * (x V) |
| `(W, V) act:g_eglu x` | `(Float, Float) -> Float -> Float` | gelu(x W) * (x V) |
| `(W, V) act:s_wiglu x` | `(Float, Float) -> Float -> Float` | silu(x W) * (x V) |

### Derivatives, in terms of the input

| Function | Type | What |
| -------- | ---- | ---- |
| `act:d_relu x` | `Float -> Float` | 1 where x > 0, else 0 |
| `a act:d_leaky x` | `Float -> Float -> Float` | 1 where x > 0, else a |
| `a act:d_prelu x` | `Float -> Float -> Float` | 1 where x > 0, else the column's slope |
| `a act:d_elu x` | `Float -> Float -> Float` | 1 where x > 0, else a e^x |
| `act:d_selu x` | `Float -> Float` | lambda where x > 0, else lambda alpha e^x |
| `act:d_sigmoid x` | `Float -> Float` | s (1 - s), s = sigmoid(x): at most 1/4 |
| `act:d_tanh x` | `Float -> Float` | 1 - tanh(x)^2 |
| `act:d_softplus x` | `Float -> Float` | sigmoid(x) |
| `act:d_silu x` | `Float -> Float` | s (1 + x (1 - s)), s = sigmoid(x) |
| `b act:d_swish x` | `Float -> Float -> Float` | s + b x s (1 - s), s = sigmoid(b x) |
| `act:d_gelu x` | `Float -> Float` | the tanh form's derivative |
| `act:d_geluErf x` | `Float -> Float` | Phi(x) + x phi(x), the normal CDF and density |
| `act:d_mish x` | `Float -> Float` | t + x (1 - t^2) sigmoid(x), t = tanh(softplus(x)) |
| `act:d_hardSigmoid x` | `Float -> Float` | 1/6 where abs x < 3, else 0 |
| `act:d_hardSwish x` | `Float -> Float` | (2x + 3)/6 where abs x < 3; 0 below, 1 from 3 on |

At a kink (ReLU at 0, the hard ones at -3 and 3) a derivative takes
one side, as stated; the tests check every derivative against central
differences away from the kinks, within 1e-5.

## Comparing them

What the curves do, and what it costs to compute each (the
transcendental calls per element; the arithmetic is one pass per
operator either way). "Dips" is the lowest value the function
reaches, measured over -6..6 by `demos/curves.xtl`.

| Activation | e_xp, l_og per element | Monotone | Bounded below | Dips to | Slope's range | Used by |
| ---------- | ---------------------- | -------- | ------------- | ------- | ------------- | ------- |
| ReLU | 0 | yes | 0 | 0 | 0 or 1 (dead for x < 0) | most CNNs; the demos here |
| leaky ReLU, PReLU | 0 | yes | no | - | a or 1 | GANs; PReLU: ResNets (He et al. 2015) |
| ELU | 1 | yes | -a | -a | up to 1 | - |
| SELU | 1 | yes | -1.758 | -1.758 | up to 1.758 | deep MLPs without a norm layer |
| sigmoid | 1 | yes | 0 | 0 | 0 to 1/4 (vanishes) | gates (LSTM, GRU), outputs |
| tanh | 1 | yes | -1 | -1 | 0 to 1, zero-centered | small nets, RNNs |
| softplus | 1 + 1 | yes | 0 | 0 | 0 to 1 | a smooth ReLU; inside Mish |
| SiLU / Swish | 1 | no | yes | -0.278 | -0.10 to 1.10 | EfficientNet, LLaMA's gate |
| GELU | 1 (tanh form) | no | yes | -0.170 | -0.13 to 1.13 | BERT, GPT-2, ViT |
| Mish | 2 + 1 | no | yes | -0.309 | -0.11 to 1.09 | YOLOv4 |
| hard sigmoid, hard swish | 0 | hard swish no | yes | -0.375 | 0 to 1/6; -0.5 to 1.5 | MobileNetV3 |

The smooth, non-monotone ones (SiLU, GELU, Mish) win in large
networks by how they train, not by what they can represent; in a
small network (the spiral of the training demos, width 4) ReLU's dead
units and sigmoid's small slope are what one sees, and tanh, leaky
ReLU, ELU or SELU are the safe choices. A gated layer costs a second
matrix product: SwiGLU at the usual hidden width has 1.5 times the
feed-forward parameters, which is why LLaMA uses 2/3 of it.

## Examples

From `../tests/basics.xtl`, on `x := -3.0 -1.0 -0.5 0.0 0.5 1.0 3.0`:

```
      act:r_elu x
0.0 0.0 0.0 0.0 0.5 1.0 3.0
      act:g_elu x
-0.0036373920817729943 -0.15880800939172324 -0.15428599017485609 0.0 0.34571400982514394 0.8411919906082768 2.9963626079182273
      act:s_ilu x
-0.14227761953270035 -0.2689414213699951 -0.1887703343990727 0.0 0.3112296656009273 0.7310585786300049 2.8577223804673
      act:d_sigmoid x
0.04517665973091214 0.19661193324148185 0.2350037122015945 0.25 0.2350037122015945 0.19661193324148185 0.045176659730912
```

Two rows of three inputs through a SwiGLU layer with a hidden width
of two:

```
      X := 2 3 r_eshape 1.0 -2.0 0.5 0.0 1.0 -1.0
      W := 3 2 r_eshape 0.5 -0.5 1.0 0.25 -1.0 2.0
      V := 3 2 r_eshape 1.0 1.0 -0.5 2.0 0.0 -1.0
      (W, V) act:s_wiglu X
-0.4768116880884702                -0.0
-0.8807970779778823 -0.7772477896663696
```

## Tests

`../tests/checks.xtl`, 37 checks with Check: the relations between
the functions (leaky ReLU as ReLU plus the negative part, PReLU with
equal slopes as leaky ReLU, SiLU as Swish 1, sigmoid's symmetry, tanh
odd and inside -1..1), the bounds (ELU at -alpha, softplus above ReLU
and finite at -1000 and 1000, Mish finite, hard sigmoid inside 0..1),
where the smooth ones dip (GELU -0.17, SiLU -0.28, Mish -0.31; ReLU
never), the two GELUs within 5e-4 of each other and the exact one
agreeing with the Integrals demo's quadrature of the Gaussian density
to four places, hard swish within 0.15 of SiLU, SELU's fixed point
(4,000 standard normals from a Box-Muller grid keep mean 0 and
variance 1 within 0.01), the four gated layers written out, and every
derivative against central differences. `types.rgt` pins the exports'
types.

## Demos

- [`demos/curves.xtl`](../demos/curves.xtl): every curve as a table,
  thirteen points from -3 to 3, one row per activation, then their
  derivatives, then where the smooth ones dip.

## Provenance

Written for X_eTaL-ML, from the papers: ReLU as used by Nair and
Hinton (2010); leaky ReLU, Maas et al. (2013); PReLU, He et al.
(2015); ELU, Clevert et al. (2016); SELU and its constants, Klambauer
et al. (2017); GELU and SiLU, Hendrycks and Gimpel (2016); Swish,
Ramachandran et al. (2017); Mish, Misra (2019); hard sigmoid and hard
swish, Howard et al. (2019, MobileNetV3); GLU, Dauphin et al. (2017);
ReGLU, GEGLU and SwiGLU, Shazeer (2020). The erf approximation is
Abramowitz and Stegun, Handbook of Mathematical Functions, 7.1.26.

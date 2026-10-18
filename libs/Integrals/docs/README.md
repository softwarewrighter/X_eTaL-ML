# Integrals

A definite integral written as a paper prints it, computed by a
quadrature rule: one macro call gives the value, another the LaTeX.
The call is X_eTaL's Saga 42 shape (its decisions 2 to 4, taken with
the user 2026-10-17): `integral` read like `sum`, the rule and the
grid as option words with a documented default, improper integrals
refused. Decision 1 puts `integral` in X_eTaL-libraries' Equations in
the end; this library is the same call standing alone, to be lifted
whole when that lands (the body here is X_eTaL, there the equation
grammar).

```
"integ:" u_se< "Integrals"
```

Put the library's `src/` on `XETAL_PATH` (`just path` prints every
library's; see the [README](../../../README.md)); `integ:` is the
recommended alias.

## The macros

| Macro | What |
| ----- | ---- |
| `@ integ:i_ntegral< "integral tau:=a..b by trapezoid 64 \| body"` | the value of the integral by the rule, a Float |
| `@ integ:t_ex< "..."` | the same call's LaTeX, a string: `\int_{a}^{b} body \, d\tau` |

The left side is `@`; the whole integral is in the right string:

- **`integral tau:=a..b`**: the variable, a lowercase name, and its
  bounds, any X_eTaL giving a Float (`0.0..p_i @`); a whole-number
  literal is made Float (`0..1`). A bound of `inf` is refused: an
  improper integral needs a written truncation (`0..20`, with the tail
  shown small) or a change of variables, and the message says so.
- **`by trapezoid 64`**, optional: the rule (`trapezoid` or `simpson`)
  and the grid, how many steps; `simpson` needs an even number.
  Without `by`: `trapezoid 64`, so the plain form works.
- **`| body`**: X_eTaL over the variable, which the macro binds to the
  grid's points as one vector (a, a + h, ..., b: n + 1 points), so the
  body is evaluated once for the whole grid and may call any function
  (`e_xp tau`, `(u:b_ox tau) * u:b_ox t - tau`).
- **Every rule of the reader uses all of its text or refuses it**
  (`error[bad-integral]` at the string, naming the text): no `integral`
  word, no `:=`, no `..`, a missing bound or body, a rule the library
  does not have, an odd grid for simpson, no steps, anything after the
  grid (`by trapezoid 8 extra`).

What the rules are good for, shown by the tests: trapezoid is exact
on constants and lines, simpson on quadratics and cubics; on a smooth
function doubling the grid divides trapezoid's error by about 4 and
simpson's by about 16 (`../tests/convergence.xtl`). At a jump of the
body the sample takes the body's value at that point as written (a
box's edges count as inside), so an integrand with jumps is first
order: high by about half a step at each edge, the error falling like
h (`../tests/boxes.xtl`).

## The expansion

Ordinary X_eTaL (`xetal expand`), one definition per line: the bounds,
the step, the variable bound to the grid, the rule's weights, and the
step times the weighted sum of the body over the grid. The macro's own
names are fresh (`[]G_ENSYM`), the variable is the caller's:

```
{ @ ->
  g1:a := f_loat 0
  g2:b := f_loat 1
  g3:h := (g2:b - g1:a) / 2.0
  x := g1:a + g3:h * f_loat o_ffsets 3
  g4:w := (1.0 c_at (1 r_eshape 4.0 2.0) c_at 1.0) / 3.0
  g3:h * '+ r_/ g4:w * x ^ 2
} @
```

## Examples

From `../demos/gelu.xtl`, GELU at -3 -2 -1 -0.5 0 0.5 1 2 3 by the
integral, then the tanh approximation's error at those points and its
largest over -4..4, then the KL integral beside its closed form:

```
      (f_loat f_loor 0.5 + 10000.0 * g) / 10000.0
-0.004 -0.0455 -0.1587 -0.1543 0.0 0.3457 0.8413 1.9545 2.996
      (f_loat f_loor 0.5 + 1000000.0 * g - u:a_pprox xs) / 1000000.0
-0.000412 -0.000098 0.000153 0.000017 0.0 0.000017 0.000153 -0.000098 -0.000412
      (f_loat f_loor 0.5 + 1000000.0 * 'm_ax r_/ a_bs ('u:g_elu e_ach fine) - u:a_pprox fine) / 1000000.0
0.000473
      (f_loat f_loor 0.5 + 1000000.0 * kl c_at closed) / 1000000.0
0.443147 0.443147
```

From `../tests/exact.xtl`:

```
      @ integ:i_ntegral< "integral x:=0..1 by simpson 2 | x ^ 2"
0.3333333333333333
      @ integ:i_ntegral< "integral x:=0..2 by trapezoid 4 | 1.0 + 2.0 * x"
6.0
```

The orders (`../tests/convergence.xtl`): the error ratios when the
grid doubles, trapezoid 8/16, 16/32, then simpson, on e^x over 0..1:

```
      f_loor 0.5 + (t8 / t16) c_at (t16 / t32) c_at (s8 / s16) c_at s16 / s32
4 4 16 16
```

The convolution of two unit boxes at t = 0, 0.5, 1, 1.5, 2
(`../tests/boxes.xtl`): on 8 steps, then on 128, approaching the
triangle 0 0.5 1 0.5 0:

```
      'u:c_onv8 e_ach ts
0.125 0.625 1.125 0.75 0.25
      'u:c_onv128 e_ach ts
0.0078125 0.5078125 1.0078125 0.515625 0.015625
```

The LaTeX (`../tests/latex.xtl`), between `$$` for a page that draws
math:

```
      @ integ:t_ex< "integral tau:=0..1 | e_xp tau"
\int_{0}^{1} e_xp tau \, d\tau
      "$$" c_at (@ integ:t_ex< "integral tau:=0..2 by trapezoid 128 | (u:b_ox tau) * u:b_ox t - tau") c_at "$$"
$$\int_{0}^{2} (u:b_ox tau) * u:b_ox t - tau \, d\tau$$
```

An improper integral is refused (`../tests/reject-infinite.xtl`):

```
      @ integ:i_ntegral< "integral x:=0..inf | e_xp n_eg x"
error[bad-integral]: an improper integral (a bound of inf): write a truncation, 0..20 say, and show the tail is small, or change variables so the bounds are finite: 0..inf
```

## Demos

- [`demos/convolution.xtl`](../demos/convolution.xtl): the continuous
  convolution of two unit boxes as a function of t, a strip of 17
  values beside the triangle it converges to, with its LaTeX; the area
  under e^x by simpson on 16 steps beside e - 1.
- [`demos/gelu.xtl`](../demos/gelu.xtl): two continuous equations of
  machine learning. GELU, the activation of BERT and GPT, as the
  integral that defines it, x (1/2 + the integral of the Gaussian
  density from 0 to x), against the tanh approximation the frameworks
  compute: at 1 it is 0.8413, and the approximation is off by at most
  0.000473 over -4..4. The KL divergence of N(0, 1) from N(1, 2) as the
  integral of p log (p / q) over -10..10 beside its closed form: both
  0.443147.

## Tests

- `exact.xtl`: each rule on what it is exact for, the default rule,
  Float and expression bounds, the grid the variable is bound to.
- `convergence.xtl`: the orders, 4 and 16.
- `boxes.xtl`: the box convolution's convergence and the sampling
  policy at jumps; the triangle's area, the plan's oracle 1.0, by hand
  and by the macro.
- `latex.xtl`: the LaTeX.
- `reject-*.xtl`: one golden per refusal.
- `expand-*.rgt`: every program's expansion, pinned.

## Provenance

The call shape and the rules are X_eTaL's Saga 42 decisions
(`../X_eTaL/docs/plan.md`, "Saga 42", 2026-10-17): the integrator's
recommendations adopted with the user. Trapezoid and Simpson's rule
are the textbook composite rules; the box-convolution fixture and the
triangle's area are the oracle numbers of X_eTaL's design note
(`docs/convolutions/`). The reader follows X_eTaL-libraries' Equations:
every rule uses all of its text or refuses it.

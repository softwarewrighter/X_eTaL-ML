# Asks for X_eTaL-libraries

What the demos and libraries here need from the general libraries of
[X_eTaL-libraries](https://github.com/softwarewrighter/X_eTaL-libraries),
pinned by `XETAL_LIBRARIES_COMMIT` (`../X_eTaL-libraries`, read only
from here: its own plan takes them). The ML libraries live here; the
mathematics they stand on (trigonometry, special functions, the
equation grammar) belongs there, so that Statistics, Geometry and the
blog's notation share it. What X_eTaL itself must give is in
[`xetal-asks.md`](xetal-asks.md) (M19 is the one these depend on).

Each entry: status, kind, the ask, which demos or libraries here need
it, and the workaround in use. Status: **not yet taken** (filed here,
to be taken to that repository's plan), **planned** or **in progress**
(in its plan), **landed** (in the commit this repository pins, the
workaround removed), **declined** (with its reason). The ledger check
(`just check-asks`) covers X_eTaL's ledger only; these are kept in
step with `../X_eTaL-libraries/docs/plan.md` by hand at each libraries
re-pin.

## Equations: what the diffusion sampler found (2026-10-20)

Equations' rules are deliberate (a fixed table of functions, every
rule uses all of its text or refuses it), so these are asks, not
bugs, but for E1. Filed from the diffusion sampler of
`demos/pipelines/diffusion.xtl`, a paper's
`X0 = D(zT + sum_{t=T}^{1} [a_t z_t + b_t eps_theta(z_t, t, c) + sigma_t eps_rand] - z_t)`.

| # | Status | Kind | Ask | Demos, libraries | Workaround |
| - | ------ | ---- | --- | ---------------- | ---------- |
| E1 | landed | bug | A descending range, `sum t:=T..1 \| dz[t-1]`, is not refused by the reader: the expansion dies with `error[domain]: a count cannot be -1` at the call, which names no rule. Asked: the reader refuses it (`error[bad-equation]`), saying a range runs low..high, or reads it as the empty sum. Landed in its ba533ac (Saga 17 step 1, in the commit pinned here, 92ce936): the range checked when the program runs, refused by name | pipelines (diffusion) | none needed: the sampler writes the range upward, `1..T`, as a sum is read; a backward range is now refused saying so |
| E2 | planned | feature | A recurrence word: a sum whose term uses the one before (the sampler's sum from T down to 1, a scan) cannot be a `sum`, whose references are all to the caller's arrays. Asked: an operator word for it (`scan t:=T..1 \| ...` with the previous term named, or a documented refusal naming `p_ower`). Planned there as its Saga 17, step scan-word: the previous term named by a reserved word, `prev` (decided in its ede0da9) | pipelines (diffusion) | one step as the summand, an Equations call applied T times by `p_ower`; the differences collected and summed by a plain `sum` |
| E3 | planned | feature | The caller's functions by name: `D(z[k])` and `eps_theta(z[k], t, c)` are refused (`d_ecode is not a function of an equation (exp, log, sqrt, abs, max, min)`). Asked: a name in scope applied in an equation (the X_eTaL spelling, `u:d_ecode(z[k])`, or a declared table) calls it, drawn `\mathcal{D}(z_k)`. Planned there as its Saga 17: the program's own functions called by name, `u:name(args)`, a bad name or arity caught when the expansion compiles (decided in its ede0da9) | pipelines (diffusion) | the networks called outside the string, their results named (`eps`) and read inside it |

Seen at its HEAD (2026-10-20): Equations reads `integral` (Saga 42's
decision 1, this repository's Integrals as its oracle) and the
functions `sin` and `cos`, so the next re-pin lifts Integrals as
planned.

## Mathematics for the Activations library (2026-10-23)

Filed while planning `libs/Activations` (`act:`; `docs/plan.md`, Saga
5 rows 27 to 31): ReLU, sigmoid and tanh out of NN, with leaky ReLU,
PReLU, ELU, SELU, GELU, SiLU/Swish, Mish, softplus, hard sigmoid and
hard swish, the GLU family as gated layers, and a derivative for each
(X_eTaL has no automatic differentiation: the training demos and
Net's `t_rain<` write every slope by hand). What the library needs is
mathematics, not machine learning, so it is asked for there; until
each lands the library carries a private copy, named as the
workaround on its page.

| # | Status | Kind | Ask | Demos, libraries | Workaround |
| - | ------ | ---- | --- | ---------------- | ---------- |
| E4 | planned | feature | A Trig library (`tr:`): what X_eTaL's builtins (`s_in`, `c_os`, `a_tan`, `e_xp`, `l_og`, `p_i @`) leave out and a library can write in closed form: `t_an`, `a_sin`, `a_cos`, `a_tan2` (the sign cases), `s_inh`, `c_osh`, `t_anh`, `a_sinh`, `a_cosh`, `a_tanh`, `d_egrees`, `r_adians`, `h_ypot`; each with a doc example and a test against a known value, the range and accuracy limits of each form stated (tanh from `e_xp` cancels near 0: `t_anh` wraps the builtin when M19 lands; X_eTaL-libraries' own X22 asks for it). Planned there as its Saga 18 (trig) | Activations (`t_anh`, GELU's tanh form, Mish), NN until the move; Geometry and Statistics there | NN's `2 s_igmoid(2x) - 1` (no overflow at either end, about 1e-6 relative error at 1e-10) |
| E5 | planned | feature | A Special library (`sp:`) of the functions without a closed form: `e_rf`, `e_rfc` (a rational approximation, Abramowitz and Stegun 7.1.26 at 1.5e-7, or better), `g_amma` and `l_gamma` (Lanczos), `b_eta`, the normal `c_df` and `q_uantile`; each wrapping the builtin when M19 lands, so a caller never sees the approximation change. Planned there as its Saga 19 (special) | Activations (`g_eluErf`, the exact GELU), the Integrals demo `gelu.xtl` (today an integral by simpson, the oracle for the approximation); Statistics there | a private erf in Activations, tested against the integral |
| E6 | planned | feature | Equations' function table widened: `tanh`, `erf`, `sigmoid` (and `softplus`) as words of the grammar, written through Trig and Special (or the builtins) and drawn `\tanh`, `\operatorname{erf}`, `\sigma`; so the activations' formulas (`x * sigmoid(1.702 * x)`, `0.5 * x * (1 + erf(x / sqrt(2)))`) are one equation string each, their LaTeX from the same reading. Planned there as its Saga 20 (equations-activations) | Activations (its reference page: each function's formula, LaTeX and value from one string), pipelines (the composer's palette) | the activation written in X_eTaL beside a `$$` line typed by hand |
| E7 | planned | feature | A numeric derivative word, in the shape of `integral`: `derivative x:=x0 by central 1e-5 \| body` (central differences at x0; `forward` and the step as option words with a stated default; a vector x0 gives the derivative at each) and `\frac{d}{dx}` in LaTeX. Evaluation only, so any body the grammar reads has it today; the oracle for E8. Planned there as its Saga 21 (derivative) | Activations (every `d_*` checked against central differences, as backprop's `u:b_ump` does by hand), backprop, cnn-backprop, Net | the finite-difference lambda written out in each test |
| E8 | planned | feature | Symbolic differentiation, a fourth reading of the same tree (Equations parses through Parsing): `e:d_iff< "y \| x" "x^2 * exp(x)"` writes the derivative's X_eTaL and `e:d_tex<` its LaTeX, by the sum, product, quotient and chain rules, `^` with a constant exponent (the general case through exp and log), a derivative per function of the table (exp, 1/x, 1/(2 sqrt), cos, -sin, sign for abs, a selection for max and min, 1 - tanh^2, 2/sqrt(pi) e^(-x^2) for erf, sigma (1 - sigma)), `sum` termwise, `integral` with respect to a bound (the fundamental theorem) or under the sign; a small simplifier (0 * e, 1 * e, e + 0) so the result reads; a differential test against E7 at random points through Check. Polynomials' `d_erivative` is the precedent. Planned there as its Saga 22 (symbolic-diff) | Activations (its derivatives written by the macro and checked against the hand forms), the training demos | each derivative written by hand |
| E9 | planned | feature | Indexed gradients: the derivative of a scalar equation with respect to an array element, `e:g_rad< "dL/dW[i,j] \| N K" "L"` by the Kronecker-delta rules of index notation (a reference `W[i,j]` differentiated gives delta_{i,i'} delta_{j,j'}, the deltas contracted against the sums), so backpropagation is written from the loss. The ambitious one: a step of its own after E8. Planned there as its Saga 23 (indexed-gradients) | backprop, train-live, cnn-backprop, Net (`t_rain<` writes the slopes by a table of three activations) | the gradients written out by hand, checked against finite differences |

What is decided here (the user, 2026-10-23; `docs/plan.md`, settled
questions): nothing leaves the base language. `s_in`, `c_os`,
`a_tan`, `e_xp` and `l_og` are what libm gives at full accuracy, and a
library cannot write them as well, which is the rule M19 states for
what is a builtin; everything derivable from them (E4) and every
approximation (E5, until a builtin) is a library, so nothing is
duplicated: a library function wraps a builtin the day it exists.
There are no users yet, so the split could be changed either way; this
one needs no breaking change.

## Equations as pipelines: what the Pipelines page found (2026-10-28)

The Pipelines page shows each example as a paper prints it, as a
pipeline of equations drawn as math (Flow's `flow:m_ath<`), as its
expansion and as plain X_eTaL, and what each stage gives. The way in
is hand work twice over: the paper's equation, read inside out, is
rewritten by hand as stages read left to right, and the paper's
LaTeX is typed again in the grammar. These two asks would let an
equation from a paper (or from OCR, which gives LaTeX) become the
pipeline the page draws, each stage highlighted back in the original.

| # | Status | Kind | Ask | Demos, libraries | Workaround |
| - | ------ | ---- | --- | ---------------- | ---------- |
| E10 | planned | feature | An equation read as a pipeline of stages: `"Y[o,h,w] \| O Ho Wo" e:s_tages< "max((sum r:=0..2, sum s:=0..2 \| P[h+r,w+s] * F[o,r,s]) + B[o], 0)"` gives the pipeline's text, inside out turned left to right: each sum and each function applied over the indices a stage of its own equation, named (`C[o,h,w] \| O Ho Wo`), `\|>` between and the earlier stages' values by label, as Flow reads it (`flow:p_ipe<`, `flow:m_ath<`); and with it where each stage came from in the equation's text (start and end), so a page highlights the part of the original a stage computes. Tested: the pipeline's value equals the equation's (`e:e_q<`) on the demos' equations. Planned there as its Saga 24 (equation-stages) | pipelines (every example's pipeline of equations; the CNN layer's and the sampler's from the paper's line), micro-bert | each pipeline of equations written by hand beside its equation, checked equal (`equations.xtl`, `rosetta.xtl`, `diffusion.xtl`) |
| E11 | planned | feature | LaTeX in: `e:f_romTex<` reads the LaTeX `e:t_ex<` writes (subscripts as indices, `\sum_{i=0}^{n-1}`, `\frac`, `\sqrt`, `\exp`, `\log`, `\max`, `\left( \right)`, Greek names, `\,` and `\cdot` as products) into the grammar's text, refusing what the grammar does not have by name (as the grammar refuses a function it lacks); so an equation taken from a paper or from OCR is read as typed. Tested: `e:t_ex<` of what it reads gives back the LaTeX, on every equation the demos draw. Planned there as its Saga 25 (latex-in) | pipelines (the composer: paste a paper's LaTeX), micro-bert | the equation typed again in the grammar, its LaTeX compared by eye |

## Plot: graphing functions (2026-10-29)

Filed for the activations explorer (`docs/plan.md`, step
activations-explorer), after X_eTaL-libraries' `docs/research2.txt`
settled the split: plotting there, knowing nothing of networks;
activations here. Plot's line charts (`l_ine!`, `l_ines!`, `c_hart!`)
place a series' points by position (1, 2, ...), so y = f(x) over x in
-6..6 cannot be drawn with its x axis labeled -6, -3, 0, 3, 6. Its
agent is extending Plot (the same library and alias), not adding one.

| # | Status | Kind | Ask | Demos, libraries | Workaround |
| - | ------ | ---- | --- | ---------------- | ---------- |
| E12 | in progress | feature | Plot's line charts over real x values: a function's graph from explicit (xs, ys), `p:f_unction!` and several in one chart, `p:f_unctions!`, with a legend (f and its derivative, or a built-in beside a user's own); round-number ticks on both axes; a gap where a value is NaN or infinite rather than a line to it; the picture's SVG text given back, as `l_ine!` does, for a page to draw; and a linspace helper (n points from a to b). Its agent's word, 2026-10-29: extending Plot, not a new library | the activations explorer (every Activations function and its derivative over -6..6; a user's own activation), Activations' reference page | the curves demo prints tables of values (`libs/Activations/demos/curves.xtl`) |

## To take to X_eTaL-libraries

Nothing waiting: E1 to E9 are in its plan since 2026-10-23 (its
07ca150, Sagas 17 to 23), E10 and E11 since 2026-10-28 (its 8db355b,
Sagas 24 and 25); E1 has landed in the commit pinned here (92ce936);
E12 its agent is doing (2026-10-29), not yet a row of its plan. A new ask
goes in a table above as not yet taken, with the demos and libraries
here as its users, until that repository's plan takes it.

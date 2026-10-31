# Activation functions

Every activation of this repository's
[Activations](../../libs/Activations/docs/README.md) library and its
slope, graphed over x from -6 to 6 by X_eTaL, a family at a time:
classical (sigmoid, tanh, hard sigmoid), rectifiers (ReLU, leaky ReLU,
ELU, SELU), smooth (softplus, SiLU, Swish, GELU in its tanh form and
exactly, Mish, hard swish) and gated (GLU, ReGLU, GEGLU, SwiGLU as
one-number layers, W = V = 1: x times its gate). The charts are drawn
by [X_eTaL-libraries](https://github.com/softwarewrighter/X_eTaL-libraries)'
Plot, whose `p:f_unctionChart!` gives a chart an x axis of real
values.

Live: [Activation functions](https://softwarewrighter.github.io/X_eTaL-ML/activations/)
(pick a family; move a slider for leaky ReLU's slope, ELU's alpha or
Swish's beta and the charts are drawn again; write an activation of
your own and Run it).

## Run it

From a clone of this repository (Rust, git and `just`):

```bash
just xetal                    # fetch and build the pinned X_eTaL (once)
just run activations          # the eight charts (as SVG files) and each family's range
just serve activations        # the web app at http://127.0.0.1:8435/
just test-demo activations    # its CLI and browser baselines and the web app's tests
```

## The program

`activations.xtl` samples 241 x values (`p:l_inspace`), then for each
family binds its curves and their slopes as rows of a matrix and
draws two charts:

```
rect := 4 241 r_eshape (act:r_elu xs) c_at (leak act:l_eaky xs) c_at (alpha act:e_lu xs) c_at act:s_elu xs
dRect := 4 241 r_eshape (act:d_relu xs) c_at (leak act:d_leaky xs) c_at (alpha act:d_elu xs) c_at act:d_selu xs
_ := ("Rectifiers" "x" "f(x)" "ReLU" "leaky ReLU" "ELU" "SELU") u:c_hart rect
```

The slopes are Activations' derivatives (`act:d_*`, each checked
against central differences in the library's tests), but for the
gates, of which the library gives no derivative: theirs are central
differences, by one line that takes the function as its argument (a
name ending in `_`):

```
u:s_lope := { f_ x -> 5000.0 * (f_ x + 0.0001) - f_ x - 0.0001 }
```

## Your own

The page's editor holds a function of one number x named `u:m_ine`,
which may use any of Activations' functions (it starts as GELU's
sigmoid approximation, `x * act:s_igmoid 1.702 * x`). Run applies it
to the 241 x values one at a time (`e_ach`), takes its slope by
central differences, and draws both beside a built-in of your choice.
An error X_eTaL reports is placed in your text (its line counted from
your first); a value that is not finite (an overflow, say) is said,
with the first x where it happens, and nothing is drawn, as Plot draws
finite values only.

## What it shows about X_eTaL

- A chart is a value: `p:f_unctionChart!` gives the picture's SVG,
  which the command line writes to a file and the page shows.
- A function is passed to a function as an argument spelled with a
  trailing `_`, so one central-difference line serves every curve.
- A page can compile and run what its reader writes: the program the
  page ran is shown below what it drew.

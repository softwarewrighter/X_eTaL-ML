# Optim

Optimizers as steps: gradient descent and Adam, each taking a weight
array, its gradient and what the optimizer keeps, and giving them back
moved, as a tuple. A training step calls one per weight array, and
`p_ower` repeats the whole step.

```
"op:" u_se< "Optim"
```

Put each library's `src/` directory on `XETAL_PATH` (`just path`
prints them; see the [README](../../../README.md)); `op:` is the
recommended alias. The demos and live pages of this repository find
the libraries without setup.

## Conventions

- Weights are an array of any shape; its gradient and Adam's averages
  have the same shape.
- Settings come first, as a tuple: `lr` for gradient descent, `(lr,
  t)` for Adam at step t (from 1), `(lr, b1, b2, t)` for Adam with its
  own decay rates.
- A step gives back what it keeps: Adam's `(w, m, v)`.

## Functions

| Function | Type | What |
| -------- | ---- | ---- |
| `lr op:s_gd (w, g)` | `Num a => a -> (a, a) -> a` | the weights moved against their gradient by the rate |
| `(lr, t) op:a_dam (w, m, v, g)` | `(Float, Float) -> (Float, Float, Float, Float) -> (Float, Float, Float)` | Adam's step t with decay rates 0.9 and 0.999: `(w, m, v)` |
| `(lr, b1, b2, t) op:a_damWith (w, m, v, g)` | `(Float, Float, Float, Float) -> (Float, Float, Float, Float) -> (Float, Float, Float)` | the same with decay rates b1 and b2 |

## Examples

From `../tests/basics.xtl`:

```
      0.5 op:s_gd (1.0 2.0, 0.2 -0.4)
0.9 2.2
```

Adam's first step moves every weight by the rate against its
gradient's sign, whatever the gradient's size; the library's `## >>`
example shows it (run by `just doc-test`).

`../tests/checks.xtl` checks that `a_dam` is exactly the written-out
Adam of the [Training live](../../../demos/train-live/README.md) demo,
two steps running; that `a_damWith` at 0.9 and 0.999 is `a_dam`; and
that both find the bottom of a bowl.

## Demos

- [`demos/bowl.xtl`](../demos/bowl.xtl): a long narrow bowl, where
  gradient descent moves the shallow direction from 10 to 8.35 in 100
  steps and Adam to 2.24.
- [`../../../demos/cnn-backprop`](../../../demos/cnn-backprop/README.md):
  CNN training's Adam on its filters, biases and dense weights.

## Provenance

Written for X_eTaL-ML from the training demos' Adam. Adam is Kingma
and Ba's (2015), with its bias correction and eps 1e-8. Training
live and Net's `t_rain<` keep Adam written out, as their lessons.

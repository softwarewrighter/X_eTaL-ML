# Training live

X_eTaL trains a small network in your browser: 2 inputs, 16 tanh
units, a softmax over the 3 arms of a spiral. The points are made in
X_eTaL, the gradient is the backprop microscope's, and Adam is written
out in X_eTaL; the page runs 25 steps at a time and redraws what the
network decides, so you watch the regions bend to follow the spiral.

Live: [Training live](https://softwarewrighter.github.io/X_eTaL-ML/train-live/)
(press Train; change the learning rate and start again). The loss and
the share of points right are drawn after every run, in one chart, by
the [Plot](https://github.com/softwarewrighter/X_eTaL-libraries) library's `p:c_hart!`, in the page's program.

[![Training live: the live page](screenshot.png)](https://softwarewrighter.github.io/X_eTaL-ML/train-live/)

## Run it

From a clone of this repository (Rust, git and `just`):

```bash
just xetal                 # fetch and build the pinned X_eTaL (once)
just run train-live        # the loss and the points right: before, after 100 and 400 steps
just tour train-live       # each statement, then its result, paced
just serve train-live      # the web app at http://127.0.0.1:8435/
just test-demo train-live  # its CLI and browser baselines and the web app's tests
```

## The program

The spiral, 100 points on each of three arms, made in X_eTaL (a fixed
seed, so the same points every run):

```
arm := i d_iv 100
t := (0.5 + f_loat i m_od 100) / 100.0
a := (2.0944 * f_loat arm) + (5.5 * t) + 0.1 * (f_loat (r_oll! n r_eshape 1001) - 501) / 500.0
X := o_\ (2 c_at n) r_eshape ((0.1 + 0.9 * t) * c_os a) c_at (0.1 + 0.9 * t) * s_in a
```

The gradient is the [backprop microscope](../backprop/README.md)'s four
lines, giving W1's and W2's gradients as a pair. The training state
is a tuple, `(W1, W2, M1, M2, V1, V2, k)` (the weights, Adam's running
averages of gradients and of their squares, the step count), so that
`p_ower` can iterate it as one value, and the step takes it apart by
name; the step's new values get new names (each name is bound once).
Adam, one step:

```
u:m_ove := { (m, v) k -> (m / 1.0 - 0.9 ^ k) / 0.00000001 + (v / 1.0 - 0.999 ^ k) ^ 0.5 }
u:a_dam := { (W1, W2, M1, M2, V1, V2, k) ->
  t := 1.0 + k
  (G1, G2) := W1 u:g_rad W2
  m1 := (0.9 * M1) + 0.1 * G1
  m2 := (0.9 * M2) + 0.1 * G2
  v1 := (0.999 * V1) + 0.001 * G1 * G1
  v2 := (0.999 * V2) + 0.001 * G2 * G2
  (W1 - lr * (m1, v1) u:m_ove t, W2 - lr * (m2, v2) u:m_ove t, m1, m2, v1, v2, t)
}
```

Every weight moves at once, by its running average of gradients over
the square root of its running average of squared gradients (each
corrected for starting at zero). Training is then the step iterated:
`s := 25 'u:a_dam p_ower s`, no loop written.

## What it prints

The loss and the share of the 300 points right, at the start, after
100 steps and after 400: from 1.319 and 29.7% to 0.884 and 43%, then
0.062 and 100%.

## History

The state was first one flat vector with offsets (`48 t_ake`), then
seven boxed arrays taken out by position (`d_isclose 3 s_elect s`),
while X_eTaL had no way to carry several arrays through `p_ower`.
Tuples with patterns (ask M13 in
[`docs/xetal-asks.md`](../../docs/xetal-asks.md)) landed in X_eTaL
in October 2026, and the state became the tuple above; the output
did not change.

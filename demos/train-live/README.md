# Training live

X_eTaL trains a small network in your browser: 2 inputs, 16 tanh
units, a softmax over the 3 arms of a spiral. The points are made in
X_eTaL, the gradient is the backprop microscope's, and Adam is written
out in X_eTaL; the page runs 25 steps at a time and redraws what the
network decides, so you watch the regions bend to follow the spiral.

Live: [Training live](https://softwarewrighter.github.io/X_eTaL-ML/train-live/)
(press Train; change the learning rate and start again).

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
lines, giving W1's and W2's gradients boxed together. The training
state is seven boxed arrays, `W1 W2 M1 M2 V1 V2 k` (the weights,
Adam's running averages of gradients and of their squares, the step
count), so that `p_ower` can iterate it as one value. Adam, one step:

```
u:m_ove := { mv k -> ((d_isclose 1 s_elect mv) / 1.0 - 0.9 ^ k) / 0.00000001 + ((d_isclose 2 s_elect mv) / 1.0 - 0.999 ^ k) ^ 0.5 }
u:a_dam := { s ->
  W1 := d_isclose 1 s_elect s
  W2 := d_isclose 2 s_elect s
  k := 1.0 + d_isclose 7 s_elect s
  g := W1 u:g_rad W2
  G1 := d_isclose 1 s_elect g
  G2 := d_isclose 2 s_elect g
  M1 := (0.9 * d_isclose 3 s_elect s) + 0.1 * G1
  M2 := (0.9 * d_isclose 4 s_elect s) + 0.1 * G2
  V1 := (0.999 * d_isclose 5 s_elect s) + 0.001 * G1 * G1
  V2 := (0.999 * d_isclose 6 s_elect s) + 0.001 * G2 * G2
  W1 := W1 - lr * ((e_nclose M1) c_at e_nclose V1) u:m_ove k
  W2 := W2 - lr * ((e_nclose M2) c_at e_nclose V2) u:m_ove k
  (e_nclose W1) c_at (e_nclose W2) c_at (e_nclose M1) c_at (e_nclose M2) c_at (e_nclose V1) c_at (e_nclose V2) c_at e_nclose k
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

## Workarounds

`p_ower` iterates one value, so the state's seven arrays travel boxed
in one vector and are taken out by position (`d_isclose 3 s_elect s`).
Boxes let the arrays keep their own shapes (an earlier version packed
everything into one flat vector, with offsets); they cannot mix
element types, and the positions are not names. Tuples with
destructuring, then records, would let the step take and give
`(W1; W2; M1; M2; V1; V2; k)` by name: being added to X_eTaL now (ask
M13 in [`docs/xetal-asks.md`](../../docs/xetal-asks.md)).

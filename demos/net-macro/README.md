# Network macro

A network written as one line. X_eTaL's macro libraries (`.xtlm`)
extend the language itself: the [Net](../../libs/Net/docs/README.md)
library reads a spec such as `"2 16 relu 16 relu 3 softmax"` when the
program is compiled and writes the forward function in its place, as
ordinary code that is then type-checked like any other. Nothing is
hidden: `xetal expand` shows what was written.

Live: [Network macro](https://softwarewrighter.github.io/X_eTaL-ML/net-macro/)
(pick one of three networks, or type a spec of your own, and see the
function the macro wrote, its parameter count, and what the network
decides; then train it in your browser from random weights, by the
training step the macro writes from the same spec).

[![Network macro: the live page](screenshot.png)](https://softwarewrighter.github.io/X_eTaL-ML/net-macro/)

## Run it

From a clone of this repository (Rust, git and `just`):

```bash
just xetal               # fetch and build the pinned X_eTaL (once)
just run net-macro       # parameter counts, accuracies, the three decision maps
just run net-macro train-it.xtl   # the deep network trained from random weights
just expand net-macro    # the program after macro expansion: what the macros wrote
just tour net-macro      # each statement, then its result, paced
just serve net-macro     # the web app at http://127.0.0.1:8435/
just test-demo net-macro # its CLI and browser baselines and the web app's tests
```

## The program

Three networks for one task (which of three spiral arms is a point
on?), each described once, by one line of `net-macro.xtl`:

```
"u:l_inear a" net:m_odel< "2 3 softmax"
"u:s_mall b" net:m_odel< "2 4 tanh 3 softmax"
"u:d_eep c" net:m_odel< "2 16 relu 16 relu 3 softmax"
```

A spec is sizes and activations, the input's size first: each number
after the first is a dense layer, each word an activation of the
[NN library](../../libs/NN/docs/README.md). From the spec the macro
writes the loading of each dense layer's weights (`data/c1.txt`,
`data/c2.txt`, `data/c3.txt`, shaped as the spec says) and the forward
function. What the third line becomes (`just expand net-macro`):

```
c1 := { v -> 48 = t_ally v ? 3 16 r_eshape v; []P_ANIC "data/c1.txt holds " c_at (f_ormat t_ally v) c_at " numbers; layer 1 of \"2 16 relu 16 relu 3 softmax\" (2 inputs and a bias, 16 outputs) needs 48" } n_umbers []N_GET "data/c1.txt"
c2 := { v -> 272 = t_ally v ? 17 16 r_eshape v; []P_ANIC "..." } n_umbers []N_GET "data/c2.txt"
c3 := { v -> 51 = t_ally v ? 17 3 r_eshape v; []P_ANIC "..." } n_umbers []N_GET "data/c3.txt"
u:d_eep := { x -> nn:s_oftmax (nn:r_elu (nn:r_elu x nn:d_ense c1) nn:d_ense c2) nn:d_ense c3 }
```

The function one would write by hand, and before it the weights read
and checked: a file holding the wrong count stops the program with an
error naming the layer, because shapes are not part of X_eTaL's types
and `r_eshape` alone would silently repeat or cut the data (ask M12
in [`docs/xetal-asks.md`](../../docs/xetal-asks.md)). The spec is the
one place each network is described: the trainer reads these lines
too. A third macro, `"spec" net:p_arams< @`, becomes the number of
weights and biases, counted when the program is compiled (9, 27 and
371 here).

## What it prints

1. The three parameter counts.
2. The share of the 120 test points each network reads right: a line
   cannot follow a spiral (0.375), the small network nearly can
   (0.875), the deep one does (1.0).
3. What each decides over the square -1.1 to 1.1, as letters a, b, c.

On the page a spec that is not one of the three has no trained
weights, so it is expanded (with `net:n_etwork<`) and counted but not
run; a spec the macro refuses (an unknown word, no input size) shows
the macro's own message.

## Training it: `train-it.xtl`

The same spec can train the network. `train-it.xtl` makes 300 points
of the spiral in X_eTaL, starts from small random weights and repeats
the step the [Net](../../libs/Net/docs/README.md) library writes:

```
"u:s_tep X Y lr" net:t_rain< "2 16 relu 16 relu 3 softmax"
...
s0 := @ net:s_tate< "w1 w2 w3"
...
s := 400 'u:s_tep p_ower s0
```

`net:t_rain<` writes the forward pass, the backward pass (each layer's
gradient, each activation's slope) and an Adam step on a state of
boxed arrays; `net:s_tate<` the starting state. It prints the loss
and the share of points right before and after: from 1.0959 and 42%
to 0.0005 and 100% (about 30 ms a step natively).

On the page, "Train it" does this for the spec in the box, any spec
from 2 inputs to 3 softmax, 10 steps a frame up to 400 (about 40 ms a
step in the browser for the deep network, so 15 to 20 seconds in
all): the decision map redraws as it learns, the loss and the share
right are drawn after each run by the
[Plot](https://github.com/softwarewrighter/X_eTaL-libraries) library's
`p:l_ine!` in the program's own last lines, and the training step the
macro wrote is shown.

## The data

`data/` holds what the program reads with `n_umbers []N_GET`:

| File | What |
| ---- | ---- |
| `points.txt` | the 120 test points, one per line: x, y and the arm (1, 2 or 3) |
| `a1.txt`; `b1.txt`, `b2.txt`; `c1.txt`, `c2.txt`, `c3.txt` | each dense layer's weights, inputs + 1 rows by outputs columns, the bias the last row |

They are written by `train/` (`just net-train`, then `just bless
net-macro`): a small Rust program with no dependencies that reads the
three specs from `net-macro.xtl` itself, so a network is described in
one place, and trains each with Adam on 300 points of the same spiral
(a fixed seed: the same weights every run, in about 2 seconds).

## Workarounds

Plot's line chart (`p:l_ine!`) keeps a vertical range of at least 1,
so a loss that changes by less would draw nearly flat: the page's
program stretches the losses to their own range first (the line says
so), and prints the real values beside the chart. A line also needs
two points, so the curves appear from the second run on.

The code the macro writes calls NN as `nn:`, so the program imports NN
under that alias: a macro's text cannot ask which alias the caller
chose (ask M11 in [`docs/xetal-asks.md`](../../docs/xetal-asks.md)).

# World model

A tiny world model: a network that has watched a ball bounce in a
box predicts the next frame from the last three. The world is a 2 x 2
ball in a 12 x 12 box, moving 1 or 2 cells a step on each axis and
bouncing off the walls: 1,936 states in all. The network (432 pixels
in, 64 ReLU units, 144 sigmoid pixels out) was trained offline on
four fifths of them; the other fifth it never saw. Run on every state
at once, it predicts 1,897 of the 1,936 next frames exactly, 348 of
the 387 it never saw. Fed its own predictions, it dreams: it plays
the ball forward on its own, bounces included.

Live: [World model](https://softwarewrighter.github.io/X_eTaL-ML/world-model/)
(click the first frame to place the ball, pick its speed, or take a
state the network never saw; step or dream; the predicted frame as
probabilities with the true ball outlined; the dreamed ball against
the true one, with their paths).

## Run it

From a clone of this repository (Rust, git and `just`):

```bash
just xetal                  # fetch and build the pinned X_eTaL (once)
just run world-model        # every state's next frame at once, one it never saw, a dream (about 9 s)
just show world-model       # as a notebook: each statement, then its output
just serve world-model      # the web app at http://127.0.0.1:8435/
just test-demo world-model  # its CLI and browser baselines and the web app's tests
just world-model-data       # retrain (25 s, std-only Rust, a fixed seed) and rewrite data/
```

## The program

`world-model.xtl` writes the world itself: a state is a row `x y vx
vy`, and one step moves and bounces every state at once:

```
u:s_tep := { s ->
  P := 2 t_ake o_\ s
  V := -2 t_ake o_\ s
  n := P + V
  lo := n < 0
  hi := n > 10
  o_\ (n + ((n_eg 2 * n) * lo) + (20 - 2 * n) * hi) c_at V * 1 - 2 * lo + hi
}
```

A frame is 144 pixels, 1.0 where the ball is (`u:f_rames`, a `t_able`
of the pixels against the ball's corner); a state's window is its
frame and the next two side by side, 432 pixels, and the frame after
them is what the network must give. The network is two dense layers of
the [NN](../../libs/NN/docs/README.md) library, the weights read from
`data/`:

```
u:p_redict := { X -> nn:s_igmoid (nn:r_elu X nn:d_ense wb1) nn:d_ense wb2 }
```

All 1,936 states go through it as one batch (the program's slow
part: 70 million multiply-adds, about 8 s), and one count says how
many next frames are exactly right: every pixel on the right side of
one half. Then one state it never saw: its true next frame and the
network's, `#` above one half and `+` above a tenth. Then the dream:
twelve steps of `p_ower` over a tuple (the window, the places so far),
each step keeping only the likeliest 2 x 2 block of the prediction
(`u:p_lace`) and moving the window on; the dreamed ball's path beside
the true one, and how many steps they agree (all twelve, for the
state shown).

## What it shows

- **It learned the motion, mostly.** On the states it trained on it
  is never wrong; on the fifth it never saw, nine times in ten. The
  misses are near the walls, where a bounce must be read from two
  frames; there the prediction spreads (the `+` pixels), and the
  likeliest block is still usually right, which is why the dream
  stays on track longer than the exact count suggests.
- **Holding out matters.** Without weight decay the same network
  reached only 7 in 10 of the unseen states while still perfect on
  the trained ones: it had memorized windows. Weight decay 0.2 in
  the trainer's Adam brought the unseen states to 9 in 10.
- **A batch is one expression.** The whole world's prediction is one
  matrix product per layer; the per-state count is a reduce.

## The trainer

`train/` (std-only Rust, a fixed seed, `just world-model-data`)
enumerates the 1,936 states, holds out a fifth by a seeded shuffle
(`data/held-out.txt`), and trains 400 epochs of Adam with weight
decay on the pixels' cross-entropy, minibatches of 32; it writes the
weights (6 decimals) and one window's predicted pixels
(`data/expected.txt`), which the program and the page's tests check
X_eTaL reproduces to 1e-6.

## Tests

- `reg/cli-world-model`: the program's output (the Rust check, the two
  counts, the frames, the dream).
- `reg/browser-world-model`: the page in headless Chrome.
- `web/tests/micro.rs`: the first run is the world as Rust writes it
  (bounces off every wall), a prediction is the trainer's Rust model
  to 1e-6, the dream from the first unseen state tracks the ball for
  twelve steps, and the page runs the program's core with every name
  bound once.

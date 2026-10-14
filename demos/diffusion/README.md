# Learned denoiser

Pictures from noise, step by step. The pictures are 10 x 10, one of
three shapes at some place: a disk, a hollow square, a cross. A small
convolutional network was trained offline to see the clean picture in
a noisy one at ten levels of noise. Sampling runs it backward: from a picture of
pure noise, each step predicts the clean picture, infers the noise
that must have been added, and steps down one level (DDIM, with no
new noise), so the picture that comes out depends only on the noise
that went in.

Live: [Learned denoiser](https://softwarewrighter.github.io/X_eTaL-ML/diffusion/)
(eight pictures at a time, every level of the way shown as a strip;
click a row for it large; draw eight more from new noise, or the same
noise again and get the same pictures).

## Run it

From a clone of this repository (Rust, git and `just`):

```bash
just xetal                 # fetch and build the pinned X_eTaL (once)
just run diffusion         # 128 pictures from noise at once, four shown, one level by level (about 20 s)
just show diffusion        # as a notebook: each statement, then its output
just serve diffusion       # the web app at http://127.0.0.1:8435/
just test-demo diffusion   # its CLI and browser baselines and the web app's tests
just diffusion-data        # retrain (about 7 min, std-only Rust, a fixed seed, eight threads) and rewrite data/
```

## The program

`diffusion.xtl` has the schedule, the denoiser and the step. The
kept share of the picture at each level is a cosine schedule, one
vector:

```
ab := 0.001 m_ax (c_os ((p_i @) / 2.0) * (f_loat o_ffsets 11) / 10.0) ^ 2.0
ra := ab ^ 0.5
rb := (1.0 - ab) ^ 0.5
```

The denoiser is a 3 x 3 convolution of the
[Conv](../../libs/Conv/docs/README.md) library (8 filters over the
noisy picture, valid, so 8 maps of 8 x 8) with ReLU, then two dense
layers of [NN](../../libs/NN/docs/README.md) (the 512 features into
256 ReLU units, to which the level adds its own weights, a row per
level; 100 sigmoid outputs, the clean picture's probability per
pixel, mapped to -1 .. 1). The network is one function of arrays
over the joined input (a row per picture: its pixels, then the level
as a one-hot), the part a GPU takes as it is
([`docs/gpu-asks.md`](../../docs/gpu-asks.md)), and the join is
outside it:

```
u:n_et := { J ->
  X := ((t_ally J) c_at 10 10) r_eshape 100 t_ake_2 J
  V := ((t_ally J) c_at 512) r_eshape nn:r_elu (9 t_ake_2 kb, r_avel -1 t_ake_2 kb) cv:c_onv X
  A := nn:r_elu (V nn:d_ense wb1) + (-10 t_ake_2 J) '+ '* i_nner wh
  (2.0 * nn:s_igmoid A nn:d_ense wb2) - 1.0
}
u:c_lean := { t X -> u:n_et X c_at_2 10 nn:o_neHot (t_ally X) r_eshape t }
```

One step down from level t, for all the pictures X at once:

```
u:s_tep := { (X, t) ->
  C := t u:c_lean X
  (((t s_elect ra) * C) + (t s_elect rb) * (X - ((1 + t) s_elect ra) * C) / (1 + t) s_elect rb, t - 1)
}
u:s_ample := { E -> (X, _) := 10 'u:s_tep p_ower (E, 10); X }
```

C is what the network sees; `(X - ra C) / rb` at level t + 1 is the
noise that would make X of it; the next picture mixes C and that
noise at level t's roots. Ten steps by `p_ower` over the tuple (the
pictures, the level). At level 0 the share is 1.0, so what is left
is the last clean picture seen.

The command line makes 128 pictures of Gaussian noise by Box-Muller
from `r_oll!` (seeded), checks that the trainer's starting noise
becomes the picture its Rust sampler made (to 1e-6), samples all 128
at once (the slow part, about 20 s: ten steps of the convolution's
windows, 6 million numbers, their product and the dense layers over
128 rows), shows four, counts how many have a shape's worth of
pixels on (8 to 40: 85 of the 128), and shows one picture at every
level.

## What it shows

- **Where the chance is.** DDIM adds no noise after the start, so
  the sampler is a function: the same noise gives the same pictures,
  on the page's "same noise again" and in the tests.
- **What the convolution buys, measured.** The first denoiser was
  two dense layers over the pixels, with no notion of neighboring
  pixels: hollow squares came out as bars or blobs. With the
  convolution in front (the same training, the same held-out
  pictures, 2,000 of them, `train/`'s `dense` flag for the
  comparison) the held-out cross-entropy per picture goes from 13.36
  to 12.77, and at the levels that decide the shape (4 to 6, where
  about half the picture is noise) from 1.93, 4.75 and 11.03 to
  1.23, 3.36 and 8.98: a quarter to a third less. Hollow squares now
  come out as squares more often, and shapes come out thinner, so
  fewer of the 128 samples have 8 pixels on (85, against 97 before):
  the count is a floor, not a score. Pooling the maps first (the
  `pool` flag) lost, 13.64: the output needs the positions.
- **Ten steps, one expression each.** Sampling is `p_ower` over the
  step; a batch of pictures costs one matrix product per layer per
  step.

## The trainer

`train/` (std-only Rust, a fixed seed, `just diffusion-data`) makes
the pictures fresh for every batch of 64 (there is no end of them),
picks a level, noises the picture by the schedule, and trains the
denoiser on the clean picture's cross-entropy with Adam for 100,000
steps (about 7 min on the development machine; the batch's gradients summed over eight threads,
in a fixed order, so the result is the same every run). It reports
the held-out cross-entropy (2,000 pictures of their own seed, by
level), writes the weights (6 decimals; sampling uses them rounded:
`filters.txt` a row per filter with its bias last, `w1.txt`,
`wh.txt` the level's rows, `b1.txt`, `w2.txt`, `b2.txt`), prints six
samples, and writes one starting noise and what it becomes
(`data/noise.txt`, `data/expected.txt`), which the program and the
page's tests check X_eTaL reproduces. `cargo run --release -- dense`
trains the earlier dense denoiser and only reports it; `pool` the
pooled variant; a number sets the steps.

## Tests

- `reg/cli-diffusion`: the program's output (the Rust check, four
  pictures, the count, one picture level by level).
- `reg/browser-diffusion`: the page in headless Chrome.
- `web/tests/micro.rs`: the sampler is the trainer's Rust one (the
  starting noise eight times over becomes its picture eight times),
  the same noise gives the same pictures and most of them are shapes,
  and the page runs the program's core with every name bound once.

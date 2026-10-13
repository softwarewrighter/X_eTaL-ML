# Learned denoiser

Pictures from noise, step by step. The pictures are 10 x 10, one of
three shapes at some place: a disk, a hollow square, a cross. A small
network was trained offline to see the clean picture in a noisy one
at ten levels of noise. Sampling runs it backward: from a picture of
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
just run diffusion         # 128 pictures from noise at once, four shown, one level by level (about 6 s)
just show diffusion        # as a notebook: each statement, then its output
just serve diffusion       # the web app at http://127.0.0.1:8435/
just test-demo diffusion   # its CLI and browser baselines and the web app's tests
just diffusion-data        # retrain (about 2.5 min, std-only Rust, a fixed seed) and rewrite data/
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

The denoiser is two dense layers of the
[NN](../../libs/NN/docs/README.md) library (110 in: the pixels and
the level as a one-hot; 256 ReLU units; 100 sigmoid outputs, the
clean picture's probability per pixel, mapped to -1 .. 1). The
network is one function of arrays over the joined input, the part a
GPU takes as it is ([`docs/gpu-asks.md`](../../docs/gpu-asks.md)),
and the join is outside it:

```
u:n_et := { J -> (2.0 * nn:s_igmoid (nn:r_elu J nn:d_ense wb1) nn:d_ense wb2) - 1.0 }
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
at once (the slow part, about 6 s: ten steps of two dense layers
over 128 rows), shows four, counts how many have a shape's worth of
pixels on (8 to 40: 97 of the 128), and shows one picture at every
level.

## What it shows

- **Where the chance is.** DDIM adds no noise after the start, so
  the sampler is a function: the same noise gives the same pictures,
  on the page's "same noise again" and in the tests.
- **What a dense denoiser can and cannot do.** Disks and crosses come
  out clean; hollow squares often come out as bars or blobs. The
  network has no notion of neighboring pixels (every output sees
  every input alike), so where it is torn between two shapes it
  averages them. A convolutional denoiser (the
  [Conv](../../libs/Conv/docs/README.md) library) would be the next
  step.
- **Ten steps, one expression each.** Sampling is `p_ower` over the
  step; a batch of pictures costs one matrix product per layer per
  step.

## The trainer

`train/` (std-only Rust, a fixed seed, `just diffusion-data`) makes
the pictures fresh for every batch of 64 (there is no end of them),
picks a level, noises the picture by the schedule, and trains the
denoiser on the clean picture's cross-entropy with Adam for 40,000
steps (about 2.5 min). It writes the weights (6 decimals; sampling
uses them rounded), prints six samples, and writes one starting noise
and what it becomes (`data/noise.txt`, `data/expected.txt`), which the
program and the page's tests check X_eTaL reproduces.

## Tests

- `reg/cli-diffusion`: the program's output (the Rust check, four
  pictures, the count, one picture level by level).
- `reg/browser-diffusion`: the page in headless Chrome.
- `web/tests/micro.rs`: the sampler is the trainer's Rust one (the
  starting noise eight times over becomes its picture eight times),
  the same noise gives the same pictures and most of them are shapes,
  and the page runs the program's core with every name bound once.

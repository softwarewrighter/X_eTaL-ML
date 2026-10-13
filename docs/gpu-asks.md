# Asks for running the ML work on GPUs

X_eTaL-ML's models are to be tested, measured and shown on real GPUs
through [X_eTaL-gpu](https://github.com/softwarewrighter/X_eTaL-gpu)
(OpenCL: Apple silicon now, the Arch machine's NVIDIA cards later).
This repo writes no GPU code: it provides each model as an X_eTaL
program, its weights in `data/` and its expected outputs; X_eTaL-gpu
runs them on its devices. This file lists what that needs from the
other X_eTaL repositories. The plan, the workloads in order, is in
[`plan.md`](plan.md) ("Combined with X_eTaL-gpu").

Each entry: status, which repo it is for, which models need it, why,
and what is done meanwhile. Status: **not yet taken** (filed here, to
be taken to that repo), **filed there** (that repo already has it,
under its number; ours adds our users), **in progress there**,
**landed** (the date, and where; the workaround gone). X_eTaL-gpu
took GA1 to GA6 into its ledger on 2026-10-12 (its `docs/plan.md`,
"asks"); its results are summarized in [`speed.md`](speed.md), "On
GPUs".

| # | Status | For | Ask | Models | Meanwhile |
| - | ------ | --- | --- | ------ | --------- |
| GA1 | in progress there (saga gpu-models: ml-bench and world-model done 2026-10-12 on the M1 Max and the RTX 5060 Ti, `XETAL_ML_COMMIT` fac586c; the denoiser queued, its step denoiser) | X_eTaL-gpu | Pin X_eTaL-ML (an `XETAL_ML_COMMIT`, as this repo pins X_eTaL) and run its batch models from it: their programs, `data/` weights and expected outputs, the same decisions on every device, timed | world-model, the benchmarks, the denoiser (plan rows 1 to 3) | the denoiser's batch run waits on its re-pin past 1eded2a (`demos/diffusion/`, 2026-10-12) |
| GA2 | landed 2026-10-12 (X_eTaL-gpu's step transpose: `transpose` in the IR, a gather or tiles; Accel `s_cores` and `w_eightGrad` checked on both cards) | X_eTaL-gpu | `transpose` in its IR | attention scores (`Q K^T`: micro-gpt, the Attention library), every backward pass (cnn-backprop, train-live) | none needed |
| GA3 | landed 2026-10-12 (its step windows: rotate, drop along an axis and reshape in the IR; Conv's `w_indows` expression runs as six gathers, no im2col) | X_eTaL-gpu | The convolution's windows in its IR: `reshape` and rotation along an axis, or one windows operation (im2col) | cnn-digits, cnn-backprop, the Conv library | none needed |
| GA4 | filed there (X-1) | X_eTaL-extensions | A newer X_eTaL inside `xetal-x` than v0.1.0: our libraries use `h:` helpers (NN) and tuples (Optim, the training state), which v0.1.0 lacks, so they cannot run under the bridge | every model through the Gpu facade | the GPU run is a separate program on the IR twin; the evaluator's run is the reference |
| GA5 | filed there (G8, G5); the input half landed in X_eTaL-gpu 2026-10-12 (binary file binding: its CLI `pack` and `--bind w=@file`, the facade's `gp:b_indFile`; the world model's batch bound in 13 ms against 460 ms as text on the RTX) | X_eTaL-extensions, X_eTaL | Arrays across the bridge in binary, not as text: X_eTaL-gpu measured 401 ms for a million Floats out and back | world-model's batch (1,936 x 432 = 836,352 inputs, about a third of a second as text), the denoiser (20 calls) | inputs as binary files; outputs still cross as text, so batches as large as possible, the bridge's time reported apart from the device's |
| GA6 | filed there (G1, G2 filed with X_eTaL as GPU1, GPU2, new; G3 declined by X_eTaL: shapes are checked at run time, so the lowering will specialize per shape when a program is bound) | X_eTaL | The accelerator IR, the lowering from X_eTaL's Core to it, and shapes known statically | every model | an IR twin written for each model (X_eTaL-gpu's `scripts/ml-bench.py` writes ours from the models' definitions), checked against the evaluator's output |

## Details

### GA1: X_eTaL-ML's models on X_eTaL-gpu

X_eTaL-gpu's next step (saga 4, tiny-nn) asks for "a complete tiny
trained network, from X_eTaL-ML's trainers". The world model is one:
432 pixels in (three 12 x 12 frames), 64 ReLU units, 144 sigmoid
pixels out, every operation already in its IR (products, `max` for
the ReLU, `exp` and `div` for the sigmoid, `table right` for the
bias). Its program runs all 1,936 states of its world as one batch and
prints how many next frames it predicts exactly: one Int, the same on
the evaluator and on every device, whatever the float width.

What this repo provides, per model:

- the X_eTaL program (`demos/<slug>/<slug>.xtl`), whose batch run is
  the reference;
- the weights and inputs in `demos/<slug>/data/`, written by its
  trainer (std-only Rust, fixed seed);
- the expected outputs as reg-rs baselines (`demos/<slug>/reg/`);
- a stated tolerance: the same decision always; numbers within 1e-5
  in f32 (Apple's OpenCL has no 64-bit floats), within 1e-9 in f64
  (NVIDIA cards with fp64) against the trainer's Rust model.

What is asked of X_eTaL-gpu: pin an X_eTaL-ML commit, write each
model's IR twin, run it on each device, and publish the table (model,
device, the decision, the largest difference, the device's time and
the bridge's). This repo's `docs/speed.md` links that table.

What came back (2026-10-12, X_eTaL-gpu's steps ml-bench and
world-model, `XETAL_ML_COMMIT` fac586c): `scripts/ml-bench.py` there
writes the twins from the models' definitions (no twin by hand), has
the windows generated by the program's own core under the pinned
xetal, and runs the evaluator, its interpreter, every device and the
`xetal-x` facade with bind, run and output timed apart. The world
model's counts, 1,897 and 348, are the same everywhere; the
benchmarks agree within a relative 1e-5 in f32 on the M1 Max and the
RTX 5060 Ti. The tables: its `docs/ml-results.md` and
`docs/rtx5060-ml-results.md`, summarized in `speed.md`. The denoiser
(row 3, `demos/diffusion/`, committed 2026-10-12 at 1eded2a) is
queued there as step denoiser, after a re-pin.

### GA2, GA3: operations the larger models need (landed)

The IR had inputs, constants, elementwise maps, reductions along an
axis, matrix products, `take`, `drop`, `ravel`, `table` and `select`,
which covers dense networks (rows 1 to 3 of the plan). The next
models needed two more, both landed in X_eTaL-gpu on 2026-10-12:

- `transpose` (its step transpose: a gather, or the last two axes in
  tiles), for `Q K^T` in attention (micro-gpt, the Attention
  library: `at:s_cores`) and for the backward pass of every layer
  (`o_\ X '+ '* i_nner dY`); Accel's `s_cores` and `w_eightGrad` are
  its two uses there, checked on both cards;
- the convolution's windows (its step windows: rotation along an
  axis, drop along an axis and reshape in the IR). Conv builds them
  by rotation and reshape (`cv:w_indows`); that expression runs there
  as six gathers. It measured im2col as one operation at 0.04 ms
  saved per 64-picture batch on the M1 Max, so there is none.

So the later workloads of the plan (ternary-net, moe-router,
cnn-digits on the MNIST test set, micro-gpt and the Attention
library, cnn-backprop's training) no longer wait on the IR; X_eTaL-gpu
has since added the embedding lookup (`s_elect` of rows by token ids)
and a top-k mask for them.

### GA4, GA5, GA6: already filed

These are X_eTaL-gpu's own asks (its `docs/xetal-asks.md`: G1 to G3,
G5, G8; its `docs/tuple-plan.md`: X-1); they are listed here with this
repo's users and sizes, so that list stands on its own. When one
lands, its row here is marked landed and the workaround removed.
Their state as of 2026-10-13: X-1 open (`xetal-x` still embeds
v0.1.0); G5 and G8 planned in X_eTaL (its Saga 23, host bindings and
native packages), and meanwhile X_eTaL-gpu binds inputs from binary
files (GA5's input half; outputs still cross as text); G1 and G2
filed with X_eTaL (GPU1, GPU2: new, not yet planned); G3 declined by
X_eTaL (array shapes are checked at run time; no annotations in v0),
so its lowering will specialize per shape when a program is bound,
as its twins do today.

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
under its number; ours adds our users), **landed**.

| # | Status | For | Ask | Models | Meanwhile |
| - | ------ | --- | --- | ------ | --------- |
| GA1 | not yet taken | X_eTaL-gpu | Pin X_eTaL-ML (an `XETAL_ML_COMMIT`, as this repo pins X_eTaL) and run its batch models from it: their programs, `data/` weights and expected outputs, the same decisions on every device, timed | world-model, the benchmarks, the denoiser (plan rows 1 to 3) | X_eTaL-gpu's own tiny network (`libs/Accel/demos/tiny-net`) |
| GA2 | not yet taken | X_eTaL-gpu | `transpose` in its IR | attention scores (`Q K^T`: micro-gpt, the Attention library), every backward pass (cnn-backprop, train-live) | the weights transposed on the host, which does not help a product of two activations |
| GA3 | not yet taken | X_eTaL-gpu | The convolution's windows in its IR: `reshape` and rotation along an axis, or one windows operation (im2col) | cnn-digits, cnn-backprop, the Conv library | the windows built on the host and passed in (nine times the picture's size) |
| GA4 | filed there (X-1) | X_eTaL-extensions | A newer X_eTaL inside `xetal-x` than v0.1.0: our libraries use `h:` helpers (NN) and tuples (Optim, the training state), which v0.1.0 lacks, so they cannot run under the bridge | every model through the Gpu facade | the GPU run is a separate program on the IR twin; the evaluator's run is the reference |
| GA5 | filed there (G8, G5) | X_eTaL-extensions, X_eTaL | Arrays across the bridge in binary, not as text: X_eTaL-gpu measured 401 ms for a million Floats out and back | world-model's batch (1,936 x 432 = 836,352 inputs, about a third of a second as text), the denoiser (20 calls) | batches as large as possible, the bridge's time reported apart from the device's |
| GA6 | filed there (G1, G2, G3) | X_eTaL | The accelerator IR, the lowering from X_eTaL's Core to it, and shapes known statically | every model | an IR twin written by hand for each model, checked against the evaluator's output |

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

### GA2, GA3: operations the larger models need

The IR has inputs, constants, elementwise maps, reductions along an
axis, matrix products, `take`, `drop`, `ravel`, `table` and `select`.
That covers dense networks (rows 1 to 3 of the plan). The next models
need two more:

- `transpose`, for `Q K^T` in attention (micro-gpt, the Attention
  library: `at:s_cores`) and for the backward pass of every layer
  (`o_\ X '+ '* i_nner dY`);
- the convolution's windows. Conv builds them by rotation and reshape
  (`cv:w_indows`); an IR without both needs the windows built on the
  host, nine times the picture's size, which the bridge's cost (GA5)
  makes slow.

### GA4, GA5, GA6: already filed

These are X_eTaL-gpu's own asks (its `docs/xetal-asks.md`: G1 to G3,
G5, G8; its `docs/tuple-plan.md`: X-1); they are listed here with this
repo's users and sizes, so that list stands on its own. When one
lands, its row here is marked landed and the workaround removed.

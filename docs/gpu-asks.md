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
| GA1 | in progress there (saga gpu-models: ml-bench and world-model done 2026-10-12 on the M1 Max and the RTX 5060 Ti; since 2026-10-16 (its main 0fbf5f2) `XETAL_ML_COMMIT` c032768 and every program below runs from its own source: the benchmarks, world-model (32 lines identical), cnn-digits' batch (971), diffusion and micro-gpt's batch, exact on the interpreter and within 1e-5 on the M1 Max) | X_eTaL-gpu | Pin X_eTaL-ML (an `XETAL_ML_COMMIT`, as this repo pins X_eTaL) and run its batch models from it: their programs, `data/` weights and expected outputs, the same decisions on every device, timed | world-model, the benchmarks, the denoiser (plan rows 1 to 3); since 2026-10-13 cnn-digits' and micro-gpt's batch programs (rows 4 and 5: `demos/cnn-digits/batch.xtl`, `demos/micro-gpt/batch.xtl`) | none for the runs; the denoiser's 1e-6 check prints 0 in 32-bit floats (step f32-tolerance here) |
| GA2 | landed 2026-10-12 (X_eTaL-gpu's step transpose: `transpose` in the IR, a gather or tiles; Accel `s_cores` and `w_eightGrad` checked on both cards) | X_eTaL-gpu | `transpose` in its IR | attention scores (`Q K^T`: micro-gpt, the Attention library), every backward pass (cnn-backprop, train-live) | none needed |
| GA3 | landed 2026-10-12 (its step windows: rotate, drop along an axis and reshape in the IR; Conv's `w_indows` expression runs as six gathers, no im2col) | X_eTaL-gpu | The convolution's windows in its IR: `reshape` and rotation along an axis, or one windows operation (im2col) | cnn-digits, cnn-backprop, the Conv library | none needed |
| GA4 | filed there (X-1) | X_eTaL-extensions | A newer X_eTaL inside `xetal-x` than v0.1.0: our libraries use `h:` helpers (NN) and tuples (Optim, the training state), which v0.1.0 lacks, so they cannot run under the bridge | every model through the Gpu facade | the GPU run is a separate program on the IR twin; the evaluator's run is the reference |
| GA5 | filed there (G8, G5); the input half landed in X_eTaL-gpu 2026-10-12 (binary file binding: its CLI `pack` and `--bind w=@file`, the facade's `gp:b_indFile`; the world model's batch bound in 13 ms against 460 ms as text on the RTX) | X_eTaL-extensions, X_eTaL | Arrays across the bridge in binary, not as text: X_eTaL-gpu measured 401 ms for a million Floats out and back | world-model's batch (1,936 x 432 = 836,352 inputs, about a third of a second as text), the denoiser (20 calls) | inputs as binary files; outputs still cross as text, so batches as large as possible, the bridge's time reported apart from the device's |
| GA6 | in progress there, provisionally (X_eTaL-gpu's A4b, 2026-10-13: G1 and G2 stayed unplanned in X_eTaL and G3 was declined, so it builds the lowering itself, from X_eTaL source through the pinned X_eTaL's own crates; its steps core-to-xir, run-from-source, accel-macro, models-from-source) | X_eTaL-gpu now; X_eTaL in the end (its G11: a crate API for source to Core) | The accelerator IR, the lowering from X_eTaL's Core to it, and shapes known statically | every model | until models-from-source lands: the twins `scripts/ml-bench.py` writes from the models' definitions, checked against the evaluator's output |
| GA7 | landed there (2026-10-16, its main 0fbf5f2: `c_at` of arrays, `'= t_able` and `^` in the subset; the denoiser and the two batch programs run from their own source unchanged) | X_eTaL-gpu | In the lowering's subset: `c_at` and `c_at_2` of arrays (it has `c_at` on shapes), the outer product of any comparison (`'= t_able`, NN's `o_neHot`), and `^` with a scalar | the denoiser's join (the pixels beside the level's one-hot: `X c_at_2 10 nn:o_neHot (t_ally X) r_eshape t`); every model that joins its inputs | done here (2026-10-13, step denoiser-split): the denoiser's network is `u:n_et`, one function of arrays over the joined input, in the subset as it is; the join is in `u:c_lean` around it, the evaluator's; the schedule's roots are the vectors `ra` and `rb`, taken once, so `u:s_tep` only selects from them. Output byte for byte unchanged. The ask stands for the join itself to move over |

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

Its step gpu-pipeline (queued 2026-10-13: pinned f32 input buffers,
non-blocking transfers, the batch axis chunked and run on streams
with events, `explain` of the stages, `--repeat` for sustained items
per second; the world model measured first, against OpenBLAS and
Accelerate) needs nothing new from here: the batch is an input and
how it is cut is the schedule's. The world model offers a stream of
independent batches of any size (its core generates the windows;
the weights resident); the denoiser offers ten dependent stages per
batch, so a stream of batches pipelines stage by stage (batch k at
step t while batch k+1 is at step t-1). Both keep their decision
whatever the chunking: the counts, and the same pictures from the
same noise.

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

### GA6 since 2026-10-13: the lowering is built there; GA7

X_eTaL-gpu's course correction (its decision A4b, the user's, after
a blog agent read every GPU program as hand-written, which it was):
since G1 and G2 are unplanned in X_eTaL and G3 declined, it lowers
X_eTaL source to its IR itself, provisionally, through the pinned
X_eTaL's own crates (a path dependency on its `work/xetal`, so the
library bodies and the real Core are X_eTaL's), the shapes taken from
the bound inputs. Four steps: core-to-xir (the twelve hand twins as
the oracle, deleted when passed), run-from-source (`xetal-gpu run
FILE.xtl --device opencl:N`: the statements before the accelerated
function evaluated by the pinned xetal, so our programs' data
preparation and file reads run as they are, with no v0.1.0 limit),
accel-macro (`gp:a_ccel<`: a macro hands a function's text to the
gpu extension, which lowers, binds and runs it with the weights
resident; this route runs under `xetal-x`, so GA4 applies to it) and
models-from-source: this repository's world model and the three
benchmarks from their own `.xtl`, the generator in `ml-bench.py`
deleted. In the end X_eTaL owns the lowering (its ask G11, a crate
API for source to Core; G12, the macro contract stated).

What this asks of this repository is that each model's accelerated
part be one function of arrays in the subset it lowers (`let` and
`eval` statements, lambdas applied to arrays, arithmetic and
comparisons with scalar extension, `f_loat`, `f_loor`, `e_xp`,
`l_og`, `m_ax`, `m_in`, `r_/` with and without an axis, `i_nner`,
`t_able` with `l_eft` and `r_ight`, `t_ake`, `d_rop`, `r_avel`,
`r_eshape`, `o_-`, `o_\`, `t_ranspose`, `s_elect`, `o_ffsets`,
`r_ange`, `t_ally`, `s_hape`, `f_irst`, `c_at` on shapes; library
functions inlined at their calls), and the rest (loops by `p_ower`,
tuples, choices by `i_ndexOf`, printing) stay in the evaluator. As
the programs stand:

- world-model: `u:p_redict` (NN's `d_ense`, `r_elu`, `s_igmoid`:
  `i_nner`, `d_rop`, `t_ake`, `f_irst`, `'r_ight t_able`, `m_ax`,
  `e_xp`) is in the subset, and so is `u:e_xact`, the count
  (`f_loor`, `r_/`, `'m_in r_/_2`, comparisons); `u:w_indow` and the
  dream (`p_ower`, `i_ndexOf`) stay in the evaluator, as the plan's
  row 1 intended;
- the benchmarks: in the subset as they are (`e_ach` over a lambda
  of arrays repeats the work; NN's `s_oftmax`);
- cnn-digits' batch (`batch.xtl`, 2026-10-13): `u:r_ead` is one
  function of arrays (Conv's `w_indows` by rotations, drops and a
  reshape, `c_onv` as one product, `p_ool` by a reshape and a
  reduction, NN's `d_ense` and `a_rgmax`), the rows 4 case of the
  plan; the count against the labels stays in the evaluator;
- micro-gpt's batch (`batch.xtl`, 2026-10-13): `u:g_pt` as the demo
  has it (the embedding lookup by `s_elect`, Norm's `r_ms`,
  Attention's `h_ead` with `o_\`, `c_at_2` of the four heads), called
  once per name by `p_ower` over the totals; the heads' `c_at_2` is
  GA7's again, and the causal mask is an argument the evaluator
  makes;
- the denoiser: `u:c_lean` joins the pixels and the level's one-hot
  with `c_at_2` and NN's `o_neHot` (`'= t_able r_ange`), and
  `u:s_tep` takes the schedule's roots with `^ 0.5`; none of the
  three is in the subset as listed, hence GA7. Its loop is `p_ower`
  over a tuple, so the ten steps are ten calls, each a bind of the
  pictures, which is what the plan's row 3 wants to measure.

GA7 asks for `c_at` and `c_at_2` of arrays, the outer product of any
comparison and `^` with a scalar in the subset. Meanwhile (done
here on 2026-10-13, step denoiser-split, the user's go) the demo's
`u:c_lean` is split: `u:n_et` is the two dense layers over the joined
input J, one function of arrays in the subset as it is, and
`u:c_lean` is the join around it (`u:n_et X c_at_2 10 nn:o_neHot
(t_ally X) r_eshape t`), the evaluator's; the schedule's roots are
taken once as the vectors `ra := ab ^ 0.5` and `rb := (1.0 - ab) ^
0.5`, so `u:s_tep` only selects from them. The program's output,
the page's tests and the baselines are unchanged byte for byte. So
the denoiser's accelerated function is `u:n_et`, called ten times
per batch with the joined input bound each time (110 numbers per
picture); the ask stands so the join can move over too. Since step
conv-denoiser (2026-10-14) `u:n_et` is a 3 x 3 convolution (Conv's
`w_indows` and `c_onv`: rotations, drops, a reshape and one product,
GA3's operations) with ReLU, then the dense layers, the level's
one-hot through its own weights (`'+ '* i_nner`, no join inside);
still one function of arrays, and row 3 is now a convolutional
workload: 6 million window numbers and a product per batch of 128,
ten times.

### Found there: no huge constants (32-bit floats)

X_eTaL-gpu, lowering Accel's top-k (a copy of Sample's `t_op`),
2026-10-13: subtracting 1e300 to take a row's largest out is
infinity in a GPU's f32, and inf times 0 is NaN, so the row comes
back wrong on the device while the CPU's f64 hides it; its runtime
now refuses any Float beyond float's range at upload. Done here
(step finite-shift): Sample's `t_op` shifts a taken item by 1 + the
range of the values and Learn's `k_nn` by 1 + the largest distance,
both finite, every output unchanged; Attention's `w_eights`
subtracts 1e9 on the mask, within f32 (exp of it is 0, no NaN), and
stays. The rule for new code: a mask's shift is a finite function
of the data, never a constant beyond 3e38.

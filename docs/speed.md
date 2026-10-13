# Speed

How fast the pinned X_eTaL runs this repository's ML workloads, to
see when an upstream change makes them faster or slower (ask M3 in
[`xetal-asks.md`](xetal-asks.md): the matrix product, and X_eTaL's
Saga 30 on the `t_able` / `i_nner` regression).

```bash
just bench           # best of 3 wall times, as the table below
just bench-check     # against this machine's baseline: SLOWER over 15%, faster listed
just bench-bless     # record this machine's baseline (a slowdown needs the user's approval)
```

The timings depend on the machine and on what else it is doing, so
they are not part of the gate; compare runs made on one quiet machine.
The baseline is per host, `bench/baseline/HOST.tsv`. A single SLOWER
on a busy machine is a reason to run the check again, not yet a
finding.

## The benchmarks

| Program | What it times |
| ------- | ------------- |
| `bench/dense.xtl` | a dense layer's matrix product, 1024 x 16 by 16 x 16 with `i_nner`, four times (ask M3's repro) |
| `bench/inner-wide.xtl` | the CNN's dense layer shape, a 1 x 1352 row by 1352 x 10, 40 times |
| `bench/conv.xtl` | the CNN's convolution: nine shifted copies of a 28 x 28 picture times 8 filters, ten times |
| `bench/softmax.xtl` | `nn:s_oftmax` by row on 1000 x 64, ten times |
| `bench/table-spread.xtl` | spreading a vector across a 512 x 512 matrix with `t_able`, four times |
| `demos/*/SLUG.xtl` | each demo's program, run from its directory (its data from `data/`) |

Each prints one small value, so printing does not count; each repeats
its work with `e_ach`, not repeated lines.

## The baseline: X_eTaL v0.1.0

Measured 2026-10-05 on the development machine (an Apple M1 Max,
release build, best of 5; other builds were running at the time),
beside the first baseline, taken at abb8274 the day before:

| Program | abb8274 (ms) | v0.1.0 (ms) | Per unit of work now |
| ------- | ------------ | ----------- | -------------------- |
| `bench/dense.xtl` | 804 | 145 | about 130 ns per multiply-add (1,048,576) |
| `bench/inner-wide.xtl` | 400 | 87 | about 150 ns per multiply-add (540,800) |
| `bench/conv.xtl` | 577 | 100 | about 190 ns per multiply-add (486,720), shifts included |
| `bench/softmax.xtl` | 353 | 229 | about 0.35 us per row element, five array operations each |
| `bench/table-spread.xtl` | 655 | 183 | about 170 ns per cell of the spread |
| `demos/cnn-digits/cnn-digits.xtl` | 1539 | 315 | |
| `demos/ternary-net/ternary-net.xtl` | 685 | 126 | |
| `demos/moe-router/moe-router.xtl` | 139 | 56 | |
| `demos/attention/attention.xtl` | 31 | 34 | |

(Each time includes about 10 ms to start the program.)

What it shows: at abb8274 a matrix product cost several hundred
nanoseconds per multiply-add and a `t_able` spread several hundred per
cell, a regression X_eTaL's Saga 30 took on (ask M3). X_eTaL v0.1.0
fixed it: the matrix product and `t_able` are 4 to 6 times faster, and
so are the demos built on them. The baseline in `bench/baseline/` is
now the v0.1.0 one, so `just bench-check` guards these times.

## On GPUs: X_eTaL-gpu's results

[X_eTaL-gpu](https://github.com/softwarewrighter/X_eTaL-gpu) pins
this repository (`XETAL_ML_COMMIT`, fac586c as of 2026-10-12) and
runs its batch programs (plan.md, "Combined with X_eTaL-gpu") as
programs of its accelerator IR on its devices, beside the pinned
X_eTaL as the reference (ask GA1 in
[`gpu-asks.md`](gpu-asks.md)). Its tables, one per card, each column
compared with the evaluator on the same host:

- [`docs/ml-results.md`](https://github.com/softwarewrighter/X_eTaL-gpu/blob/main/docs/ml-results.md):
  the Apple M1 Max;
- [`docs/rtx5060-ml-results.md`](https://github.com/softwarewrighter/X_eTaL-gpu/blob/main/docs/rtx5060-ml-results.md):
  the NVIDIA GeForce RTX 5060 Ti in the Arch Linux machine;
- its README, "Results by GPU": the world model through a kept
  session against NumPy's BLAS on the same host, and binary binding.

What came back, as of X_eTaL-gpu 7c85c7c (2026-10-13):

| Workload | The decision | Numbers | Apple M1 Max | RTX 5060 Ti |
| -------- | ------------ | ------- | ------------ | ----------- |
| 1: world-model, all 1,936 states and the 387 held out | 1,897 and 348 exact on the evaluator, its interpreter, both cards in f32, and through the `xetal-x` facade | the counts are Ints: the same everywhere | kernels 0.10 ms, launches 8 ms, the process 99 ms; binding the batch as text 220 ms | kernels 0.24 ms, launches 0.5 ms, the process 317 ms; binding as text 458 ms, as a binary file 13 ms |
| 2: `bench/dense.xtl` | 40394.999999997555 | largest difference 2.4e-9 | kernels 0.007 ms (the evaluator 114 ms) | kernels 0.07 ms (the evaluator 212 ms) |
| 2: `bench/inner-wide.xtl` | 1014.2999999999904 | 0.0025 (relative 2.5e-6) | kernels 0.63 ms (57 ms) | kernels 1.15 ms (117 ms) |
| 2: `bench/softmax.xtl` | 10000.000000000902 | 0.001 (relative 1e-7) | kernels 0.05 ms (171 ms) | kernels 0.50 ms (371 ms) |
| 3: the denoiser, 128 pictures in ten steps | queued there (its step denoiser), once it re-pins this repository past 1eded2a | | | |

The tolerance held: the same decision on every device, and the
benchmarks' numbers within a relative 1e-5 in f32 (both cards ran
the f32 schedule; the 1e-9 f64 tolerance has not been exercised
yet). The kernels take a fraction of a millisecond everywhere; what
costs is around them: compiling the kernels on every run (about 70
ms on the M1 Max, 160 ms on the RTX), and the bridge, which carries
arrays as text (the world model's 836,352 inputs: a quarter to half
a second; ask GA5, whose input half landed there as binary files).
Against tuned CPU code the picture is honest: the world model
through a kept session, its weights resident, runs the 1,936-state
batch in 2.5 ms on the M1 Max against NumPy's 1.3 ms, and in 3.0 ms
on the RTX against 1.9 ms; neither card beats BLAS on a model this
small, and the gap closes as the batch grows (8,192 states: 0.85 and
0.82 of NumPy's time). A model with more work per uploaded byte is
where a GPU starts to pay, which the later workloads are for.

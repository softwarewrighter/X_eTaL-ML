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

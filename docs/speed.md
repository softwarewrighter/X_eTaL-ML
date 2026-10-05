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

## The baseline: X_eTaL abb8274

Measured 2026-10-04 on the development machine (an Apple M1 Max,
release build, best of 5; other builds were running at the time):

| Program | Best of 5 (ms) | Per unit of work |
| ------- | -------------- | ---------------- |
| `bench/dense.xtl` | 804 | about 770 ns per multiply-add (1,048,576) |
| `bench/inner-wide.xtl` | 400 | about 740 ns per multiply-add (540,800) |
| `bench/conv.xtl` | 577 | about 1.2 us per multiply-add (486,720), shifts included |
| `bench/softmax.xtl` | 353 | about 0.55 us per row element, five array operations each |
| `bench/table-spread.xtl` | 655 | about 620 ns per cell of the spread |
| `demos/ternary-net/ternary-net.xtl` | 685 | |
| `demos/moe-router/moe-router.xtl` | 139 | |
| `demos/cnn-digits/cnn-digits.xtl` | 1539 | |
| `demos/attention/attention.xtl` | 31 | |

What it shows: a matrix product costs several hundred nanoseconds per
multiply-add, and spreading with `t_able` several hundred per cell,
while whole-array arithmetic costs tens of nanoseconds per element
(X_eTaL-demos measured about 8x faster elementwise at abb8274 than at
06d39fa, and `t_able` 2.7x and `i_nner` 1.5x slower). Those two
operations are what the ML demos spend their time in: the CNN page
takes about 100-150 ms per digit in the browser, the 1.58-bit page's
maps about a second. When a pinned X_eTaL fixes them, `just
bench-check` lists these programs as faster.

# Conv

Convolution for a batch of pictures: 3 x 3 windows, filters as one matrix product, max-pooling, and their backward passes

```
"cv:" u_se< "Conv"
```

| Directory | What |
| --------- | ---- |
| [`src/`](src/) | the library, `Conv.xtl` |
| [`docs/`](docs/README.md) | the reference: every function, its type, examples, provenance |
| [`demos/`](demos/) | programs that use it |
| [`tests/`](tests/) | reg-rs baselines: the test programs, the pinned export types (`types.rgt`) and the demos (`demo-*.rgt`) |

```bash
just demo Conv         # run its demos
just run Conv          # run its test programs
just test-lib Conv     # check every baseline
```

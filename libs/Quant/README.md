# Quant

Weights in fewer bits: float rounding, integer and ternary quantization, a ternary layer as additions

```
"qz:" u_se< "Quant"
```

| Directory | What |
| --------- | ---- |
| [`src/`](src/) | the library, `Quant.xtl` |
| [`docs/`](docs/README.md) | the reference: every function, its type, examples, provenance |
| [`demos/`](demos/) | programs that use it |
| [`tests/`](tests/) | reg-rs baselines: the test programs, the pinned export types (`types.rgt`) and the demos (`demo-*.rgt`) |

```bash
just demo Quant         # run its demos
just run Quant          # run its test programs
just test-lib Quant     # check every baseline
```

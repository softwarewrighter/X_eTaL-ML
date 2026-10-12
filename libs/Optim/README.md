# Optim

Optimizers as steps: gradient descent and Adam on a weight array and its averages

```
"op:" u_se< "Optim"
```

| Directory | What |
| --------- | ---- |
| [`src/`](src/) | the library, `Optim.xtl` |
| [`docs/`](docs/README.md) | the reference: every function, its type, examples, provenance |
| [`demos/`](demos/) | programs that use it |
| [`tests/`](tests/) | reg-rs baselines: the test programs, the pinned export types (`types.rgt`) and the demos (`demo-*.rgt`) |

```bash
just demo Optim         # run its demos
just run Optim          # run its test programs
just test-lib Optim     # check every baseline
```

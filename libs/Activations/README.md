# Activations

activation functions and their derivatives, the gated layers

```
"act:" u_se< "Activations"
```

| Directory | What |
| --------- | ---- |
| [`src/`](src/) | the library, `Activations.xtl` |
| [`docs/`](docs/README.md) | the reference: every function, its type, examples, provenance |
| [`demos/`](demos/) | programs that use it |
| [`tests/`](tests/) | reg-rs baselines: the test programs, the pinned export types (`types.rgt`) and the demos (`demo-*.rgt`) |

```bash
just demo Activations         # run its demos
just run Activations          # run its test programs
just test-lib Activations     # check every baseline
```

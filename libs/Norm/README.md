# Norm

Normalizing each row: layer norm and RMS norm, with a gain and a bias

```
"nm:" u_se< "Norm"
```

| Directory | What |
| --------- | ---- |
| [`src/`](src/) | the library, `Norm.xtl` |
| [`docs/`](docs/README.md) | the reference: every function, its type, examples, provenance |
| [`demos/`](demos/) | programs that use it |
| [`tests/`](tests/) | reg-rs baselines: the test programs, the pinned export types (`types.rgt`) and the demos (`demo-*.rgt`) |

```bash
just demo Norm         # run its demos
just run Norm          # run its test programs
just test-lib Norm     # check every baseline
```

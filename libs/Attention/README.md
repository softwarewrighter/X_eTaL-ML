# Attention

Attention: scaled dot-product scores, masked softmax weights, causal masks, a head

```
"at:" u_se< "Attention"
```

| Directory | What |
| --------- | ---- |
| [`src/`](src/) | the library, `Attention.xtl` |
| [`docs/`](docs/README.md) | the reference: every function, its type, examples, provenance |
| [`demos/`](demos/) | programs that use it |
| [`tests/`](tests/) | reg-rs baselines: the test programs, the pinned export types (`types.rgt`) and the demos (`demo-*.rgt`) |

```bash
just demo Attention         # run its demos
just run Attention          # run its test programs
just test-lib Attention     # check every baseline
```

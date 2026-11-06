# MoE

Mixture-of-experts layers: router scores, top-k gates, experts as adapted projections, the sparse combine, gradients and routing statistics

```
"moe:" u_se< "MoE"
```

| Directory | What |
| --------- | ---- |
| [`src/`](src/) | the library, `MoE.xtl` |
| [`docs/`](docs/README.md) | the reference: every function, its type, examples, provenance |
| [`demos/`](demos/) | programs that use it |
| [`tests/`](tests/) | reg-rs baselines: the test programs, the pinned export types (`types.rgt`) and the demos (`demo-*.rgt`) |

```bash
just demo MoE         # run its demos
just run MoE          # run its test programs
just test-lib MoE     # check every baseline
```

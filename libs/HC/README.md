# HC

Hyper-Connections and mHC: streams mixed by tables, the Sinkhorn projection, the layer and its gradients, the gains by depth, the input-dependent tables

```
"hc:" u_se< "HC"
```

| Directory | What |
| --------- | ---- |
| [`src/`](src/) | the library, `HC.xtl` |
| [`docs/`](docs/README.md) | the reference: every function, its type, examples, provenance |
| [`demos/`](demos/) | programs that use it |
| [`tests/`](tests/) | reg-rs baselines: the test programs, the pinned export types (`types.rgt`) and the demos (`demo-*.rgt`) |

```bash
just demo HC         # run its demos
just run HC          # run its test programs
just test-lib HC     # check every baseline
```

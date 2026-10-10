# Learn

Classic machine learning: k-means, k nearest neighbors, PCA, logistic regression

```
"ml:" u_se< "Learn"
```

| Directory | What |
| --------- | ---- |
| [`src/`](src/) | the library, `Learn.xtl` |
| [`docs/`](docs/README.md) | the reference: every function, its type, examples, provenance |
| [`demos/`](demos/) | programs that use it |
| [`tests/`](tests/) | reg-rs baselines: the test programs, the pinned export types (`types.rgt`) and the demos (`demo-*.rgt`) |

```bash
just demo Learn         # run its demos
just run Learn          # run its test programs
just test-lib Learn     # check every baseline
```

# NN

Neural-network building blocks: activations, softmax by row, dense
layers, one-hot labels, loss and accuracy.

```
"nn:" u_se< "NN"
```

| Directory | What |
| --------- | ---- |
| [`src/`](src/) | the library, `NN.xtl` |
| [`docs/`](docs/README.md) | the reference: every function, its type, examples, provenance |
| [`demos/`](demos/) | programs that use it (XOR by a two-layer network) |
| [`tests/`](tests/) | reg-rs baselines: the test programs, the pinned export types (`types.rgt`) and the demos (`demo-*.rgt`) |

```bash
just demo-lib NN           # run its demos
just run-lib NN            # run its test programs
just test-lib NN           # check every baseline
```

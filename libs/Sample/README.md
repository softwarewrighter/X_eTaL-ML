# Sample

Choosing the next token from scores: greedy, temperature, top-k, drawing by given uniform numbers

```
"sm:" u_se< "Sample"
```

| Directory | What |
| --------- | ---- |
| [`src/`](src/) | the library, `Sample.xtl` |
| [`docs/`](docs/README.md) | the reference: every function, its type, examples, provenance |
| [`demos/`](demos/) | programs that use it |
| [`tests/`](tests/) | reg-rs baselines: the test programs, the pinned export types (`types.rgt`) and the demos (`demo-*.rgt`) |

```bash
just demo Sample         # run its demos
just run Sample          # run its test programs
just test-lib Sample     # check every baseline
```

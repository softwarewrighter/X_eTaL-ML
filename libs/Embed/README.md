# Embed

Word embeddings from counts: co-occurrence, PPMI, a fixed projection, cosine neighbors

```
"em:" u_se< "Embed"
```

| Directory | What |
| --------- | ---- |
| [`src/`](src/) | the library, `Embed.xtl` |
| [`docs/`](docs/README.md) | the reference: every function, its type, examples, provenance |
| [`demos/`](demos/) | programs that use it |
| [`tests/`](tests/) | reg-rs baselines: the test programs, the pinned export types (`types.rgt`) and the demos (`demo-*.rgt`) |

```bash
just demo Embed         # run its demos
just run Embed          # run its test programs
just test-lib Embed     # check every baseline
```

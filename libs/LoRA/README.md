# LoRA

Low-rank adapters (LoRA): the adapted forward pass in factored form, the merge, the counts

```
"lo:" u_se< "LoRA"
```

| Directory | What |
| --------- | ---- |
| [`src/`](src/) | the library, `LoRA.xtl` |
| [`docs/`](docs/README.md) | the reference: every function, its type, examples, provenance |
| [`demos/`](demos/) | programs that use it |
| [`tests/`](tests/) | reg-rs baselines: the test programs, the pinned export types (`types.rgt`) and the demos (`demo-*.rgt`) |

```bash
just demo LoRA         # run its demos
just run LoRA          # run its test programs
just test-lib LoRA     # check every baseline
```

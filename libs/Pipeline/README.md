# Pipeline

ML data flow written left to right, `|>` between stages: `pipe:p_ipe<` turns the pipeline into X_eTaL, one line per stage, with named intermediates and shape contracts checked when the program runs; `pipe:t_ext<` gives the pipeline back as read

```
"pipe:" u_se< "Pipeline"
```

| Directory | What |
| --------- | ---- |
| [`src/`](src/) | the macro library, `Pipeline.xtlm` |
| [`docs/`](docs/README.md) | the reference: the call, what it refuses, the expansion, examples |
| [`demos/`](demos/) | programs that use it: `mlp.xtl`, a two-layer network and a stable softmax as pipelines, beside the same by hand |
| [`tests/`](tests/) | reg-rs baselines: the test programs, the pinned macro types (`types.rgt`), the demo (`demo-*.rgt`) and every program's expansion (`expand-*.rgt`) |

```bash
just demo Pipeline          # run its demo
just run-lib Pipeline       # run its test programs
just test-lib Pipeline      # check every baseline
just expand-lib Pipeline mlp   # the demo's calls and what each became
```

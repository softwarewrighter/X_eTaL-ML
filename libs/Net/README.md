# Net

A network written as one line: a macro library that turns
`"784 128 relu 10 softmax"` into an ordinary forward function of NN
calls, visible with `xetal expand`.

```
"nn:"  u_se< "NN"
"net:" u_se< "Net"
```

| Directory | What |
| --------- | ---- |
| [`src/`](src/) | the macro library, `Net.xtlm` |
| [`docs/`](docs/README.md) | the reference: every macro, what it writes, examples, limits |
| [`demos/`](demos/) | programs that use it (XOR as one line) |
| [`tests/`](tests/) | reg-rs baselines: the test programs, what each becomes after expansion (`expand-*.rgt`), the macros' types (`types.rgt`) and the demos |

```bash
just demo-lib Net          # run its demos
just expand-lib Net xor    # what the macros wrote
just test-lib Net          # check every baseline
```

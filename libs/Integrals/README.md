# Integrals

a definite integral written as a paper prints it, by a quadrature rule: `@ integ:i_ntegral< "integral tau:=a..b by trapezoid 64 | body"` gives the value, `integ:t_ex<` its LaTeX; the continuous convolution of two boxes is its demo

```
"integ:" u_se< "Integrals"
```

| Directory | What |
| --------- | ---- |
| [`src/`](src/) | the macro library, `Integrals.xtlm` |
| [`docs/`](docs/README.md) | the reference: the call, what it refuses, the rules and their orders, examples |
| [`demos/`](demos/) | programs that use it: `convolution.xtl`, two unit boxes convolved into a triangle, and e^x's area; `gelu.xtl`, GELU as the integral that defines it against the tanh approximation, and a Gaussian KL divergence beside its closed form |
| [`tests/`](tests/) | reg-rs baselines: the test programs, the pinned macro types (`types.rgt`), the demo and every expansion (`expand-*.rgt`) |

```bash
just demo Integrals                      # run its demo
just run-lib Integrals                   # run its test programs
just test-lib Integrals                  # check every baseline
just expand-lib Integrals convolution    # the demo's calls and what each became
```

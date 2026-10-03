# foundation

Saga 1 of X_eTaL-ML (docs/plan.md): the process, the vendored
interpreter, both layouts (demos/<slug>/ as in X_eTaL-demos,
libs/<Name>/ as in X_eTaL-libraries), the three ML demos taken over
from X_eTaL-demos and live, and the first two ML libraries.

Model: ../X_eTaL-demos (demo layout, microscope pages, vendoring),
../X_eTaL-libraries (library layout, reg-rs tests, pinned types),
../X_eTaL (CHANGES.md). Why: ../X_eTaL/docs/research3.txt.

Rules: X_eTaL only through the vendored snapshot in vendor/xetal/;
missing features and bugs go in docs/xetal-asks.md, workarounds
named; .xtlm macro libraries wait until X_eTaL supports them (ask M1).
Every step: tests exist and `just gate` passes, docs (README,
CHANGES.md, plan, asks, the demo's or library's page) updated,
.gitignore sane, a detailed commit to main including .agentrail/,
`agentrail complete`, push, then a report: what was pushed, next
steps, blockers, questions, asks.

## Steps

1. scaffold -- process, CLAUDE.md/AGENTS.md, README, COPYRIGHT,
   LICENSE, CHANGES.md, justfile, gate, docs/plan.md,
   docs/xetal-asks.md.
2. vendor-xetal -- just vendor [REF], VENDORED, just xetal / eval,
   tools/vendor-probe, check-vendor in the gate.
3. layout -- demos/_template + demo tooling (reg-rs), templates/Library
   + library tooling (scripts/xt, pinned types), shared/microscope.
4. port-demos -- ternary-net, moe-router, cnn-digits (CLI) from
   X_eTaL-demos with trainers; baselines re-run; asks re-checked.
5. pages -- the live catalog and demo pages, browser checks, Pages.
6. nn -- the NN library (nn:); demos moved onto it.
7. quant -- the Quant library (qz:); ternary-net moved onto it.

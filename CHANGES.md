# Changes

Every commit, newest first, grouped by day. Times are Pacific
(UTC-07:00), as committed.

Categories: `feat` new capability, `fix` a bug or wrong behavior,
`refactor` structure without behavior change, `test` tests only,
`build` build and tooling, `demo` a demo or a change to one, `lib` a
library or a change to one, `docs` documentation, `plan` saga
planning and reordering, `release` milestone release, `chore`
agentrail bookkeeping (step complete, saga archive), `vendor` a
refresh of the vendored X_eTaL.

## 2026-10-03

- 08:58 `chore` Saga step vendor-xetal completed.
- 08:56 `build` Vendoring: `just vendor [REF]` (scripts/vendor-xetal.sh: git archive of a committed ref of ../X_eTaL), `just xetal` (built into target/xetal/ with XETAL_BUILD_SHA so `xetal --version` names the vendored commit), `xetal-version`, `eval`, `check-vendor` (eval, version, a demo, a standard-library import, tools/vendor-probe natively and for wasm32) in the gate; .cargo/config.toml (one target dir, the wasm stack size).
- 08:54 `vendor` X_eTaL abb8274 vendored (transpose, exponent literals, linear Int strands, build provenance; no .xtlm yet).
- 08:42 `chore` Saga step scaffold completed.
- 08:40 `build` Scaffold: the agentrail saga foundation (scaffold, vendor-xetal, layout, port-demos, pages, nn, quant); CLAUDE.md (AGENTS.md a symlink); README (why an array language for ML, how X_eTaL differs from APL, the three kinds of extensibility, the X_eTaL repositories, demos, libraries); COPYRIGHT and LICENSE (MIT) as in the sibling repos; docs/plan.md (A1-A12, the gallery, the libraries, sagas 1-4, the Net.xtlm design); docs/xetal-asks.md (M1-M9); .gitignore; justfile; the gate (ASCII-only markdown).
- 07:07 `docs` The empty README (the repository's first commit).

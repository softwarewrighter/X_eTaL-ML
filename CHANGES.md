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

- 10:45 `fix` The catalog's moe-router card showed a picture of a "404 File not found" page (the screenshot came that way from X_eTaL-demos; the page itself works). `just screenshots` serves pages/ on a free port and refuses to shoot a page that does not answer 200, so a stale server on a fixed port can no longer answer instead; both screenshots retaken, pages/ rebuilt.
- 10:04 `chore` Saga step pages completed (deploy verified: the catalog, ternary-net and moe-router pages answer at https://softwarewrighter.github.io/X_eTaL-ML/).
- 10:00 `build` The live site: `just pages` (trunk per demo into pages/<slug>/ under /X_eTaL-ML/, the catalog pages/index.html retitled ML with links to the sibling sites), `serve-pages`, `serve`, `browser-check` (headless Chrome; browser-SLUG baselines for ternary-net and moe-router, passing), `screenshots`; .github/workflows/pages.yml (upload only); GitHub Pages enabled at https://softwarewrighter.github.io/X_eTaL-ML/; README live links; the logo as images/modern-xetal-logo.jpg with the favicon.
- 09:42 `chore` Saga step port-demos completed.
- 09:40 `demo` ternary-net, moe-router and cnn-digits taken over from X_eTaL-demos (programs, READMEs, reg-rs CLI baselines, web apps, trainers with `just ternary-train` and `just cnn-train`, scripts/mnist.sh): all baselines and web tests pass unchanged on abb8274, and both trainers rewrite their weights byte for byte. moe-router's page passes the word numbers as Ints (ask M7 landed). Asks M3-M9 re-run against abb8274: M7, M8 landed; M3 slower (about 700 ns per multiply-add); M4, M5, M9 open.
- 09:12 `chore` Saga step layout completed.
- 09:10 `build` Layout: demos/<slug>/ (demos/_template, scripts/demos.py, test-demos.sh with reg-rs baselines, run-demo.sh, new-demo.sh, selftest-demos.sh, build-catalog.py) as in X_eTaL-demos, and libs/<Name>/ (templates/Library, scripts/xt, libs.py, test-libs.sh with pinned types, run-lib.sh, new-lib.sh, selftest-libs.sh, check-examples.py) as in X_eTaL-libraries; shared/microscope taken over; recipes (demos, new-demo, run, show, test-demo, bless, libs, path, new-lib, run-lib, demo-lib, show-lib, types, test-lib, bless-lib, test); the gate runs all of it.
- 08:58 `chore` Saga step vendor-xetal completed.
- 08:56 `build` Vendoring: `just vendor [REF]` (scripts/vendor-xetal.sh: git archive of a committed ref of ../X_eTaL), `just xetal` (built into target/xetal/ with XETAL_BUILD_SHA so `xetal --version` names the vendored commit), `xetal-version`, `eval`, `check-vendor` (eval, version, a demo, a standard-library import, tools/vendor-probe natively and for wasm32) in the gate; .cargo/config.toml (one target dir, the wasm stack size).
- 08:54 `vendor` X_eTaL abb8274 vendored (transpose, exponent literals, linear Int strands, build provenance; no .xtlm yet).
- 08:42 `chore` Saga step scaffold completed.
- 08:40 `build` Scaffold: the agentrail saga foundation (scaffold, vendor-xetal, layout, port-demos, pages, nn, quant); CLAUDE.md (AGENTS.md a symlink); README (why an array language for ML, how X_eTaL differs from APL, the three kinds of extensibility, the X_eTaL repositories, demos, libraries); COPYRIGHT and LICENSE (MIT) as in the sibling repos; docs/plan.md (A1-A12, the gallery, the libraries, sagas 1-4, the Net.xtlm design); docs/xetal-asks.md (M1-M9); .gitignore; justfile; the gate (ASCII-only markdown).
- 07:07 `docs` The empty README (the repository's first commit).

# macros

Saga 3 of X_eTaL-ML (docs/plan.md): the network macro. X_eTaL v0.1.0
has .xtlm macro libraries and xetal expand (asks M1, M2 landed); this
saga writes Net.xtlm, a macro that turns a layer list into an ordinary,
visible, type-checked forward function built from nn: calls, and a
demo that shows the source beside its expansion.

Rules as before (CLAUDE.md): pinned X_eTaL only; asks filed,
workarounds named; American spellings; programs short, data in files;
a page shows all the code it runs. Every step: tests exist and
`just gate` passes, docs updated, .gitignore sane, a detailed commit
to main including .agentrail/, `agentrail complete`, push, `just
publish` and `just check-live` when a page changed, then a report.

## Steps

1. macro-survey -- read what X_eTaL v0.1.0 implemented (.xtlm, MC10 to
   MC13, lib/Macros.xtlm, xetal expand); confirm or revise plan A10
   and the Net design; the library tooling accepts .xtlm.
2. net-macro -- libs/Net/src/Net.xtlm: the network macro; tests of the
   expansion (xetal expand) and of the result; page; a demo.
3. net-demo -- a demo (CLI and page) of a network written with the
   macro, its expansion beside it.

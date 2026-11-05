# Asks from X_eTaL-fine-tuning

What [X_eTaL-fine-tuning](https://github.com/softwarewrighter/X_eTaL-fine-tuning)
asks of this repository, filed in its `docs/xetal-ML-asks.md` (FM1,
FM2, ...), with this repository's word on each. That repository pins a
commit of this one (`XETAL_ML_COMMIT`), so an ask marked landed is
usable there once its pin passes the commit named.

Status: **landed** (in the commit named), **planned** (a step of
`docs/plan.md` named), **not planned** (no step yet: when it would be
judged), **declined** (with its reason).

Read on 2026-11-05 at their file's state of that day.

| # | Status | What landed or is planned | Where |
| - | ------ | ------------------------- | ----- |
| FM1 | landed (step fine-tuning-asks) | `shared/microscope/build.rs` embeds the library roots a dependent names in `MICROSCOPE_LIBS` (directories separated by `:`, each a root of `<Name>/src/<Name>.xtl` or one library's own directory; absolute, or `relative = true` in the dependent's `.cargo/config.toml` `[env]`), after this repository's, a dependent's library replacing one of the same name; `microscope::libs::names()` lists what was embedded; `microscope::libs::add("Name.xtl", source)` is documented as the other way, at run time | `shared/microscope/build.rs`, `src/libs.rs`, `tests/chrome.rs` |
| FM2 | landed (step fine-tuning-asks) | `microscope::chrome::Site` (name, URL, commit, the logo's link, the footer's links) with `header_of(&site, ...)` and `footer_of(&site)`; `header` and `footer` are `THIS` site's. Another repository's footer shows its own repository and commit, then `X_eTaL-ML <commit>` (`chrome::ML_COMMIT`, the clone's commit) and the pinned X_eTaL; `chrome::provenance(&site)` gives that text | `shared/microscope/src/chrome.rs`, `tests/chrome.rs` |
| FM3 | landed (step fine-tuning-asks) | `alpha lo:s_calingRs r`, rsLoRA's alpha / sqrt r, beside `lo:s_caling` | `libs/LoRA` |
| FM4 | planned, in part | QLoRA: steps quant-nf4 and qlora; VeRA: step vera (Saga 6). DoRA: not planned, as asked (after X_eTaL-fine-tuning has built and checked it) | `docs/plan.md`, Saga 6 |
| FM5 | landed (step fine-tuning-asks) | the explorer's heat map is `microscope::heat` (`Matrix`, `heat`, `largest`, `column`, `fill`; CSS `figure.heat` in `microscope.css`); the LoRA explorer uses it. Your offered additions (a legend, a cell inspector, keyboard focus) are welcome as a patch to it or a further ask | `shared/microscope/src/heat.rs` |
| FM6 | planned | steps moe-lib, moe-base and moe-finetune (Saga 6) | `docs/plan.md`, Saga 6 |
| FM7 | not planned | a FineTune library (tasks, splits, validation and retention, a run's record, a comparison harness, full and selective fine-tuning). Judged when your Saga 2 needs the harness; your Saga 1 workaround stands | |
| FM8 | not planned | a WeightSpace library (MeZO, evolution strategies and their variants). Judged after your Saga 4's planning step specifies it | |
| FM9 | not planned | an Edit library (ROME, MEMIT, AlphaEdit). Judged after your Saga 5's planning step specifies it | |
| FM10 | not planned | shared-basis adapters, in LoRA or their own library, with a family of tasks and transfer and forgetting measured. Judged after your Saga 3's planning step; `lo:s_eeded` already gives a basis from a seed | |

FM7 to FM10 would be a saga of this repository after Saga 6, in the
order your sagas need them (FM7, FM10, FM8, FM9), unless the user asks
for one sooner.

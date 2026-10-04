# launch

Saga 2 of X_eTaL-ML (docs/plan.md, "Reprioritized for the launch"):
after ../X_eTaL/docs/research4.txt, no new breadth; finish what a
newcomer sees first, measure the table / inner regression, sync the
asks, and record the X_eTaL commit this repo is known to work with.

Rules as in saga 1 (CLAUDE.md): vendored X_eTaL only; asks filed,
workarounds named; .xtlm waits (saga 3). Every step: tests exist and
`just gate` passes, docs (README, CHANGES.md, plan, asks, the demo's
pages and recording) updated, .gitignore sane, a detailed commit to
main including .agentrail/, `agentrail complete`, push, then a report:
what was pushed, next steps, blockers, questions, asks.

## Steps

1. cnn-digits-page -- the CNN live page: draw or pick a digit; every
   stage with its shape; click a conv output for patch x kernel;
   X_eTaL pass equals the trainer's Rust pass; browser baseline.
2. attention -- the attention microscope (definitions in the
   program): Q K V, scores, softmax heatmap, output, causal mask; CLI
   baseline, page, browser baseline, recording.
3. bench -- just bench: the ML workloads timed against the abb8274
   baseline (docs/speed.md), warning over 15 percent.
4. start-here -- the catalog as the front door; README trimmed to what
   works today with a status table.
5. asks-audit -- refresh the vendor, re-run every ask, remove landed
   workarounds, list promotion blockers.
6. release-1 -- docs, retrospective, the known-good X_eTaL commit.

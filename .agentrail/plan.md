# moe-finetune

Saga 6 of X_eTaL-ML (docs/plan.md): the user, 2026-11-01, demonstrate
the ways of fine-tuning mixture-of-experts models with new macros,
libraries and demos, in the order the research recommends. The
research: docs/fine-tune-moe-research.txt; the plan with its reasons
and the research's line numbers: docs/fine-tune-moe-demo-plan.md.

Adapters (LoRA, LoRA-FA, QLoRA, VeRA) on one projection first, then on
a tiny MoE's experts, the router frozen before it is trained; every
method on the same base model, measured by the new task, the
regression of the old, the trainable numbers and bytes, and the
routing. Everything in X_eTaL on a CPU, at the command line and live
in the browser; tiny sizes; shapes, the merge identity and finite
differences checked; no special LoRA syntax in the language.

Every step: tests, the gate, docs (README, the library or demo pages,
CHANGES, docs/plan.md, the asks), a commit, a push, a publish when a
page changes, a report.

1. lora-lib  2. lora-grads  3. lora-explorer  4. adapter-macro
5. quant-nf4  6. qlora  7. moe-lib  8. moe-base  9. moe-finetune
10. vera  11. adapters-deploy  12. preference (optional)

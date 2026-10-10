# post-launch

Saga 5 of X_eTaL-ML (docs/plan.md): the follow-up stream after the
launch, which the user has postponed further (2026-10-09), so it goes
first. The libraries the demos were written ahead of (Quant, Conv,
Attention, Norm, Sample, Embed, Optim), then a tiny GPT, an embedding
explorer, the libraries on the live site, and the deferred research
demos. Every step: tests, the gate, docs, a commit, a push, a publish
when a page changes, a report.

0. xetal-091 -- pin X_eTaL past its step 091 (names bound once; the
   repository is already clean, step bound-once) and X_eTaL-libraries
   at its latest; every baseline re-run, bench-check.
1. quant -- the Quant library (qz:); ternary-net moved onto it, its
   baselines unchanged.
2. conv -- Conv (cv:); cnn-digits and cnn-backprop moved onto it.
3. attention-lib -- Attention (at:); the attention demo keeps its lesson.
4. norm-sample -- Norm (nm:), Sample (sm:).
5. micro-gpt -- microgpt inference in X_eTaL on offline-trained weights.
6. embedding-explorer -- Embed (em:); 64 dimensions to a rotatable 3-D
   cloud by Learn's PCA.
7. optim -- Optim (op:): Adam and SGD as library steps (Net's t_rain<
   and the demos' hand-written Adam onto it).
8. lib-site -- the libraries on the live site, as X_eTaL-libraries' site.
9. research -- the deferred research demos (world model, diffusion).

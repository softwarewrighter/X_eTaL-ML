# in-context-learning

Saga 7 of X_eTaL-ML (docs/plan.md): the user, 2026-11-17, relate Gao
and Gao, "A Circuit for In-Context Learning" (arXiv 2611.05344), and
XSkill (arXiv 2603.12056) to this repository. Both learn without
changing a weight. Saga 6 showed fine-tuning shared weights forgets and
adapters kept apart do not; this saga adds the end where nothing is
trained: the knowledge in the prompt, or in a memory retrieved into it.

The circuit: writer heads stamp each label token with the class of the
sentence before it (key: class, value: label); reader heads match the
query's class against those keys by a softmax and copy the label; a
gate opens the read only once a label has repeated (otherwise the
readers rest on the sink at position 0); brake heads work against the
most recent label. Built by hand in 3 attention layers (the paper's
Section 12) and grown in a 3.4M-parameter model trained from scratch.

Everything in X_eTaL on a CPU, at the command line and live in the
browser; tiny sizes; every number checked against the paper or a
finite difference; the scale stated (a toy's in-context learning is
weaker than a large model's).

Every step: tests, the gate, docs (README, the library or demo pages,
CHANGES, docs/plan.md, the asks), a commit, a push, a publish when a
page changes, a report.

1. xetal-repin  2. icl-circuit  3. icl-trained  4. icl-vs-finetune
5. icl-timeline

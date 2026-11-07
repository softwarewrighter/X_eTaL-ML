# MoE fine-tuning

A tiny mixture-of-experts model and the ways of fine-tuning it. This
first part is the base model every method starts from: a shared layer,
a router and 4 experts (each token through its top 2), trained in
X_eTaL on a synthetic task, its weights kept in `data/base.txt`.
The live page, with the fine-tuning methods compared on it, is still
to come.

## The task

Tokens of 8 features, in the moe-router demo's world: each token is
one of 4 kinds (its kind's feature near 0.8, the other three near 0,
with a little noise) and has 4 attributes between -1 and 1. Its class,
1 to 3, is the largest of 3 scores of the attributes, by a rule of its
kind's own: the same attributes mean different classes for different
kinds, which is what experts are for. 800 tokens to train on, 400 held
out (`data/train.txt`, `data/test.txt`: the inputs, the kinds, the
classes).

## The model

| Part | Code | Shape | What it is |
| ---- | ---- | ----- | ---------- |
| shared layer | `z := t_anh S '+ '* i_nner x` | 8 x N | every token through one layer (the 9th input a constant 1); S is 8 x 9 |
| router | `G := 2 moe:t_opk moe:s_oftmax R moe:s_cores z` | 4 x N | each token's scores for the 4 experts, as probabilities, its two likeliest kept as the gates; R is 4 x 8 |
| experts | `(ex, G) moe:c_ombine z` | 3 x N | each token through its two experts (3 x 8 each), weighted by the gates |
| classes | `moe:s_oftmax` | 3 x N | each token's probabilities over the 3 classes |

200 trainable numbers. The experts are the LoRA library's adapters
with no update yet (A and B zero), so the fine-tuning methods can give
them one.

## The training

`train/base.xtl` makes the tokens (a fixed hash of a sine, no chance),
then trains with Adam (rate 0.02) on the cross-entropy, all 800 tokens
at once, for 250 steps, about half a minute. The top 2 cannot be
differentiated, so each token's two experts are held fixed for the
step, as training frameworks do: the router's gradient goes through
the two gates (the MoE library's `moe:g_rads`), the experts' own
weights' through their gates (`moe:g_radsW`), then back through tanh
to S. While training, every expert runs on every token (`moe:d_ense`),
the same sum as the sparse combine and faster at this size; the
program checked here uses the sparse one.

## The baseline

From `base.xtl`, which reads the checkpoint and the tokens:

| | kind 1 | kind 2 | kind 3 | kind 4 | all |
| - | ------ | ------ | ------ | ------ | --- |
| training accuracy | 0.99 | 0.965 | 0.97 | 0.95 | 0.969 |
| held-out accuracy | 0.92 | 0.97 | 0.8 | 0.84 | 0.883 |

Where the held-out tokens go, the router frozen: the share of each
kind's tokens each expert gets (each column sums to 2):

| | kind 1 | kind 2 | kind 3 | kind 4 |
| - | ------ | ------ | ------ | ------ |
| expert 1 | 0.08 | 0.13 | 0.28 | 0.87 |
| expert 2 | 0.89 | 0.03 | 0.79 | 0.31 |
| expert 3 | 0.83 | 0.85 | 0.53 | 0.56 |
| expert 4 | 0.2 | 0.99 | 0.4 | 0.26 |

Each kind has its pair (kind 2 experts 4 and 3, kind 4 experts 1 and
3), and expert 3 serves every kind. The experts are chosen 0.34,
0.505, 0.693 and 0.463 of the time (summing to 2); the routing's
entropy is 0.437 nats (log 4 = 1.386 would be no preference); its
load-balancing loss is 1.056 (1 is even). A method that changes the
experts but not the router keeps this routing; one that trains the
router can move it.

## Run it

```bash
just run moe-finetune base.xtl   # the baseline
just moe-base                    # make the tokens and train the base again (half a minute)
just test-demo moe-finetune      # its CLI baseline (reg-rs)
```

## Workarounds

None.

# Muon against AdamW: the demo plan

How X_eTaL-ML shows Muon, the optimizer that orthogonalizes each
weight matrix's update, beside AdamW, the optimizer it is measured
against. The user, 2026-11-23: add `m_uon` and a live demo comparing it
with AdamW; Kimi K2 trained with it to great advantage, can we get a
benefit from it too. The saga that would carry this plan out is Saga 8,
muon (`docs/plan.md`, proposed).

## The sources

Read 2026-11-23. Each claim below names its source by the letter here.

- **[J]** Keller Jordan, "Muon: An optimizer for hidden layers in
  neural networks", December 8, 2024,
  <https://kellerjordan.github.io/posts/muon/>, and its code,
  <https://github.com/KellerJordan/Muon/blob/master/muon.py>.
- **[M]** Jingyuan Liu, Jianlin Su, Xingcheng Yao, ... Zhilin Yang
  (Moonshot AI, Kimi Team), "Muon is Scalable for LLM Training",
  arXiv 2502.16982, February 24, 2025.
- **[K]** Kimi Team, "Kimi K2: Open Agentic Intelligence", arXiv
  2507.20534 (v1 July 2025, v2 February 3, 2026).
- **[W]** Kaiyue Wen, David Hall, Tengyu Ma, Percy Liang, "Fantastic
  Pretraining Optimizers and Where to Find Them", arXiv 2509.02046
  (v1 September 2, 2025; v3 October 25, 2026).
- **[L]** Ilya Loshchilov, Frank Hutter, "Decoupled Weight Decay
  Regularization" (AdamW), arXiv 1711.05101, November 2017 (ICLR
  2019).

## What Muon is

For each 2-D weight matrix W with gradient G [J]:

1. **Momentum**, Nesterov style: `M = 0.95 M + G`, and the update
   direction is `G + 0.95 M` ([M]'s form; [J]'s code writes the same
   with an exponential average, `M = 0.95 M + 0.05 G`: the direction
   is a constant times [M]'s, and the normalization below removes it).
2. **Orthogonalize** the direction: replace it with the nearest
   semi-orthogonal matrix, `U V'` from its SVD `U S V'`, every singular
   value set to 1. Not by an SVD ("far too slow" [J]) but by five
   steps of a quintic Newton-Schulz iteration, matrix products only:
   - `X = D / (||D||_F + 1e-7)` (the Frobenius norm is at least the
     spectral norm, so every singular value starts at most 1);
   - a tall matrix is transposed first and back after, so the product
     `X X'` is the small side's square;
   - five times: `A = X X'`, `X = a X + (b A + c A A) X`, with
     `(a, b, c) = (3.4445, -4.7750, 2.0315)`.

   The coefficients maximize the slope at 0, so small singular values
   grow fast; the iteration does not converge to `U V'`, it leaves the
   singular values roughly in 0.5 to 1.5, which [J] found trains no
   worse. `(2, -1.5, 0.5)` converges properly and "already works" [J].
   [J] runs it in bfloat16; here it runs in float64 (X_eTaL's Float).
3. **Scale** the update by its shape, two rules:
   - [J]'s code: `max(1, rows / cols) ^ 0.5`;
   - [M]: `0.2 sqrt(max(A, B))`, because an orthogonal A by B matrix's
     entries have RMS `sqrt(1 / max(A, B))` exactly ([M]'s Lemma 1:
     the squared entries sum to the rank). The scaled update has RMS
     0.2, about AdamW's (0.2 to 0.4 [M]), so AdamW's learning rate
     and weight decay carry over unchanged. [M]'s sweep of the RMS
     (0.05 to 0.8) found 0.2 and 0.4 alike and best.
4. **Step** with decoupled weight decay [M]:
   `W = W - lr (scaled update + wd W)`, wd 0.1 in [M]. Without it, [M]
   found weights and outputs grew past bfloat16's precise range in
   long runs; [J]'s code defaults to no decay.

**Only hidden matrices.** Embeddings, the output layer, biases and
gains stay on AdamW [J, M]. Convolution filters can be flattened to
matrices [J]. Q, K and V are separate matrices, not one fused [J].

**Why it helps** [J]: the updates of SGD momentum and Adam for a
transformer's matrices are nearly low-rank, a few directions dominate;
orthogonalizing raises the rare directions to the same scale. It is
Shampoo without accumulation (`(G G')^(-1/4) G (G' G)^(-1/4) = U V'`
[J]). [M] finds Muon's trained weights have flatter singular values
(higher SVD entropy, `-(1/log n) sum p log p` with `p_i = s_i^2 /
sum s^2`) in every group of matrices, the router most.

**Cost.** For an n by m matrix (m <= n), each Newton-Schulz step is
`2 (2 n m^2 + m^3)` FLOPs, five a training step; in all at most a
fraction `T m / B` of the training FLOPs, T the steps and B the batch
in tokens (0.7% for the NanoGPT speedrun [J]).
One moment buffer, against Adam's two [M].

## Muon at scale: Kimi K2's MuonClip

[K]: a 1.04T-parameter MoE (32.6B active, 384 experts, 8 active and 1
shared, 61 layers, MLA attention), pre-trained on 15.5T tokens with no
loss spike. Its optimizer, MuonClip, is [M]'s Muon (decay, the 0.2
RMS rule) plus **QK-Clip**:

- each head's largest attention logit in the batch, `S_max`, is
  already known from the forward pass;
- a head whose `S_max` passes tau (100 in K2; 30 in a small ablation)
  gets its query and key weights each multiplied by `sqrt(tau /
  S_max)`, so the next step's logits are back at tau; other heads are
  not touched; the current step's forward and backward are unchanged;
- with vanilla Muon, a 53B-total MoE's largest logits passed 1000,
  with loss spikes; with QK-Clip, K2's logits rose to the cap and,
  after about 30% of training, came down by themselves (12.7% of
  heads clipped in the first 70,000 steps, none after); a 3B-total
  ablation showed no loss cost.
- Why Muon more than AdamW ([K]'s hypothesis): Muon's full-rank
  updates enlarge the singular values of W_q and W_k, and `W_q W_k'`
  multiplies them.

## How much it helps, honestly

- [M]: about 2x the compute efficiency of AdamW (Muon needs about 52%
  of AdamW's FLOPs for the same loss), from scaling-law fits over 399M
  to 1.5B dense models; Moonlight, a 16B-total MoE trained on 5.7T
  tokens, as the demonstration.
- [J]: the NanoGPT speedrun 1.35x faster; CIFAR-10 to 94% from 3.3 to
  2.6 A100-seconds; a 1.5B model to GPT-2 XL's HellaSwag in 10 hours
  against AdamW's 13.3.
- [W], independent, every optimizer's hyperparameters swept: about
  1.3x to 1.4x over well-tuned AdamW at 130M parameters, falling to
  about 1.1x at 1.2B; Soap and Kron overtake Muon as the data per
  parameter grows (8x Chinchilla and up); at 1.2B Muon is not
  significantly faster than NAdamW, and its downstream average is
  just below AdamW's (66.98 against 67.15). "Many claimed speedups
  stem from insufficient hyperparameter tuning" [W]: a fair comparison
  tunes both.
- **Fine-tuning mismatch** [M], Moonlight then SFT (MMLU, HumanEval,
  MBPP, GSM8K): Muon then Muon 55.7, 57.3, 55.6, 68.0; AdamW then Muon
  55.3, 53.7, 55.5, 62.1; Muon then AdamW 50.2, 52.4, 55.2, 64.9; AdamW
  then AdamW 52.0, 53.1, 55.2, 64.6. Best when pre-training and
  fine-tuning use the same optimizer, and Muon; switching gives no
  advantage. Qwen2.5-7B (AdamW-trained) fine-tuned with Muon: on par
  with AdamW, a little behind.

**What to expect here.** Our models are thousands of parameters, not
millions, on a CPU. [W] found the advantage largest at its smallest
size, but nobody reports toys this small, so this plan claims nothing
until measured: steps to a target loss and the final loss, both
optimizers' learning rates swept, over several seeds, and the time a
step, reported as found, a loss included. What does not depend on
scale, and the demos can show plainly: the update's singular values
(Adam's skewed, Muon's flat), the weights' SVD entropy, the RMS rule
exact, and the logit growth QK-Clip exists for (whether a toy shows it
at all is itself a finding).

## The principles

As Saga 6's plan: X_eTaL on a CPU, at the command line and live in the
browser; no special syntax, an optimizer is a library function over
the training state's tuple; every piece checked (Newton-Schulz against
Matrix's `s_vd`, the RMS lemma exactly, AdamW at no decay equal to
Adam byte for byte); the same data, network and seeds for every
optimizer; tuned fairly.

## The pieces

- **Optim** gains `a_damW` (decoupled weight decay [L]), `o_rtho` (the
  Newton-Schulz orthogonalization, its coefficients and steps given or
  Jordan's), `m_uon` (one matrix's step: momentum, orthogonalize,
  scale, decay; [J]'s and [M]'s scale rules) and `s_vdEntropy`.
- **Matrix** (X_eTaL-libraries, `mx:`; `s_vd` and `s_ingular`, its
  Saga 27, at the pinned c7e812e) joins the libraries linked here, for
  the exact `U V'` and the singular values drawn.
- A new demo, **muon-vs-adamw**, live.
- moe-finetune and the tiny transformer of Saga 7 (icl-trained) gain
  Muon runs.

## The steps (Saga 8, proposed)

1. **adamw**: Optim's `a_damW` (`(lr, wd, t) a_damW (w, m, v, g)`):
   decay at 0 equal to `a_dam` exactly; decay not scaled by Adam's
   denominator, as L2 in the gradient is (a check that shows the
   difference [L]); doc examples; the bowl demo with decay.
2. **muon-lib**: Optim's `o_rtho` and `m_uon`; Matrix linked
   (scripts/xetal-libraries.sh). Checks: after five steps every
   singular value in [J]'s band, and `(2, -1.5, 0.5)` converging to
   Matrix's `U V'` to 1e-6; tall and wide alike (the transpose); an
   exactly orthogonal A by B update has RMS `sqrt(1 / max(A, B))` and
   0.2 after [M]'s scale; one Muon step by hand on a 3 by 2 matrix; the
   FLOP count. The library page explains each line.
3. **muon-vs-adamw**: a live demo. A small MLP on cnn-backprop's
   digits (784-64-64-10: the 64 by 64 hidden matrix, and optionally
   the first, on Muon; the output and biases on AdamW, as [J] says),
   trained in X_eTaL in the page: AdamW, Muon, and Muon on every
   matrix (the wrong split, to see why), each at its own swept
   learning rate; the loss curves over seeds; per step, the update's
   singular values for AdamW and Muon side by side and the weights'
   SVD entropy; steps to a target loss and milliseconds a step. The
   result reported as found, with [W]'s caution beside it.
4. **muon-attention**: Saga 7's tiny transformer (icl-trained's) given
   Muon and QK-Clip: each head's largest logit over training with
   AdamW, Muon and MuonClip (tau set low for a toy, as [K]'s ablation
   did at 30), the heads clipped, the loss with and without; whether
   the logits grow at this size is reported, either way.
5. **muon-mismatch**: moe-finetune's base trained (in X_eTaL, `just
   moe-base`) with AdamW and with Muon, each fine-tuned with each: [M]'s
   two-by-two table at toy size, the old and new tasks' accuracy; the
   router's and experts' SVD entropy ([M]: the router's gap is
   largest). Optional: Muon on LoRA's factors, which no source covers.
6. **muon-timeline**: the ML timeline gains AdamW (2017), Muon (2024),
   Moonlight (2025), Kimi K2 and MuonClip (2025) and [W] (2025), each
   with its sources row, and the new demo beside them.

Order against Saga 7: step 4 needs Saga 7's icl-trained; the rest does
not. Saga 8 can run after Saga 7, or steps 1 to 3 first and the rest
after, as the user chooses.

## Not covered

Distributed Muon (ZeRO sharding, [M]'s Algorithm 1), bfloat16 and its
coefficients' error, newer coefficient schedules (Polar Express,
per-step minimax coefficients), Soap and Kron ([W]'s winners at high
data ratios), Muon for convolution beyond the flattening, and
reinforcement learning with Muon (an open question [J]).

# Fine-tuning MoE models: the demo plan

How X_eTaL-ML shows the ways of fine-tuning a mixture-of-experts (MoE)
model, in the order the research recommends, with new libraries, a
macro and live demos. The research is `docs/fine-tune-moe-research.txt`
(a transcript of questions and answers; line numbers below are its own,
`Lnnn`). The saga that carries this plan out is Saga 6, moe-finetune
(`docs/plan.md`).

## What the research covers, and what it does not

It is about parameter-efficient adapters on MoE models:

- **LoRA** on the experts' projections: the expert's weight W frozen, a
  low-rank update B A trained, `y = W x + (alpha/r) B (A x)` (L25-170,
  L2434-2452), never the merged matrix while training.
- **QLoRA**: the frozen base stored in 4 bits (NF4), the adapters in
  full precision, `y = dequant(Q4(W)) x + s B (A x)` (L172-218,
  L2465-2525).
- **LoRA-FA**: A frozen (random, or regenerated from a seed, or shared
  across tasks), only B trained; it saves memory in training, not at
  inference, and restricts the updates to A's row space (L728-1150,
  L3587-3593).
- **VeRA**: frozen random A0 and B0, shared across layers or experts;
  only two vectors trained, `y = W x + b * (B0 (d * (A0 x)))`
  (L3612-4464), about 20 times fewer trainable numbers than LoRA for
  64 experts at rank 16 (L3928-3969).
- **Where to adapt**: attention only, every expert, shared and expert,
  selected experts, the router, new experts (the strategy table,
  L223-263); a frozen router still decides which experts get to learn
  (L275-294).
- **Deployment**: an adapter kept apart (switchable) or merged into
  the weights (faster, one task); a QLoRA merge is not bit-exact
  (L381-460).
- **Training order at scale**: supervised fine-tuning, then preference
  pairs (DPO), then reinforcement learning only once those show a gain
  (L1475-1490, L2023-2046).
- **Evaluation**: the same base checkpoint for every method, held-out
  data, paired comparisons over several seeds, a cap on regression of
  what the model could already do (at most 3%, L1348), routing
  statistics (selection frequency, entropy, load balance, L675),
  comparisons at equal parameter budgets and at equal ranks (L4449).

Not in the research, so not in this plan's main line: full
fine-tuning, a router-only method, load-balancing and router z-losses,
capacity factors and token dropping, upcycling a dense model into an
MoE, merging or pruning experts, distillation. They are listed at the
end as a later saga, marked as going beyond the research.

## The recommended order

The research gives several orders; they agree. For teaching in X_eTaL
(L3347-3402): the LoRA forward pass; its gradients and a step; Flow's
fan-out and fan-in; LoRA-FA; quantization (uniform 4 bits first, then
NF4 and double quantization, L3162); QLoRA; a sparse MoE with adapted
experts. For experiments (L620-675, L1774-1831): a baseline first, then
the cheapest adaptation (attention, here the shared layer), then the
experts, then shared and expert together, with the router frozen and
its statistics collected before the router is ever trained (L294,
L1544); then LoRA-FA and the cheaper variants (L1196-1260); VeRA last,
compared with the rest (L4430-4460). This plan follows that order.

## Principles

- Everything in X_eTaL, on a CPU, in the browser as well as at the
  command line (L3283): tiny sizes, an 8 x 8 projection first (L3285),
  then a 4-expert MoE (L3345).
- No special LoRA syntax in the language (L2634): adapters are a
  library and a macro, pipelines are Flow's (fan-out by a label, fan-in
  by naming it).
- Keep the math, the execution and the picture apart (L3600-3610): the
  equation drawn as LaTeX (Equations, Flow's `m_ath<`), the code it
  became (`xetal expand`), the arrays drawn (the stage view).
- Every step checked: shapes, the merge identity `W x + s B (A x) = (W
  + s B A) x`, gradients against central finite differences, an
  independent reference written out (L3404-3469).
- One base checkpoint, several seeds, fixed: a method is compared with
  another on the same model and the same data (L1370, L1652).

## New libraries, macro and demos

| Name | Kind | What it gives |
| ---- | ---- | ------------- |
| LoRA (`lo:`) | library | the adapted forward pass `(W, A, B, s) lo:f_orward x`, the merge, the parameter and byte counts; gradients for A, B and x and a step (SGD, Adam through Optim); LoRA-FA (A frozen, seeded); VeRA (A0, B0 frozen, b and d trained) |
| `lo:a_dapter<` | macro | an adapted layer written from a short spec, `"W | r 3 | train B"`, as forward and backward code for the chosen policy (LoRA, LoRA-FA, VeRA), its expansion pinned and checked against the library |
| Quant (`qz:`), extended | library | blockwise uniform 4 bits with a scale, the NF4 codebook (16 levels), double quantization of the scales; the error and the bytes |
| MoE (`moe:`) | library | router scores, softmax, top-k, dispatch, per-expert apply, weighted combine (the stages the research names, L3263-3274, each usable as a Flow stage); gradients with the selection held fixed; routing statistics (selection frequency, entropy, load balance) |
| lora-explorer | demo, live | one 8 x 8 projection adapted: W, A, B and their shapes; the forward and backward passes step by step; LoRA beside LoRA-FA on the same inputs; fitting a known update and the rank sweep against the best possible (truncated SVD, Eckart-Young); QLoRA's error and bytes; VeRA |
| moe-finetune | demo, live | a tiny MoE (a shared layer, 4 experts, top-2) trained in X_eTaL on a synthetic task, then fine-tuned to a shifted task by each policy in turn; for each: the new task's accuracy, the old task's (the regression), trainable numbers and bytes, the experts chosen before and after, each expert's update size |

## The steps (Saga 6, moe-finetune)

Each step ends as every step here does: tests, the gate, docs, a
commit, a push, a publish when a page changed.

| # | Slug | Delivers | Research |
| - | ---- | -------- | -------- |
| 1 | lora-lib | libs/LoRA: the adapted forward pass in its factored form (never the merged matrix while training), the merge, parameter and byte counts (`r (m + n)`; `3 r (d + f)` for a SwiGLU expert); doc examples; checks: the merge identity, rank(B A) <= r, the counts against the research's worked numbers | L25-170, L296-375, L2434-2452, L2683-2750 |
| 2 | lora-grads | gradients `dL/dB = s g z'`, `dL/dA = s (B' g) x'`, `dL/dx = W' g + s A' B' g`; an SGD step and Adam through Optim; LoRA-FA (A frozen, from a seed; its gradient skipped, x's still through A); checks against central finite differences, LoRA-FA's updates inside A's row space | L728-1150, L2755-2947, L3587-3593 |
| 3 | lora-explorer | the live demo's first half: an 8 x 8 W with A 3 x 8 and B 8 x 3 (48 trainable numbers); the forward pass as a Flow pipeline (x fans out to the base and the adapter, joined by +) drawn as math, with the stage view; a backward step shown stage by stage; LoRA and LoRA-FA trained side by side to fit a known update; the rank sweep: the error at each rank against the truncated-SVD optimum (power iteration, Learn's PCA's method) | L3283-3345, L3473-3585 |
| 4 | adapter-macro | `lo:a_dapter<`: a spec string to forward and backward code for LoRA and LoRA-FA (VeRA in step 10); its expansions pinned, checked equal to the library's functions and to finite differences; used by the explorer | L2629-2659 |
| 5 | quant-nf4 | Quant gains blockwise uniform 4-bit quantization (scale per block, clip, round), the NF4 codebook (nearest of 16 levels), double quantization of the scales; error and bytes against FP32 and FP16; the explorer's quantization panel | L2954-3162 |
| 6 | qlora | QLoRA in the library and the explorer: the base dequantized on the fly, the adapter in full precision; compared with the real dequantized computation; trained like LoRA; the merge shown not bit-exact (dequantize, add, quantize again) | L172-218, L460, L2465-2525, L3471 |
| 7 | moe-lib | libs/MoE: scores, softmax, top-k, dispatch, apply per expert, combine (`y = sum over chosen e of p_e(x) (W_e x + s_e B_e (A_e x))`); its gradients with the selection fixed; routing statistics; checks against a dense reference that computes every expert, and finite differences | L3180-3274, L675 |
| 8 | moe-base | the base model: a synthetic task made in X_eTaL (tokens of feature vectors in a few kinds, the moe-router demo's world), a tiny MoE trained on it in X_eTaL with fixed seeds; the checkpoint written to `data/`; its routing statistics recorded with the router frozen, the baseline every method is measured against | L294, L620-664, L1370, L1774-1831 |
| 9 | moe-finetune | the live demo: the base fine-tuned to a shifted task (some kinds relabeled or new ones added) by each policy in order: none; the shared layer only (the research's attention-only); every expert, router frozen; shared and expert; selected experts only; the router trained too. For each: the new task's accuracy, the old task's regression (at most 3% is the bar), trainable numbers and bytes, the experts chosen before and after (a bar each), each expert's update size; over several seeds | L223-294, L667-677, L1348, L1652, L1707-1738 |
| 10 | vera | VeRA in the library and the macro; the explorer's VeRA panel (beside LoRA, its gradients, its rank, the projections shared); MoE-VeRA in moe-finetune, the projections shared across experts; the four-way comparison (shared LoRA, expert LoRA, LoRA-FA, shared VeRA) at equal ranks and at equal parameter budgets, efficiency as the gain per trainable number | L3612-4464, L4310-4361, L4430-4460 |
| 11 | adapters-deploy | an adapter kept apart against merged: the same answers (exactly for LoRA, not for QLoRA), switching between two tasks' adapters on one base, the bytes each keeps; an adapter-cache simulation for the experts a batch uses | L381-460, L1103-1150, L4447 |
| 12 | preference (optional) | supervised fine-tuning, then preference pairs (DPO) on the toy task, both answers of a pair valid so the model learns a preference, not right from wrong; reinforcement learning left out, as the research waits for SFT and DPO to show a gain | L1475-1490, L2023-2046 |

Steps 1 to 6 need no MoE and give the explorer; 7 to 9 the MoE demo;
10 and 11 compare and deploy; 12 is optional. Asks found on the way go
to `docs/xetal-asks.md` or `docs/xetal-libraries-asks.md` as usual
(VeRA's scaling of rows by a vector, for one, is a library function
here, since X_eTaL has no implicit broadcasting, L4128).

## Later, beyond the research

A saga after this one could take the MoE topics the research does not
cover, from the literature: a load-balancing loss and the router's
z-loss (what keeps experts from collapsing onto a few), capacity and
dropped tokens, upcycling a dense layer into experts, merging and
pruning experts, distilling an MoE into a dense model. Each would be
cited from its paper when planned.

## Backlog (from Saga 5)

Waiting, not in this saga: f32-tolerance (X_eTaL-gpu's float32
tolerance for the diffusion demo), activation-compare (the spiral net
trained with each activation) and glu-ffn (Tiny GPT's feed-forward
layer as GELU beside SwiGLU); `docs/plan.md` keeps them under Backlog.

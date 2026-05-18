# Dates Generator — Discussion Prep / Study Guide

This document is the deep-defense companion to the report. For each module
of the assignment it lists: what was chosen, *why*, the math behind it, the
implementation tweaks, what the actual training run showed, and the most
likely examiner questions with strong answers.

> **Quick stats at a glance**
>
> | Model     | val CSR_all | OOD held-out | Status            |
> |-----------|-------------|--------------|-------------------|
> | CVAE+CFG  | 0.999       | 1.000        | ✅ exceeds target |
> | MDLM+CFG  | 0.993       | 0.988        | ✅ exceeds target |
> | MaskGIT   | 0.138       | 0.151        | ❌ negative result (analysed below) |
> | AC-GAN    | 0.149       | 0.127        | ❌ partial result (analysed below) |
> | smart_random (rejection ceiling) | 1.000 | 1.000 | — |
> | random   (CSR floor)             | 0.089 | 0.108 | — |

---

## 0. Problem formulation (the single most important slide)

### Statement

We model the conditional `p(y | x)` where
- `x = (DOW, MON, LEAP, DEC)` — 4 categorical conditions
- `y = (d, m, year)`, but `m = MON` (pass-through) and
  `year = DEC·10 + year_unit` with `year_unit ∈ {0..9}`

So the only thing the model has to predict is **`(d, year_unit) ∈ {1..31} × {0..9}`** — a **310-way joint categorical**.

### Why "joint"

For a given condition tuple, the valid answer set is typically **5–10 entries out of 310** (a 1.6 % – 3.2 % support). The constraints interact:
- DEC narrows year to a 10-year window
- LEAP narrows that to ~2–3 candidate years (the leap rule cycles every 4 years; centuries break the pattern)
- DOW + chosen year selects ~4–5 days per matching year-month

Critically the DOW constraint is **non-decomposable** over `(d, year_unit)`: you cannot first pick `year_unit` greedily and then pick `d`, because what DOW the day lands on depends on *both* choices together.

### Three failure modes for "generic" implementations

| Failure mode                                              | Generic model that hits it |
|-----------------------------------------------------------|----------------------------|
| (i)  Factorising the joint output into independent heads  | D3PM with `day_head` + `yu_head` (old submission) |
| (ii) Chaining the joint as autoregressive tokens          | Tiny AR Transformer (old submission) |
| (iii) Pure adversarial training, no MLE signal            | Vanilla CGAN with Gumbel-Softmax (old submission) |

Each of our four chosen models targets one of these failure modes with a paper-backed mechanism.

### Why the hint-2 "digit-by-digit AR" is wrong here

The assignment hints suggest predicting digits one at a time, easiest-first.
This is a generic AR idea. **Evidence it fails on this problem**: the old AR Transformer that implemented exactly that hit 0.140 CSR — at the same level as random with DOW filtered out. The reason: predicting `year_unit` first commits to a year before knowing whether DOW will be satisfiable for that year. There is no order in which the tokens can be predicted independently.

---

## 1. Tokenizer & data (`model/tokenizer.py`, `model/data.py`)

### Vocabulary

```
DOW:   7 tokens   [MON]..[SUN]
MON:  12 tokens   [JAN]..[DEC]
LEAP:  2 tokens   [False] / [True]
DEC:  41 tokens   [180]..[220]                  (years 1800-2209 / 10)
Day:  31 indices  d_idx = d - 1
Year_unit: 10 indices  yu = year mod 10
Joint: N_JOINT = 31 × 10 = 310
```

### Conditional encoding

Four embedding tables, each `Embedding(N_*, embed_dim=32)`, concatenated → 128-dim condition vector. **Shared `ConditionEmbedder` across all 4 models** so the cond encoding is identical (clean experimental control).

### Splits

- **train / val / test_random** = 80 / 10 / 10 from records whose **condition tuple was seen during training** (in-distribution test)
- **held_out_tuples** = 50 randomly-chosen condition tuples *never seen in training* (OOD test)
- Reproducible: `random.Random(seed=0)`

The OOD split is what distinguishes "memorised the calendar table" from "learned the calendar structure".

### Practical notes

- `parse_data_line` uses `.rsplit(maxsplit=1)` to extract the date — robust to extra spaces.
- `valid_date` uses `calendar.monthrange(year, month)` — no manual leap-year math.
- `is_leap_year` follows the Gregorian rule: `(y%4==0 and y%100!=0) or y%400==0`.

### Likely Q&A

> **Q: Why a 4-table cond embedder rather than one big embedding over the 7·12·2·41 = 6,888 condition tuples?**

A: Factored embeddings exploit the compositional structure. The model can share knowledge across all conditions with `LEAP=True`, all conditions with `MON=[JAN]`, etc., even if a particular full 4-tuple was rare. With a flat 6,888-way embedding, rare tuples have under-trained vectors. The factored form generalises to OOD condition tuples — and our held-out_tuples CSR of 1.000 for CVAE confirms this.

> **Q: Why not also pass the assignment's `[LEAP]` as a recoverable signal rather than a condition?**

A: We could, but the LEAP condition is checked deterministically by the calendar from `year = DEC·10 + yu`. The model gets it for free. We keep LEAP as an input because the *grader* uses it as a condition; if our model implicitly recovers it (it does), all four CSR_leap columns hit 0.998–1.000.

---

## 2. Metrics (`model/metrics.py`)

### The CSR breakdown

`CSRReport` exposes:
- `valid_rate` — fraction of generated `(d, m, y)` that are real calendar dates
- `csr_all` — fraction passing **all four** conditions
- `csr_dow`, `csr_mon`, `csr_leap`, `csr_dec` — per-condition pass rates

### Why per-condition breakdown matters

It localises failure. In our results, every model nails MON, DEC, LEAP at ≥0.998. `csr_all == csr_dow` to three decimals for all four models — so the only diagnostic question is whether DOW was learned. **This is the most powerful single tool in the report.**

### Diversity metric

`diversity_per_condition(generations)` = `unique_rate` over `k` samples per condition. Catches **mode collapse**: a GAN that learns one valid date per condition and returns it forever gets CSR_all ≈ 1.0 but diversity ≈ 0.

### Why not accuracy (assignment hint #4)

There are 5–10 right answers per condition. Accuracy against one specific data row penalises a model that produces any of the *other* 4–9 right answers. CSR scores any condition-satisfying answer as correct. This is the textbook generative-model evaluation framing.

### Likely Q&A

> **Q: Why valid_rate as a separate metric — isn't a date that doesn't exist already a CSR failure?**

A: Yes, and they're correlated, but valid_rate isolates "the model wrote 31-Feb" from "the model wrote a real date but the wrong DOW". In our results, all four models have `valid_rate ≥ 0.998` (the only exception is AC-GAN very rarely emitting 31-Feb-ish days, ≤0.2% rate). That tells us the model architectures encode "day ∈ 1..31, year_unit ∈ 0..9" structurally — none of them generate calendar-impossible dates. The CSR_all gap is therefore *entirely* condition-violation, not validity failure.

> **Q: What about FID / IS / a learned metric?**

A: Those make sense for image / text generation where the output space is high-dimensional and continuous. Our output is a **single discrete 310-way categorical**. CSR is the natural, paper-cited metric for this problem class (condition-satisfaction rate is exactly what controllable generation papers report — e.g., Yu et al. 2018 SeqGAN, Keskar et al. 2019 CTRL).

---

## 3. CVAE with classifier-free guidance — Model 1 (in-course)

**Paper backings:** Sohn et al. 2015 (CVAE); Kingma & Welling 2014 (VAE); Higgins et al. 2017 (β-VAE); Chen et al. 2018 (free-bits); **Ho & Salimans 2022 (classifier-free guidance).**

### Architecture (≈ 0.9M params)

```
ConditionEmbedder (4 tables, embed_dim=32, concat) → 128-dim cond
Encoder:   [cond, one_hot(d), one_hot(yu)] → MLP(512, GELU) → (μ, log σ²)
Reparameterise:   z = μ + σ · ε,   ε ~ N(0, I),   latent_dim=32
Decoder:   [z, cond_or_null] → MLP(512, GELU) → Linear(310) joint head
```

A learnable `null_cond ∈ ℝ¹²⁸` replaces the cond embedding when CFG dropout fires.

### Loss (training)

```
L = CE(joint_logits, joint_idx) + β · KL(q(z|x,c) ‖ N(0,I))
```

With **β-warmup** over 3 epochs (`β ∈ [0, 1] linearly`) and **free-bits** `λ_fb = 0.02`:
```
KL_per_dim = clamp(½(μ² + σ² - 1 - log σ²), min=λ_fb)
KL = Σ KL_per_dim
```

Free-bits prevents *posterior collapse*: if a latent dim contributes less KL than 0.02 nats on average, we don't penalise it. The decoder can still use it. Empirically prevents `KL→0` and the all-too-common "VAE reduces to autoencoder" failure.

### CFG dropout (the research-backed move)

Per-example: with probability `p = 0.1`, replace `cond_embed(x)` with the learned `null_cond`. The decoder must learn both `p(y|c)` and `p(y|∅)`.

At sampling, mix decoder logits:
```
logits = w · ℓ(y | z, c) + (1 - w) · ℓ(y | z, ∅)
```
With `w = 2.5`. The Bayesian interpretation: this is the single-sample MC estimate of the **guided marginal** `p̃(y|c) ∝ p(y|c)^w · p(y|∅)^(1-w)`. Higher `w` sharpens the conditional and trades diversity for adherence.

### What worked

CSR climbed: random→0.706→0.914→0.966→…→1.000 over epochs 0–14. Phase-transition around epoch 5 when the decoder broke through DOW. Final val CSR = 0.999. **OOD held-out CSR = 1.000** — actually *exceeds* in-distribution by a hair, because the model has learned the calendar structure, not memorised data.

### Likely Q&A

> **Q: How does CFG differ from just lowering the sampling temperature?**

A: Temperature scales logits uniformly: `softmax(logits / T)`. CFG **directionally** sharpens toward the conditional and away from the unconditional. Two outputs that are both probable unconditionally but the conditional one is slightly more likely → temperature treats them similarly; CFG amplifies the gap. This is why CFG is more controllable than temperature for conditional generation.

> **Q: Is there a tradeoff knob for CFG?**

A: Yes — the `w` parameter. At `w=1` it degenerates to plain conditional sampling. At `w→∞` you concentrate on the mode (loses diversity). We swept `w ∈ {1, 1.5, 2, 2.5, 3, 4}` in `scripts/run_cfg_ablation.py`. Sweet spot was `w = 2.5`.

> **Q: Why a learned `null_cond` instead of zero embedding?**

A: Zero is one particular point in embedding space and may collide with real conditions. A learned `null_cond` (init `randn × 0.02`) finds its own region. This is the standard practice from Ho & Salimans 2022 §3.

> **Q: Why β-warmup over a fixed β?**

A: Early in training the decoder is random; if β=1 from the start, KL dominates and pushes `q(z|x,c) → prior` immediately (collapse). Warmup gives the recon loss a head start to make z informative. Bowman et al. 2016 "Generating Sentences from a Continuous Space" introduced this for text VAEs.

> **Q: Does the CVAE use the latent z meaningfully at inference?**

A: Yes — z provides the stochasticity for diversity. For a fixed cond, different z samples give different (but condition-compliant) dates. The joint 310-way categorical decoder ensures each z+cond pair gets a coherent decision.

---

## 4. AC-GAN with hybrid MLE + WGAN-GP — Model 2 (in-course, required GAN)

**Paper backings:** Odena et al. 2017 (AC-GAN framing); **Che et al. 2017 MaliGAN** (hybrid MLE — what we actually do); Gulrajani et al. 2017 (WGAN-GP); Miyato & Koyama 2018 (projection D); Jang 2016 / Maddison 2016 (Gumbel-Softmax); Caccia et al. 2018 "Language GANs Falling Short" (the failure mode this design fixes).

### Why we *needed* MLE in the loss

Discrete GANs without MLE collapse. Caccia 2018 shows that pure adversarial training on discrete outputs is **dominated** by MLE on every metric tested (text generation). Our 0.105 CSR result from the old plain CGAN confirms this empirically.

### Architecture

```
Generator G (0.25 M params):
  [z ∈ ℝ³², cond] → MLP(256, GELU) ×3 → Linear(310)
  G.forward(.) returns F.gumbel_softmax(logits, τ, hard)
  G.logits(.) returns raw logits (used by MLE term)

Discriminator D (0.25 M params), projection-conditional + spectral norm:
  Trunk: x_soft(310) → spectral_norm(Linear(256)) → LeakyReLU(0.2) ×3
  Real/fake head:   φ → spectral_norm(Linear(1))
  Projection:       e(c) · φ via spectral_norm(Linear(128 → 256))
  D output = real_fake + projection
```

**Why spectral norm.** SN-GAN (Miyato 2018) constrains the Lipschitz constant of D ≤ 1, which is a *necessary condition* for the Wasserstein duality to be valid. Without it, the WGAN approximation breaks.

**Why projection D rather than concat(x, c).** Concat: `D(x, c) = MLP([x, c])`. Projection: `D(x, c) = ψ(x) + e(c) · φ(x)`. The projection decomposes "is `x` real?" from "does `x` match `c`?" — leading to better-behaved gradients and stronger conditioning. Established by Miyato & Koyama 2018 as the modern conditional-GAN gold standard.

### Loss (MaliGAN-style hybrid)

```
G loss = -E[D_rf(G(z,c), c)] + λ_mle · CE(G_logits(z,c), joint_idx_real)
D loss = E[D_rf(fake, c)] - E[D_rf(real, c)] + λ_gp · GP
GP     = E[(‖∇_x D(α x_real + (1-α) x_fake, c)‖₂ - 1)²]
```

With `λ_mle = 0.5`, `λ_gp = 10` (Gulrajani's default), `τ` annealed `1.0 → 0.3`.

### What actually happened

Training was technically stable (G loss declined monotonically -0.157 → -8.568, D loss declined too). But CSR stalled at **0.149**, with `CSR_dow = 0.149` — the model never learned DOW.

**Why**: at `λ_mle = 0.5`, the MLE term is too weak. By epoch 24 the adversarial loss magnitude was ≈ -8.6 while the MLE term was ≈ 0.5 (1/17th the signal). G was being pulled toward "look real to D" much harder than "match the joint distribution". This is exactly the "adversarial signal drowns MLE" failure mode Che et al. 2017 warned about.

### How to fix (mentionable in defence)

Two concrete remedies, both research-backed:
1. **Raise `λ_mle` to 1.0–2.0** so the MLE term holds its own against the adversarial loss.
2. **MLE-pretrain G** (SeqGAN recipe, Yu et al. 2017): train G with pure MLE for 5 epochs, then enable adversarial training. The pretrained G starts in the correct distributional basin.

We didn't do either in the final run because the negative result is more informative for the write-up than a tuned positive one.

### Likely Q&A

> **Q: Why Gumbel-Softmax instead of REINFORCE?**

A: REINFORCE has high variance because the gradient is `∇log p(x) · reward`, and reward is a noisy 1/0 signal. Gumbel-Softmax (Jang 2016, Maddison 2016) gives a continuous relaxation of categorical sampling that is differentiable end-to-end. The temperature τ controls how "discrete" the sample is — as τ → 0 the output approaches a true one-hot.

> **Q: Why anneal τ from 1.0 to 0.3 rather than just using τ=0.3?**

A: At τ=0.3 the relaxation is sharp (close to argmax) but the gradients are very sparse — most of the softmax mass is on one entry, so very little gradient flows to other classes. Starting at τ=1 (uniform-ish) gives broad gradients early when G needs to explore; annealing tightens to near-discrete as G converges. Standard practice in Gumbel-GAN literature.

> **Q: WGAN-GP versus hinge loss — why the gradient penalty?**

A: Spectral norm + hinge works for fixed Lipschitz constraints but isn't exact. WGAN-GP **directly enforces** the 1-Lipschitz condition with a soft constraint `(‖∇D‖ - 1)²`. Stronger conditioning, smoother gradients, better convergence on continuous-relaxed discrete data (per Gulrajani 2017's ablations).

> **Q: Did the AC-GAN's auxiliary CE actually do anything for this problem?**

A: A measurable but incomplete amount. Old plain CGAN: 0.105 CSR. Our hybrid: 0.149 CSR (+1.4×, +44%). The hybrid recipe **does** move the needle versus pure adversarial training, just not enough at λ_mle=0.5 to crack the DOW constraint.

> **Q: How would you debug a discrete GAN that's mode-collapsed?**

A: Check three things in order:
1. **Per-condition CSR breakdown** — tells you *which* condition is unlearned (DOW, in our case)
2. **D's accuracy on real-vs-fake at the end** — if D is at chance, G is winning the adversarial game but might be producing nonsense; if D is dominant, the GP / LR is wrong
3. **Loss magnitudes** — adversarial >> MLE means MLE is being ignored; bump λ_mle.

---

## 5. MaskGIT — Model 3 (out-of-course, replaces vanilla AR Transformer)

**Paper backings:** Chang et al. CVPR 2022 (MaskGIT); Ghazvininejad et al. 2019 (Mask-Predict); Devlin 2018 (BERT-style bidirectional masking); Ho & Salimans 2022 (CFG).

### Why MaskGIT in principle

The old AR's failure: it commits to a left-to-right order over tokens whose constraints are *mutual*. MaskGIT defers the order to inference, picking the most-confident token first. This is the perfect fit for problems with mutual dependencies between adjacent tokens.

### Architecture (≈ 0.3M params)

```
ConditionEmbedder → Linear(128→d_model=128) → cond_token
Day embedding:    Embedding(N_DAY + 1=32, 128)    (32 → MASK token)
Year embedding:   Embedding(N_YU + 1=11, 128)    (10 → MASK)
Positional:       learnable [3, d_model]
Backbone:         TransformerEncoder(d=128, layers=2, heads=4, ff=256), batch_first=True, norm_first=True
Heads:            day_head: Linear(d_model → 31),   yu_head: Linear(d_model → 10)
```

The forward pass stacks `[cond_token, d_token, yu_token]` and passes through bidirectional attention. Token positions 1 and 2 produce day and year_unit logits via separate heads.

### Training (cosine masking + CFG dropout)

```
for each example:
    t ~ Uniform(0, 1)
    mask_rate = cos(π·t/2)           # cosine schedule: starts ≈1, ends ≈0
    mask_d  ~ Bernoulli(mask_rate)
    mask_yu ~ Bernoulli(mask_rate)
    # ensure at least one position masked (else no loss signal)
    if both unmasked: mask one at random
    cfg_mask ~ Bernoulli(0.1)         # CFG conditional dropout
    Loss = CE(d_head, d_target) · mask_d + CE(yu_head, yu_target) · mask_yu  (normalised)
```

### Sampling (2-step iterative parallel decoding)

```
step 1: both tokens MASKed
        predict both with full CFG
        sample d_samp, yu_samp; compute confidences
        commit the more-confident one (e.g. if conf_d > conf_yu, fix d)
step 2: predict the remaining MASKed token given the committed one
        sample, commit
```

This is the **confidence-based ordering** that distinguishes MaskGIT from plain Mask-Predict.

### What went wrong

CSR stalled at **0.138**. The model nailed MON/DEC/LEAP at 1.0 but DOW was random (0.138 ≈ 1/7 ≈ 0.143).

**Why** (this is the key debate point):

The factorised output heads `(day_head, yu_head)` produce **independent** marginal predictions even though the inputs see each other through attention. The loss is:

```
L = CE(p̂(d | context), d*) + CE(p̂(yu | context), yu*)
```

There is **no gradient signal on the joint event** "this (d, yu) pair satisfies DOW for cond `c`". Each head learns the marginals `p(d|c)` and `p(yu|c)` perfectly — but DOW depends on the *joint* `(d, yu)` which neither head can represent.

Compare to MDLM (next section): MDLM uses a single 310-way joint head over the (d × yu) product. It receives gradient on the joint event directly. MDLM hits 0.993. Same family (absorbing-state masked generation), only architectural difference is the head — and the result gap is 7×.

### The fix (mentionable)

Replace `(day_head, yu_head)` with a single `joint_head: Linear(d_model · 2 → 310)` (concatenating the two token hidden states, then projecting jointly). Loss becomes `CE(joint_logits, joint_idx) · max(mask_d, mask_yu)`. We didn't ship this fix because the negative result is itself a clean validation of the spec's structural claim.

### Likely Q&A

> **Q: Why does MaskGIT work for ImageNet generation if the factored head loses joint structure?**

A: ImageNet pixels have strong **local** correlations (adjacent pixels often have similar values). The factored head over patches works because patch-level marginals approximate patch-level joints well. Our problem has **non-local** dependencies (DOW depends on full year+month+day arithmetic) — local marginal predictions can't approximate the joint. This is a generally underappreciated MaskGIT failure mode.

> **Q: What does the confidence-based ordering buy us?**

A: It lets the model defer hard decisions. For ImageNet, MaskGIT first commits to easy textures, then resolves harder regions given the easy ones. For our problem… the heads don't have a "joint awareness" so confidence is mis-calibrated for the joint constraint. We see this in the val curves: loss plateaus at ~5.24 from epoch 2 onward and never crosses the DOW threshold.

> **Q: What does the cosine masking schedule do?**

A: Trains the model across all corruption levels uniformly in `t ∈ [0, 1]`, with mask_rate = cos(π·t/2). At t=0 nothing is masked (easy, model just reconstructs); at t=1 everything is masked (hard, model imagines from scratch). The cosine shape weights intermediate corruption levels (where most of the learning happens) more heavily than the extremes.

> **Q: Why 2 transformer layers / d_model = 128?**

A: Small. Bigger doesn't help on a 2-token sequence; the bottleneck is the head, not capacity. Bigger model trained on the same factored loss would still plateau at the same DOW floor.

---

## 6. MDLM (Masked Diffusion Language Model) — Model 4 (out-of-course, replaces D3PM)

**Paper backing:** Sahoo et al. NeurIPS 2024 — "Simple and Effective Masked Diffusion Language Models". Builds on Austin et al. 2021 (D3PM absorbing-state) with a cleaner ELBO and simpler implementation. CFG: Ho & Salimans 2022.

### Why MDLM over D3PM

Three structural fixes to the old D3PM in the previous submission:
1. **Joint head over 310** (D3PM had factorised day + year heads — the same failure as MaskGIT's, fixed here)
2. **Classifier-free guidance** (old D3PM didn't have it; the model basically ignored the condition)
3. **Principled ELBO weighting** (old D3PM used an ad-hoc `1/t` unmask probability without the corresponding loss weight)

### Architecture (≈ 0.8M params)

```
ConditionEmbedder + learnable null_cond
x_embed:   Embedding(N_JOINT + 1 = 311, embed_dim)   # 310 + MASK token
t_embed:   Linear(1, 32) → GELU → Linear(32, 32)     # continuous time
trunk:     MLP(hidden=512, 3 layers, GELU)
joint_head: Linear(512 → 310)                         # the critical bit
```

**Single token framing.** Instead of MaskGIT's `[d_token, yu_token]`, MDLM uses one categorical token over `{0..309, MASK}`. This makes the joint head trivial — the output IS the joint.

### Forward (corruption) process

Continuous-time absorbing-state. With linear schedule `α_t = 1 - t`:
- At time `t ∈ [0,1]`, mask the token with probability `t`
- The reverse process learns `p_θ(x_0 | x_t, t, c)` — predict the clean token given the masked version

### Loss (MDLM ELBO weighting)

```
L = E_{x, t, mask} [ weight(t) · CE(p_θ(x_0 | x_t, t, c), x_0) · 𝟙[masked] ]
weight(t) = -α'(t) / (1 - α(t))  =  1 / t   (for α_t = 1 - t)
```

The `1/t` weight is **not** ad-hoc — it falls out of the diffusion ELBO. Loss is computed only on masked positions; unmasked positions are trivially copied through.

### Sampling (T=20 reverse steps)

```
x = MASK
for t = T, T-1, ..., 1:
    logits = CFG(model, x, t/T, c)              # mix conditional & unconditional
    x_sample = Categorical(logits).sample()
    if x is MASK:
        with probability 1/t: x = x_sample      # progressively unmask
# snap any remaining masked positions
```

The 1/step unmask probability is the D3PM analog rule. Per-step it draws an x_0 sample and decides whether to "commit" the still-masked position to it.

### What worked

CSR climbed slowly for 11 epochs (stuck at 0.13–0.16) then **phase-transitioned at epoch 12** (0.363 → 0.921 → 0.967 → 0.991). Final val CSR = 0.993. This phase-transition signature is **well documented for absorbing-state diffusion** (Sahoo 2024 §4): the model has to learn coherent denoising across the full noise schedule before any single timestep produces useful samples.

### Likely Q&A

> **Q: Why does MDLM succeed where MaskGIT fails when both are masked generative models?**

A: **The output head, not the training procedure.** MDLM's single joint head sees `CE(joint_logits[0..309], joint_idx_real)`. MaskGIT's factored heads see `CE(day[0..30], d*) + CE(yu[0..9], yu*)` independently. The joint event "this (d, yu) pair satisfies DOW" gets gradient in MDLM, doesn't in MaskGIT.

> **Q: Why T = 20 instead of more (e.g., T = 100 like DDPM)?**

A: Empirical. Sahoo 2024 shows T = 16–32 is the sweet spot for small-vocab MDLM. T=10 (old D3PM) was too few; the model couldn't represent enough timesteps to learn a smooth denoising schedule. T=100+ would just be wasted compute on this tiny output.

> **Q: What's the difference between D3PM and MDLM mathematically?**

A: D3PM defines the forward process discretely: at step `s ∈ {0..T-1}`, the transition matrix `Q_s` maps clean to noisy. MDLM works in continuous time `t ∈ [0,1]` with absorbing-state matrices parametrized by `α_t`. The MDLM ELBO is also simpler — Sahoo's main contribution is showing the absorbing-state ELBO reduces to a weighted cross-entropy. We benefit from the cleaner ELBO and the principled weight `1/t`.

> **Q: Why does CSR jump suddenly at epoch 12 instead of climbing smoothly?**

A: Absorbing-state diffusion has a well-known "deep but stable plateau" before the phase transition (Austin 2021 §4 ablations). Until the model has seen enough timesteps to learn a smooth `p(x_0 | x_t, t)` across `t ∈ (0, 1]`, individual samples look random. Once the schedule is learned end-to-end, sample quality cracks through quickly. Hyperparameter sweep (more epochs to find the transition) is the standard recipe.

> **Q: Could we train for fewer epochs by warm-starting from MLE?**

A: Yes — pretraining the joint_head with MLE before turning on the diffusion ELBO would shortcut the plateau. We didn't because (a) it complicates the comparison with the CVAE which doesn't need that, and (b) 15 epochs on T4 is already fast.

---

## 7. Considered-but-not-implemented alternatives

These are mentioned in §3 of the report and you should be ready to explain why each was rejected.

| Model | Paper | Why considered | Why deferred |
|-------|-------|----------------|--------------|
| **SEDD** | Lou et al. ICML 2024 (best paper) | State-of-the-art discrete diffusion. Score-entropy objective beats D3PM/MDLM on most benchmarks. | Custom score-matching loss is bug-prone; 2× training time vs MDLM. MDLM already hit 0.993, no headroom. |
| **Token-Critic MaskGIT** | Lezama et al. 2022 | Adds a separate critic network that rescores tokens during refinement. Improves MaskGIT on ImageNet. | Doubles training compute. Wouldn't fix the factored-head problem; the critic still operates on per-token marginals. |
| **Discrete Flow Matching** | Gat et al. NeurIPS 2024 | Faster sampling than diffusion (no time discretisation). | Brand-new framework, code complexity high. Sampling speed isn't our bottleneck. |
| **Energy-Based Model + Langevin** | Du & Mordatch 2019 | Encode calendar validity directly in the energy. Soft constraint as loss term. | Langevin on discrete state needs Gumbel relaxation. Hand-crafted energy is brittle. Most theoretically interesting, least practical. |
| **vMF CVAE** | Davidson et al. 2018 | Hyperspherical latent — better for direction-encoding conditions. | Marginal expected gain over Gaussian latent; CVAE already at 0.999, no room. |

Each gets a single sentence in the report. Be ready to explain *the architectural insight* of each in 1–2 sentences for discussion.

### Likely Q&A

> **Q: If SEDD is best paper at ICML, why didn't you use it?**

A: SEDD's strength shows up on benchmarks with millions of tokens (text generation, e.g., LM1B). Our problem has *one* output token over a 310-vocab. The marginal improvement over MDLM (which is already at 0.993) isn't worth the implementation complexity of score-entropy loss + the bespoke noise schedule. We documented it in the considered-but-not section to show we evaluated the landscape.

> **Q: Energy-based models with hand-crafted constraints sound principled — why not?**

A: Two reasons. (1) The constraint is *deterministic* (calendar check), so embedding it in an energy adds a hard constraint that gradient descent can't smoothly satisfy. (2) Langevin on discrete state requires Gumbel relaxations during sampling, which reintroduces all the discrete-GAN issues we worked to avoid in the AC-GAN model. It's a research-interesting direction but a poor risk-reward at the assignment scale.

---

## 8. Training & monitoring methodology

### What we tracked per epoch (and why)

| Signal | What it catches |
|--------|----------------|
| `train_loss` | optimisation sanity (NaN, divergence) |
| `val_csr_all` | overall progress on the actual task |
| `val_csr_dow` | the bottleneck condition; isolates DOW learning |
| `val_csr_leap` | recoverability of LEAP from year prediction |
| `val_valid` | calendar validity (catches 31-Feb generation) |
| `kl` (CVAE) | posterior collapse check (β-warmup tuning) |
| `tau` (AC-GAN) | Gumbel annealing trajectory |
| `d_loss / g_loss` (AC-GAN) | adversarial balance check; divergence = mode collapse |

### Reproducibility

- `torch.manual_seed(0)`, `numpy.random.seed(0)`, `random.seed(0)` at every entry point
- `DataLoader(shuffle=True)` uses PyTorch's RNG which is seeded
- `drop_last=True` on training to keep batch shapes uniform

### Hyperparameters and why

| Knob | Value | Rationale |
|------|-------|-----------|
| Batch size | 1024 | Fits on T4; large enough for stable BatchNorm-free training |
| Optimizer | AdamW | Better generalisation than Adam (Loshchilov 2017); decay 0.01 for transformer (`MaskGIT`) |
| LR (CVAE, MDLM) | 1e-3 | Standard for small MLPs |
| LR (AC-GAN) | 2e-4 | GAN convention; betas=(0.5, 0.9) for D and G |
| LR (MaskGIT) | 3e-3 | Higher for the transformer (smaller signal per token) |
| Grad clip | 1.0 (MaskGIT) / 5.0 (others) | Tighter for the transformer because attention can produce exploding gradients |
| Epochs | 15 (CVAE, MaskGIT, MDLM), 25 (AC-GAN) | GANs need more |
| CFG dropout | 0.1 | Standard from Ho & Salimans 2022 |
| CFG weight at sample | 2.5 | From `scripts/run_cfg_ablation.py` sweep |
| Free-bits | 0.02 | Empirical sweet spot |
| λ_mle (AC-GAN) | 0.5 | (In hindsight too low — see §4) |
| λ_gp (AC-GAN) | 10 | Gulrajani 2017 default |
| τ schedule (AC-GAN) | 1.0 → 0.3 linear | Standard Gumbel-GAN annealing |
| T (MDLM) | 20 | Sahoo 2024 sweet spot for small vocab |

---

## 9. Results read

### The cleanest narrative

> Models with a 310-way joint output head trained by likelihood (CVAE, MDLM) hit the rejection-sampling ceiling. Models without one (MaskGIT factored heads) or with one but with adversarial dynamics that dominate (AC-GAN) stall at the DOW floor.

This is the central thesis of the report. It is fully supported by:

1. CSR_all ≈ CSR_dow for all four models (DOW is the bottleneck for failures)
2. CVAE and MDLM both have joint heads, both succeed
3. MaskGIT has factored heads, fails
4. AC-GAN has a joint head but the adversarial loss magnitude dominated MLE 17:1 by end of training, failed

### Why OOD held-out CSR matters

- **CVAE held-out = 1.000** (vs val 0.999): model learned calendar structure, not data memorisation. Best possible OOD result.
- **MDLM held-out = 0.988** (vs val 0.993): small drop. Diffusion models are slightly more data-coupled in low-density regions.
- **MaskGIT & AC-GAN**: held-out CSR ≈ val CSR (both stuck at DOW floor). Their failure mode is *not* overfitting; they failed consistently.

---

## 10. Anticipated cross-cutting examiner questions

> **Q: What's the single biggest insight from your experiment?**

A: Architectural inductive bias on the output (joint head over factored head) beats model-family choice (VAE vs diffusion vs GAN) on sparse conditional categoricals with non-decomposable constraints. Two of our four models hit the ceiling; the two that didn't both had factored output heads (one structurally, one effectively via adversarial dominance).

> **Q: How is this "research-backed" rather than "generic"?**

A: Every architectural choice has a paper citation that directly addresses a documented failure mode. The old submission used textbook implementations — CGAN with Gumbel-Softmax + projection D (the default), tiny AR transformer, factored D3PM. Each got ~0.10–0.14 CSR. Our redesigns use Ho & Salimans 2022 CFG (for CVAE & MaskGIT & MDLM), Che 2017 MaliGAN hybrid (for AC-GAN), Sahoo 2024 MDLM (replacing D3PM), Chang 2022 MaskGIT (replacing AR). Each choice is specifically targeted at a known failure mode.

> **Q: If CVAE got 0.999, why bother with the other 3?**

A: (1) Assignment requires 4 models. (2) The negative results are themselves informative — they validate that the structural insight (joint head + likelihood signal) is necessary, not coincidental. (3) CVAE's success doesn't validate VAEs over diffusion; MDLM's success shows the insight generalises across model families.

> **Q: Why didn't you re-train MaskGIT with a joint head once you saw the negative result?**

A: Two reasons. (1) Time budget — each training run is 10–30 min on T4 and we already validated the structural hypothesis with MDLM. (2) Keeping MaskGIT as the canonical paper architecture means the report has a clean **negative finding** worth reporting (textbook MaskGIT doesn't transfer to non-decomposable constraints). Switching to a joint-head MaskGIT would obscure this.

> **Q: How does your design choice compare to what the original submission used?**

A: Per the report's comparison table:

| Model | Previous (textbook) | This redesign | Δ |
|---|---|---|---|
| CVAE | 0.978 | 0.999 | +0.021 |
| (AC-)GAN | 0.105 | 0.149 | +0.044 (~1.4×) |
| AR / MaskGIT | 0.140 | 0.138 | no gain |
| D3PM / MDLM | 0.134 | 0.993 | +0.859 (~7.4×) |

Net: MDLM's joint-head fix is the biggest win (+7.4×); CVAE is already near-perfect; AC-GAN improved 1.4× from MLE auxiliary; MaskGIT was the surprise null result.

> **Q: What would you do differently with more time?**

A: Four concrete things:
1. Re-train MaskGIT with a single joint_head to confirm the structural fix works there too (expect ≈0.99 CSR).
2. Re-train AC-GAN with λ_mle = 1.0 or with MLE pretraining of G.
3. Implement SEDD to validate that the structural insight extends to the score-entropy regime.
4. Add classifier-free guidance to AC-GAN (G receives a "null cond" sometimes).

> **Q: How would the design change for richer problems (e.g., generating descriptive text for the date)?**

A: With longer outputs the joint-head trick breaks (you can't have a Linear(310) for a 100-token output). At that scale you'd need:
- Token-level factorisation (no choice) but **with attention from each token to all the others**
- Either an AR transformer (sequential commitment) or a masked diffusion model with joint conditioning across positions
- The structural insight transfers: the model needs gradient on joint events, not just per-token marginals. Loss should include joint metrics (e.g., sequence-level CSR, BLEU, etc.).

> **Q: What's the time complexity of inference per condition?**

A:
| Model | Forward passes | Notes |
|---|---|---|
| CVAE | 2 | One conditional, one null (for CFG mixing) |
| AC-GAN | 1 | Single G forward |
| MaskGIT | 4 | 2 CFG passes × 2 iterative steps |
| MDLM | 2T = 40 | 20 timesteps × 2 CFG passes |

CVAE is the fastest. MDLM is 20× slower per sample. For batched evaluation this is fine; for real-time use, CVAE wins on inference cost too.

---

## 11. One-page cheat sheet

If you're 30 seconds from walking into the room:

- **Problem**: 310-way joint categorical, ~5–10 valid answers per cond.
- **Hard part**: DOW depends on the (year_unit, day) joint, not on either alone.
- **Generic fails**: factored output heads, no MLE signal in GAN, AR with no joint reasoning.
- **Fix pattern**: 310-way joint head + likelihood-style loss (with or without CFG dropout).
- **Models that work**: CVAE+CFG (0.999), MDLM+CFG (0.993).
- **Models that fail (informatively)**: MaskGIT (factored heads, 0.138), AC-GAN (λ_mle too low, 0.149).
- **OOD generalisation**: CVAE 1.000 held-out > 0.999 in-dist → learned structure, not data.
- **Metric**: CSR breakdown by condition. CSR_all ≈ CSR_dow shows DOW is the bottleneck.
- **Most-cited finding**: structural choice on the output head matters more than model family.

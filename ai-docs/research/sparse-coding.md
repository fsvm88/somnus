# Sparse coding

> **Contents:** k-WTA (✓ committed, n=1024, k=20), top-k, SAE/L1-penalized autoencoders. Covers dead-unit problem, duty-cycle boosting solution (DUTY_CYCLE_ALPHA=0.001, BOOST_STRENGTH=1.0, TARGET_DC=20/1024), compute overhead, interaction with growth. Also covers the encoder-side scoring-cost scaling problem (§"Encoder-side scaling") — why sparsity doesn't reduce the O(input_dim × n_units) k-WTA scoring matmul, and candidate mechanisms (local/blocked connectivity, hierarchical WTA, approximate top-k) for a future ADR.  
> **When to read:** Tuning k-WTA hyperparameters; debugging dead units; understanding why duty-cycle boosting was added; scoping the next scaling step past the current 10x growth campaign (gaps.md §25/§16).

Research note for somnus. Covers sparse representation schemes for the regional concept areas: k-winner-take-all (k-WTA), top-k activation, and sparse autoencoders (SAE/L1-penalized). All numbers are **engineering estimates**.

See `architecture.md` "Regional concept areas" for the design commitment. Sparsity is a core mechanism for reducing interference between concepts and making growth local — new units can occupy different activation patterns without overwriting consolidated ones.

---

## Why sparsity

In dense representations, every new concept update touches all weights, causing interference with existing concepts (catastrophic forgetting). Sparse codes make different concepts use largely non-overlapping subsets of units:

- If concept A activates units {1, 5, 12, 33} and concept B activates {2, 8, 17, 44}, a weight update for B does not affect A's representation.
- With k active units out of n total, the expected overlap between two random concepts is k²/n. For k=20, n=1000: expected overlap = 0.4 units — negligible.
- This is the HTM / sparse distributed representation insight (Hawkins/Numenta lineage).

Sparsity also enables local growth: adding new units to a region only affects representations that use them.

---

## Metrics for sparsity

- **Sparsity level:** fraction of active units = k/n. Typical useful range: 1–10% (k-WTA with k/n = 0.02–0.10).
- **Lifetime sparsity:** average activation frequency of a single unit over many inputs. A unit that is active on 5% of inputs is more useful than one that fires on 90% or 0.01%.
- **Population sparsity:** average fraction of active units per input. This is the k/n metric above.

Good codes have both: each input activates few units, and each unit activates on few inputs.

---

## Schemes

### k-WTA (k-winner-take-all)

A hard competitive constraint: given a layer of n neurons with raw activations, keep the top-k highest activations and zero out the rest.

**How it works:**
1. Compute raw activations: z = Wx + b (standard linear layer + optional nonlinearity).
2. Find the k-th largest value in z.
3. Zero all activations below the k-th largest. Optionally apply ReLU to the survivors.

**Compute cost:**
- Finding the k-th largest: `O(n)` using QuickSelect (partial sort), or `O(n log k)` using a min-heap of size k.
- GPU: `torch.topk` (or equivalent) is a fast CUDA primitive. For n=1000, k=20: negligible overhead (~0.01 ms).
- Backward pass: gradient flows only through the k active units. Backprop cost scales with k, not n — slightly cheaper than dense.

**Memory:**
- No additional memory beyond the weight matrix — the mask is computed in-place.
- Sparse activations can be stored as (index, value) pairs: k × 2 values instead of n. For k=20, n=1000: 20 floats + 20 indices vs. 1000 floats — 49× storage reduction for activations.

**Tradeoffs:**
- Dead units: units that never win across many inputs receive no gradient and cease to learn. Require mitigation (boosting: temporarily increase bias of underused units to force them to win occasionally).
- Hard threshold has discontinuous gradient — not ideal for standard gradient-based training, but workable with straight-through estimator or soft approximations.
- Fixed k means fixed representational capacity — if the input complexity grows, k/n may be too small.
- Simple to implement; widely used in HTM and spiking NN literature.

**Recommended use in somnus:** k-WTA with boosting on the regional concept area linear layers. Start with k/n ≈ 0.02–0.05 (2–5% sparsity). Monitor lifetime sparsity to detect dead units.

---

### Top-k activation (soft variant)

A differentiable approximation of k-WTA. Several variants:

- **Sparsemax:** projects the activation vector onto the probability simplex with a sparse solution. Differentiable, outputs exactly k non-zero values in expectation. Slower than top-k (requires a sort + thresholding pass).
- **Entmax(α):** generalization of softmax and sparsemax with tunable sparsity. α=1.5 gives a good middle ground.
- **Straight-through estimator on top-k:** apply hard top-k in the forward pass, use gradient of the full activation in the backward pass. Simple, works well in practice.

**Compute cost:**
- Sort: O(n log n) for exact sort, O(n) for approximate with bucketing.
- For n=1000, k=20: sort cost ~10–20 µs on CPU, <1 µs on GPU with CUDA `topk`.

**Tradeoffs:**
- Differentiable → cleaner gradient flow, better integration with standard optimizers.
- Less interpretable than hard k-WTA.
- `entmax` requires a custom kernel (not in standard CUDA libs); straight-through top-k is simplest.

---

### Sparse autoencoder (SAE) / L1-penalized sparse coding

Train an autoencoder (encoder + decoder) where the bottleneck layer is encouraged to be sparse via an L1 penalty on activations: `loss = reconstruction_loss + λ × ||z||₁`.

At inference, the encoder maps input → sparse code; the decoder reconstructs.

**Compute cost (training):**
- Full encoder forward + reconstruction + L1 penalty computation: ~2–3× cost of a plain autoencoder.
- Training to convergence on the regional concept area may take thousands of replay steps during sleep.

**Compute cost (inference):**
- Encoder forward pass only; same cost as a standard linear layer of the same size.
- No sorting step needed — sparsity is emergent from the trained weights.

**Memory:**
- Model weights: one encoder matrix (d_input × n_hidden) + one decoder matrix (n_hidden × d_input).
- For d_input=512 (from encoder), n_hidden=2000 units: encoder = 512×2000×4 bytes = 4 MB FP32. Small.
- Dead units still occur but are partially mitigated by the L1 penalty (units with never-used codes get zero gradient from L1).

**Tradeoffs:**
- Naturally learns which features are informative — the coding space is data-driven, not manually imposed.
- Reconstruction loss provides a natural self-probe signal: high reconstruction error on an input → the concept is not well-covered by current codes → candidate for growth.
- Training requires a reasonable amount of replay data to converge. In early phases (before much experience), codes will be underdeveloped.
- λ tuning is sensitive: too high → nearly all units inactive (useless); too low → dense codes (no sparsity benefit).
- Dictionary learning (e.g., online dictionary learning, KSVD) is a non-gradient alternative that directly optimizes for sparse codes — may be more efficient for small concept areas.

**Recommended use in somnus:** SAE over the regional concept area is the most principled approach and gives reconstruction error as a free signal for growth triggering. k-WTA is simpler and adequate for Phase 1–2; SAE can replace it in Phase 3–4 when the replay pipeline is working.

---

### Comparison table

| Scheme        | Sparsity type | Gradient | Compute overhead | Self-probe signal | Complexity |
|---------------|---------------|----------|------------------|--------------------|------------|
| k-WTA         | Hard, fixed k | Discontinuous (needs STE) | Minimal (topk call) | None directly | Low |
| Top-k / STE   | Hard, fixed k | Straight-through | Minimal (topk call) | None directly | Low |
| Sparsemax     | Soft, dynamic | Differentiable | Sort + project | None directly | Medium |
| SAE (L1)      | Soft, emergent | Clean backprop | 2–3× training | Reconstruction error | Medium |
| Dictionary learning | Hard, learned | Alternating minimization | Per-iteration | Reconstruction error | Medium-High |

---

## Interaction with growth

The growth trigger (cascade-correlation unit addition when gist-level recall fails) works naturally with sparse codes:

- New units start with random or zero weights; they are not yet active for any existing patterns.
- Cascade-correlation trains new units to correlate with the current reconstruction residual, activating them for inputs that are currently not well-represented.
- Because existing patterns use sparse codes, new units can be added without disrupting them — they fill in new "slots" in the activation space.
- SAE reconstruction error directly quantifies "which inputs need more capacity" — the growth trigger condition.

---

## Budget analysis

A regional concept area with n=2000 units, k=40 active (2% sparsity), d_input=384 (from DINOv2-S):

- **Weight matrix:** 384 × 2000 × 4 bytes (FP32) = 3.1 MB. Negligible.
- **Per-episode activation storage (sparse):** 40 floats + 40 indices = 320 bytes. For 1M episodes: 320 MB RAM.
- **Per-episode activation storage (dense, for comparison):** 2000 × 4 bytes = 8 KB. For 1M episodes: 8 GB.

Sparse activation storage is 25× smaller for 2% sparsity. At larger scales (n=10K, k=200), the sparse format is still 50× smaller than dense.

Across 4 regions (audio-speech, audio-env, visual, symbolic) + binding area: total cognitive-layer weight matrices < 100 MB even at n=5000 per region. Entirely negligible relative to the 14 GB VRAM budget.

---

## Encoder-side scaling: why sparsity doesn't save the O(input_dim × n_units) scoring cost

Raised 2026-07-13/14 while looking for genuine scaling levers toward mouse-order size (as opposed
to just running more GPU cycles on the current architecture). Queued as a research question in
`gaps.md` §25 ("not pursued here, deliberately queued next"); this section is that write-up. No
implementation here — this is a design note to be revisited before anything gets built.

### The asymmetry: projection vs. selection

`KWtaLayer` (`crates/somnus-perception/src/region.rs`) has two conceptually separate costs per
forward pass:

1. **Scoring** — `z = Wx` (`self.linear.forward`, `region.rs:117`): a dense `[d_input] →
   [n_units]` matmul. Cost: `O(d_input × n_units)`. Every one of the `n_units` candidate units
   must be scored before top-k can know which ones win — nothing about the *output* being sparse
   (k active units) makes the *input* projection cheaper to compute, because sparsity is a
   property of z's post-selection distribution, not of W or x.
2. **Selection** — `select_top_k` (`region.rs:151`): `select_nth_unstable_by` partition, `O(n_units)`
   average case. Cheap relative to the matmul once `d_input` is non-trivial (all four sensory
   regions have `d_input` in the hundreds to low thousands post-encoder).

The binding-projection fix (gaps.md §25, `crates/somnus-perception/src/binding.rs`) sparsified a
*downstream* consumer of an already-sparse `SparseActivation` — it gathered only the k active rows
out of the binding weight matrix instead of building a dense `n_units`-length vector and matmul-ing
all of it. That works because the binding projection's *input* is already sparse (only k of
n_units entries are nonzero) — the matmul's FLOP count genuinely drops from `O(n_units × d_out)` to
`O(k × d_out)`.

The k-WTA scoring step has no such input-side sparsity to exploit: `x` (the raw encoder embedding)
is dense, and every output unit's score is an independent dot product against all of `x`. There is
no way to know a unit will lose before computing its score — that's the definition of top-k
selection over an unstructured competition. So `scaled_k` (region.rs:20, keeps `k/n` ≈ 2% constant
as regions grow) controls storage and downstream FLOPs, but **does nothing for the scoring matmul,
which scales linearly in `n_units` regardless of `k`.**

### What this means for the 10x campaign and beyond

At `d_input` ~ a few hundred to ~1,500 (varies by region; see `architecture.md` encoder dims) and
`n_units` growing toward the campaign's ~9,660/region target (honeybee-order, `brain-scale-
comparison.md`), the scoring matmul is already the dominant per-region FLOP cost, and it grows
*linearly with n_units* — a 10x region-size increase is a 10x scoring-cost increase, full stop, no
sparsity discount available under the current all-to-all connectivity. Approaching further scale
(honeybee → mouse is another ~1,300x in unit count, per `brain-scale-comparison.md`) under this
architecture would mean a proportional FLOP increase, not a sub-linear one — the opposite of what
biological scaling looks like (cortical columns are *not* all-to-all; a given input reaches a small,
structured fan-out of candidate units, not every unit in the area).

This is the real lever: not "make the existing computation faster" (ORT/MIGraphX, §22, already
did that for the dense case; §25 did it for already-sparse downstream consumers) but "change what
gets computed at all" — reduce the number of units that must be *scored* per input, not just the
number that end up *active*.

### Candidate mechanisms (not evaluated in depth — first-pass survey only)

- **Local / blocked connectivity.** Partition `n_units` into blocks (e.g. by locality-sensitive
  hash of input space, or simply fixed contiguous blocks assigned at growth time) and route each
  input to a small subset of blocks based on a cheap gating signal, scoring only units within the
  routed blocks. Closest biological analogue: cortical minicolumns / receptive-field locality —
  units don't compete globally, they compete within a neighborhood. Engineering analogue: mixture-
  of-experts routing (sparse MoE gating, e.g. Shazeer et al. 2017 "Outrageously Large Neural
  Networks," arXiv:1701.06538) or LSH-based approximate nearest-neighbor bucketing. Risk: routing
  quality determines whether the "right" units are even in the candidate set — a bad router
  reintroduces the dead-unit problem this codebase already fought once (duty-cycle boosting,
  `region.rs` `BOOST_STRENGTH`/`DUTY_CYCLE_ALPHA`).
- **Hierarchical / coarse-to-fine WTA.** Two-stage competition: a cheap coarse scorer over
  super-units (e.g. `n_units / block_size` groups, each represented by a small summary
  projection) picks a handful of promising blocks, then full-resolution scoring runs only within
  those blocks. Reduces the dense matmul from `O(d_input × n_units)` to
  `O(d_input × n_blocks) + O(d_input × block_size × n_selected_blocks)`. Similar in spirit to
  ANN search structures already used elsewhere in this codebase for retrieval (HNSW in the
  warm/cold memory stores) — worth checking whether an existing ANN-style index could double as
  the coarse stage rather than building a bespoke gating network.
- **Approximate top-k via sampling/sketching.** Score a random or structured subset of units each
  forward pass (different subset each time, or importance-weighted by duty cycle / recent
  activation) instead of all of them, trading exactness for cost. Cheapest to implement, weakest
  guarantee — units outside the sampled subset can never win on a given input even if they'd have
  scored highest, which could interact badly with the growth mechanism's assumption that new units
  get a fair chance to activate (cascade-correlation training assumes the new unit's correlation
  with residual error is actually measured, not stochastically missed).

### What would need to happen before implementation

1. **Quantify the actual cost.** No isolated benchmark of the k-WTA scoring matmul alone exists yet
   (unlike the binding-projection fix, which had a clean before/after benchmark at the campaign's
   real region size). First step for whoever picks this up: measure `KWtaLayer::forward`'s wall-
   clock at current and projected region sizes, on both the `burn` path and — since ORT owns most
   live wake-path cost per §22/§24/§25's correction — check whether the exported ONNX encoder graph
   (`scripts/export_perception_graph.py`) has the same linear-in-n_units scoring cost baked in, since
   that's what's actually running on this campaign.
2. **Decide whether this is worth doing at honeybee scale at all.** The current 10x campaign target
   (~9,660 units/region) is not large enough to make `O(d_input × n_units)` scoring dominate wall-
   clock catastrophically (it's already linear, not quadratic) — this is a "prepare for the next
   order of magnitude" question, not an urgent fix for the in-flight campaign. Worth an explicit
   check against `brain-scale-comparison.md`'s scale ladder before investing design time.
3. **If pursued, needs an ADR**, not a mechanical fix — this changes representational structure
   (units are no longer globally competing), which interacts with growth (`grow_region_seeded`),
   duty-cycle boosting, and the binding area's assumption that any of the `n_units` per region can
   be "the" active set for a given input. Each candidate mechanism above needs its own numerical
   validation (does sparsified routing still let `per_region_mse` improve the way pilot-2/3 showed
   full competition doing) before being adopted.

**ADR written:** `decisions/0038-sparse-encoder-scoring.md` (2026-07-14, status `proposed`) records
the chosen direction — hierarchical/coarse-to-fine WTA, picked over MoE-style routing and
approximate sampling — so a future session doesn't have to re-derive the three-way comparison above.
Implementation remains explicitly not started, gated on the same prerequisites.

---

## Open questions → future ADR

1. Start with k-WTA (simpler) or SAE (more principled) for Phase 2 regional areas? SAE gives the reconstruction signal that the growth mechanism needs — worth the added complexity.
2. Shared sparsity level across all regions, or per-region tuning? Visual and audio concepts likely differ in intrinsic dimensionality.
3. How to handle dead units in k-WTA — boosting threshold, or explicit reinitialisation?
4. Binding area sparsity: does the integrative area benefit from sparsity, or is a dense representation more appropriate there?

---

## Sources

- Olshausen & Field, 1996. "Emergence of simple-cell receptive field properties by learning a sparse code for natural images." *Nature* 381: 607–609.
- Hawkins, Ahmad, & Cui, 2017. "A Theory of How Columns in the Neocortex Enable Learning the Structure of the World." *Frontiers in Neural Circuits*. HTM sparse distributed representations.
- Makhzani & Frey, 2013. "k-Sparse Autoencoders." arXiv:1312.5663.
- Elhage et al., 2022. "Toy Models of Superposition." Anthropic. (Motivates SAE for disentangled features.) Transformer Circuits Thread.
- Bricken et al., 2023. "Towards Monosemanticity: Decomposing Language Models with Dictionary Learning." Anthropic. (Sparse autoencoders applied to transformer features.)
- Martins & Astudillo, 2016. "From Softmax to Sparsemax." ICML. arXiv:1602.02068.
- Peters et al., 2019. "Sparse Sequence-to-Sequence Models." ACL. Entmax.

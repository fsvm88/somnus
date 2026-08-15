# Sparse coding

> **Contents:** k-WTA (✓ committed, n=1024, k=20), top-k, SAE/L1-penalized autoencoders. Covers dead-unit problem, duty-cycle boosting solution (DUTY_CYCLE_ALPHA=0.001, BOOST_STRENGTH=1.0, TARGET_DC=20/1024), compute overhead, interaction with growth. Also covers the encoder-side scoring-cost scaling problem (§"Encoder-side scaling") — why sparsity doesn't reduce the O(input_dim × n_units) k-WTA scoring matmul, and candidate mechanisms (local/blocked connectivity, hierarchical WTA, approximate top-k) for a future ADR.
> **When to read:** Tuning k-WTA hyperparameters; debugging dead units; understanding why duty-cycle boosting was added; scoping the next scaling step past the current 10x growth campaign (gaps-resolved.md §25/§16).

Research note covering sparse representation schemes for the regional concept areas: k-winner-take-all (k-WTA), top-k activation, sparse autoencoders (SAE/L1-penalized). All numbers are **engineering estimates**. See `architecture.md` "Regional concept areas" for the design commitment — sparsity reduces interference between concepts and makes growth local.

---

## Why sparsity

Dense representations touch every weight on each update, causing interference (catastrophic forgetting). Sparse codes give concepts largely non-overlapping unit subsets: concept A on {1,5,12,33}, concept B on {2,8,17,44} — a B update doesn't touch A. Expected overlap between two random concepts ≈ k²/n (k=20,n=1000 → 0.4 units, negligible). This is the HTM/sparse-distributed-representation insight (Hawkins/Numenta). Sparsity also enables local growth — new units only affect representations that use them.

---

## Metrics for sparsity

**Sparsity level** = k/n (useful range 1-10%). **Lifetime sparsity** = a unit's average activation frequency across inputs — a unit firing on 5% of inputs is more useful than 90% or 0.01%. **Population sparsity** = k/n per input. Good codes have both low.

---

## Schemes

### k-WTA (k-winner-take-all)

Hard competitive constraint: `z=Wx+b`, keep top-k, zero the rest. **Compute:** finding the k-th largest is O(n) (QuickSelect) or O(n log k) (min-heap); `torch.topk`-equivalent is a fast CUDA primitive, negligible at n=1000/k=20 (~0.01ms). Backward scales with k, not n. **Memory:** no extra beyond the weight matrix; sparse (index,value) storage is k×2 vs. n floats — 49× reduction at k=20/n=1000. **Tradeoffs:** dead units (no gradient without boosting mitigation); discontinuous gradient (needs straight-through estimator); fixed k = fixed capacity; simple, widely used (HTM, spiking NN). **Recommended in somnus:** k-WTA + boosting on regional linear layers, k/n≈0.02-0.05, monitor lifetime sparsity for dead units.

### Top-k activation (soft variant)

Differentiable approximations: **sparsemax** (projects onto the probability simplex, sparse solution, needs sort+threshold, slower than hard top-k); **entmax(α)** (softmax/sparsemax generalization, tunable, α=1.5 a good middle ground, needs a custom kernel); **straight-through top-k** (hard forward, full-gradient backward — simplest, works well). Sort cost O(n log n) exact / O(n) approximate bucketed; ~10-20µs CPU, <1µs GPU at n=1000/k=20. Cleaner gradient flow than hard k-WTA, less interpretable.

### Sparse autoencoder (SAE) / L1-penalized

`loss = reconstruction + λ·||z||₁`. **Training cost:** ~2-3× a plain autoencoder; convergence needs thousands of replay steps. **Inference cost:** encoder forward only, no sorting, sparsity is emergent. **Memory:** small (e.g. d_input=512, n_hidden=2000 → 4MB FP32 encoder). Dead units still occur but the L1 penalty partially mitigates (zero-gradient on never-used codes). **Tradeoffs:** data-driven coding space; reconstruction error is a free growth-trigger signal; needs replay data to converge (underdeveloped early); λ tuning sensitive (too high→all-inactive, too low→dense); dictionary learning (online DL, KSVD) is a non-gradient alternative, possibly more efficient for small areas. **Recommended in somnus:** the principled long-run choice — gives reconstruction error as a free growth signal; k-WTA is simpler and adequate short-term.

**Comparison:**

| Scheme | Sparsity type | Gradient | Compute overhead | Self-probe signal | Complexity |
|---|---|---|---|---|---|
| k-WTA | Hard, fixed k | Discontinuous (needs STE) | Minimal (topk call) | None directly | Low |
| Top-k / STE | Hard, fixed k | Straight-through | Minimal (topk call) | None directly | Low |
| Sparsemax | Soft, dynamic | Differentiable | Sort + project | None directly | Medium |
| SAE (L1) | Soft, emergent | Clean backprop | 2-3× training | Reconstruction error | Medium |
| Dictionary learning | Hard, learned | Alternating minimization | Per-iteration | Reconstruction error | Medium-High |

---

## Interaction with growth

The cascade-correlation growth trigger (add units on gist-recall failure) fits naturally: new units start random/zero-weighted, inactive for existing patterns; cascade-correlation trains them against the current reconstruction residual, activating for currently-underrepresented inputs. Because existing patterns are sparse, new units fill new "slots" without disrupting them. SAE reconstruction error directly quantifies which inputs need more capacity — the growth condition.

---

## Budget analysis

Region n=2000, k=40 (2% sparsity), d_input=384 (DINOv2-S): weight matrix 384×2000×4B=3.1MB (negligible). Per-episode sparse storage: 40 floats+40 indices=320 bytes (vs. dense 2000×4B=8KB) — **25× smaller** at 2% sparsity; still 50× at n=10K/k=200. Across 4 regions + binding, total weight matrices <100MB even at n=5000/region — negligible against the 14GB VRAM budget.

---

## Encoder-side scaling: why sparsity doesn't save the O(input_dim × n_units) scoring cost

Raised 2026-07-13/14 looking for genuine scaling levers toward mouse-order size, not just faster GPU cycles on the current architecture. Queued in `gaps.md` §25; this section is that write-up — design note only, not implemented.

**The asymmetry: projection vs. selection.** `KWtaLayer` (`somnus-perception/src/region.rs`) has two costs: **scoring** (`z=Wx`, `region.rs:117`, dense `[d_input]→[n_units]` matmul, `O(d_input×n_units)` — every candidate must be scored before top-k can pick winners, so an already-sparse *output* doesn't make the *input* projection cheaper, since sparsity is a property of z's post-selection distribution, not of W or x) and **selection** (`select_top_k`, `region.rs:151`, `O(n_units)` average — cheap once `d_input` is non-trivial).

The binding-projection fix (gaps-resolved.md §25, `binding.rs`) sparsified a *downstream* consumer of an already-sparse `SparseActivation` — gathering only k active rows instead of a dense matmul, genuinely dropping FLOPs from `O(n_units×d_out)` to `O(k×d_out)` because its *input* was already sparse. The k-WTA scoring step has no such input-side sparsity: `x` is dense, every unit's score is an independent dot product against all of `x`, and there's no way to know a unit will lose before scoring it — that's what top-k selection over an unstructured competition means. So `scaled_k` (keeps k/n≈2% constant as regions grow) controls storage and downstream FLOPs but **does nothing for the scoring matmul, which scales linearly in n_units regardless of k.**

**What this means for the 10x campaign and beyond.** At `d_input` ~hundreds-to-~1,500 and `n_units` growing toward ~9,660/region (honeybee-order), the scoring matmul is already the dominant per-region FLOP cost and scales linearly with n_units — a 10x region-size increase is a 10x scoring-cost increase, no sparsity discount under current all-to-all connectivity. Approaching mouse-order (another ~1,300x in unit count) under this architecture means a proportional FLOP increase, not sub-linear — the opposite of biological scaling (cortical columns aren't all-to-all; inputs reach a small structured fan-out, not every unit). The real lever isn't "make the existing computation faster" (ORT/MIGraphX §22 did that for the dense case, §25 for already-sparse downstream consumers) but "change what gets computed at all."

**Candidate mechanisms (first-pass survey, not evaluated in depth):**

- **Local/blocked connectivity.** Partition n_units into blocks, route each input to a small subset via a cheap gating signal, score only routed blocks. Biological analogue: cortical minicolumns/receptive-field locality. Engineering analogue: sparse MoE gating (Shazeer et al. 2017, arXiv:1701.06538) or LSH bucketing. Risk: a bad router reintroduces the dead-unit problem duty-cycle boosting already fought once.
- **Hierarchical/coarse-to-fine WTA.** Cheap coarse scorer over super-units picks promising blocks, full-resolution scoring runs only there — `O(d_input×n_units)` → `O(d_input×n_blocks)+O(d_input×block_size×n_selected)`. Similar in spirit to the HNSW indexes already used for retrieval elsewhere in this codebase — worth checking whether an existing ANN structure could double as the coarse stage.
- **Approximate top-k via sampling/sketching.** Score a random/importance-weighted subset each pass. Cheapest, weakest guarantee — units outside the sample can never win even if they'd score highest, which could interact badly with growth's assumption that new units get a fair, measured chance (cascade-correlation needs actual correlation measurement, not stochastic misses).

**Before implementation:** (1) quantify the actual cost — no isolated benchmark of the k-WTA scoring matmul exists yet (unlike the binding-projection fix); measure `KWtaLayer::forward` at current/projected sizes on both `burn` and the exported ONNX encoder graph, since ORT owns most live wake-path cost. (2) decide if it's worth doing at honeybee scale — the 10x target isn't large enough to make linear scoring cost catastrophic; check against `brain-scale-comparison.md`'s scale ladder first. (3) if pursued, needs an ADR, not a mechanical fix — changes representational structure (units no longer globally compete), interacting with growth/duty-cycle/binding assumptions; each mechanism needs its own numerical validation that `per_region_mse` still improves the way full competition does.

**ADR written:** `decisions/0038-sparse-encoder-scoring.md` (2026-07-14, `proposed`) records the chosen direction — hierarchical/coarse-to-fine WTA, over MoE-routing and approximate sampling — so this comparison doesn't need re-deriving. Implementation not started, gated on the same prerequisites.

---

## Open questions → future ADR

1. k-WTA (simpler) or SAE (more principled, gives the growth-signal reconstruction error) for Phase 2 regional areas?
2. Shared sparsity level across regions, or per-region tuning (visual/audio likely differ in intrinsic dimensionality)?
3. Dead-unit handling: boosting threshold vs. explicit reinitialisation?
4. Does the binding/integrative area benefit from sparsity, or is dense more appropriate there?

---

## Sources

- Olshausen & Field, 1996. "Emergence of simple-cell receptive field properties by learning a sparse code for natural images." *Nature* 381: 607-609.
- Hawkins, Ahmad, & Cui, 2017. "A Theory of How Columns in the Neocortex Enable Learning the Structure of the World." *Frontiers in Neural Circuits.*
- Makhzani & Frey, 2013. "k-Sparse Autoencoders." arXiv:1312.5663.
- Elhage et al., 2022. "Toy Models of Superposition." Anthropic, Transformer Circuits Thread.
- Bricken et al., 2023. "Towards Monosemanticity: Decomposing Language Models with Dictionary Learning." Anthropic.
- Martins & Astudillo, 2016. "From Softmax to Sparsemax." ICML, arXiv:1602.02068.
- Peters et al., 2019. "Sparse Sequence-to-Sequence Models." ACL. Entmax.

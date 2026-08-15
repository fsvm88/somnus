# Memory compression

> **Contents:** Warm→cold tier compression options — INT8 quantization (✓ implemented, 640 bytes/record), Product Quantization (✓ implemented, 27.5× reduction), structured pruning, distillation. Covers compression pipeline and quality tradeoffs.
> **When to read:** Tuning the cold archive format; adding a new compression stage; understanding the PQ codebook.

Research note covering warm→cold tier compression: numeric quantization (FP16/INT8/INT4), structured pruning, knowledge distillation. All numbers are **engineering estimates**. See `architecture.md` "Sleep stages" (NREM: "Compression") and "Memory hierarchy" — compression moves semantic memory content warm→cold during NREM sleep as the warm tier approaches capacity or content consolidates sufficiently.

---

## What we are compressing and why

The warm tier (~56GB RAM) holds region-sharded sparse-coded concept embeddings + the HNSW index over them. Older/less-cued content needs to move to cold (disk) to free RAM; cold holds far more but queries are expensive (disk I/O + deserialization), so cold content should be compact. Goals: reduce storage footprint of vectors moved to cold; preserve recall quality; keep compression runtime affordable within the sleep window. This is *representation* compression (stored embeddings), not model compression (though the cognitive-layer networks that encode/decode them are a separate, related target below).

---

## Numeric precision reduction

**FP32→FP16:** 2× reduction, negligible quality loss (cosine similarity preserved to >4 decimal places, industry standard for ML inference), near-zero runtime cost (a cast, native CUDA FP16), lossless-ish reversal. **All warm-tier vectors should be FP16 by default** — free 2× savings, no reason to store FP32 there.

**FP32/FP16→INT8:** 4×/2× reduction, ~0.5-2% cosine-recall drop (est.), acceptable for most retrieval. Per-vector scale = max(|v|)/127, quantized = round(v/scale), storage = n bytes + 4-byte scale (or per-cluster scale, less accurate, simpler). Compression: one pass, trivially parallel, ~1.5GB/1M vectors <5s GPU (est.). Decompression: one multiply/element, ~5-10% retrieval overhead (est.). **Primary cold-tier compression format**, applied during NREM spill.

**FP32→INT4:** 8×/4× reduction, ~3-8% recall drop (est.) — noticeable but potentially acceptable for remote memories. Similar to INT8 but 16 levels, needs careful per-vector/per-group calibration; NF4 (QLoRA/bitsandbytes) is a non-uniform variant better fitting embeddings' typical Gaussian distribution. Higher per-element decompression cost (lookup tables). **For the deepest cold-tier archive only** (very old/rarely-cued content) — a two-level cold tier (INT8 near, INT4 far) may be worth modeling in Phase 5+.

**Product Quantization (PQ):** split into `m` sub-vectors, quantize each independently against a shared `k`-centroid codebook. Size: `m×log₂(k)` bits/vector — m=8,k=256 → 8 bytes/vector regardless of dimension. Quality depends on m (384-dim/m=8 → 48-float sub-vectors; m=16 better recall); expected recall@1 drop ~5-15% (est.). Needs one-time k-means training per region; fast codebook lookups after that. **Right tool for the cold tier at tens of millions of vectors** — not needed until Phase 5+.

---

## Structural / pruning compression

**Cognitive-layer network pruning** (compressing the encoder/decoder/k-WTA weights themselves): **structured** (remove low-lifetime-activation units' rows/columns — smaller network, faster forward pass, lower sleep-time VRAM; requires a recovery pass ensuring any concept the removed unit represented is re-coded elsewhere first; cost O(n_episodes×n_units) to measure activations) vs. **magnitude** (zero individual weights below a threshold — marginal benefit since sparse BLAS isn't faster than dense at our matrix sizes, <10M params/region; not worth the complexity through Phase 1-4).

## Knowledge distillation (semantic compression)

Train a smaller student to match a larger teacher's outputs — use case: periodically distill a cascade-correlation-grown region into a more compact network during sleep. Needs replay data for teacher outputs; student training ~10-100 GPU-hours for ~10M params (est.) — too expensive to do often, a quarterly/annual operation. Phase 5+ mechanism, premature until the network has grown substantially.

**Alternative — compression by regularization** (SHY-inspired homeostatic downscaling): L2 decay toward zero during NREM, with replay-reinforced weights resisting decay and unreinforced weights shrinking — already the design's homeostatic downscaling mechanism, a soft continuous compression rather than an explicit distillation pass. Cost: one multiply-add/weight/sleep-step, ~40MB weights for a 10M-param region, <1ms/step GPU.

---

## Compression pipeline during sleep

1. **Identify content:** warm-tier vectors last accessed >T sleep cycles ago (or below a salience threshold) — cold-spill candidates.
2. **Quantize to INT8:** one GPU pass.
3. **Serialize to disk:** quantized vectors + scale factors + metadata, region-sharded files.
4. **Remove from warm HNSW:** logical tombstone; periodic rebuild during long sleep windows.
5. **Homeostatic downscaling:** L2 decay on unreinforced cognitive-layer weights, O(n_params), <1ms/region.
6. **Optionally rebuild HNSW** (tombstone accumulation past threshold): O(n_remaining log n) — infrequent.

---

## Budget analysis

**Warm capacity** (56GB RAM, FP16): 384-dim = 768B/vector → ~73M vectors; with HNSW overhead (~2×) → ~36M vectors. **Cold capacity** (disk, INT8): 384-dim+scale = 389B/vector → ~2.6B vectors/TB; PQ (8B/vector) effectively unlimited at any expected scale. **Compression runtime:** INT8-quantize 1M vectors <5s GPU; write 1M INT8 vectors (~389MB) ~0.5-2s NVMe; total per-sleep overhead for 1M spilled <10s, negligible. **The real bottleneck is rebuilding HNSW after deletions** — 10M vectors ≈ 10-30min. Strategy: defer rebuilds to long sleep windows, tombstone for short ones.

---

## Open questions → future ADR

1. INT8 vs. FP16 as the warm-tier storage format — FP16 is simpler/2× smaller than FP32 and probably sufficient at 56GB; INT8 adds complexity for a benefit that may not be needed.
2. Per-vector vs. per-cluster INT8 scale — per-vector is more accurate but doubles metadata overhead.
3. HNSW tombstones vs. periodic rebuilds — what deletion threshold triggers a rebuild (10%? 30%)?
4. When (if ever) to trigger cognitive-layer distillation vs. relying on homeostatic downscaling alone?

---

## Sources

- Dettmers et al., 2022. "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale." arXiv:2208.07339.
- Dettmers et al., 2023. "QLoRA: Efficient Finetuning of Quantized LLMs." arXiv:2305.14314. (NF4.)
- Jégou, Douze, & Schmid, 2011. "Product Quantization for Nearest Neighbor Search." *IEEE TPAMI*.
- Han, Pool, Tran, & Dally, 2015. "Learning Both Weights and Connections for Efficient Neural Networks." NeurIPS.
- Hinton, Vinyals, & Dean, 2015. "Distilling the Knowledge in a Neural Network." arXiv:1503.02531.
- Tononi & Cirelli, 2014. "Sleep and the price of plasticity." *Neuron*.
- OpenSearch FP16 scalar quantization: opensearch.org/blog/optimizing-opensearch-with-fp16-quantization/

# Memory compression

> **Contents:** Warm→cold tier compression options — INT8 quantization (✓ implemented, 640 bytes/record), Product Quantization (✓ implemented, 27.5× reduction), structured pruning, distillation. Covers compression pipeline and quality tradeoffs.  
> **When to read:** Tuning the cold archive format; adding a new compression stage; understanding the PQ codebook.

Research note for somnus. Covers warm→cold tier compression: numeric quantization (FP16/INT8/INT4), structured pruning, and knowledge distillation. All numbers are **engineering estimates**.

See `architecture.md` "Sleep stages" (NREM-like: "Compression") and "Memory hierarchy". Compression moves semantic memory content from warm (RAM) to cold (disk) during NREM-like sleep when the warm tier approaches capacity or when content has been sufficiently consolidated.

---

## What we are compressing and why

The warm tier (semantic memory, ~56 GB RAM) holds region-sharded sparse-coded concept embeddings and the HNSW index over them. Over time, older or less-frequently-cued content needs to move to the cold tier (disk) to free RAM for new consolidation. The cold tier can hold far more, but queries against it are expensive (disk I/O + deserialization), so cold content should be as compact as possible.

Compression goals:
1. **Reduce storage footprint** of vectors moved to cold (fewer bytes = faster disk I/O + more fits on disk).
2. **Preserve recall quality** — retrieved vectors must still be useful for recognition and binding.
3. **Keep compression runtime affordable** — NREM-like sleep must compress a meaningful chunk of content within the sleep window.

This is *not* model compression (pruning weights of a neural network for faster inference). It is *representation compression*: compressing stored embeddings and, separately, potentially compressing the cognitive-layer networks that encode and decode them.

---

## Numeric precision reduction

### FP32 → FP16

- **Size reduction:** 2×
- **Quality loss:** Negligible for embeddings (cosine similarity is preserved to >4 decimal places). Industry standard for ML inference.
- **Runtime cost:** near-zero (a cast, no computation). CUDA FP16 support is native.
- **Reversibility:** FP16 → FP32 is lossless (FP16 is a subset of FP32 representable values; dequantization may lose the last ~3 bits but is negligible).

**Decision for somnus:** all warm-tier vectors should be stored in FP16 by default (not FP32). There is no reason to store FP32 in the warm tier. This is a free 2× savings.

---

### FP32/FP16 → INT8 (8-bit quantization)

- **Size reduction:** 4× from FP32, 2× from FP16.
- **Quality loss:** ~0.5–2% drop in cosine recall on typical embedding benchmarks (est.). Acceptable for most retrieval tasks.
- **How it works:** scale each vector to the INT8 range [−128, 127] using a per-vector scale factor; store the scale factor alongside the quantized codes.
  - Per-vector: scale = max(|v|) / 127; quantized = round(v / scale). Storage: n bytes + 4 bytes for scale.
  - Per-layer/per-cluster: one scale for a group of vectors. Slightly less accurate but simpler.
- **Runtime cost for compression:** one pass over the vectors; trivially parallelizable. For 1M vectors × 384 dims: ~1.5 GB processed; takes <5 s on GPU (est.).
- **Decompression cost (at retrieval):** dequantize on-the-fly before distance computation: one multiply per element. Adds ~5–10% overhead to retrieval (est.).

**Decision for somnus:** INT8 is the primary compression format for cold-tier vectors. Applied during NREM-like sleep when content is spilled to cold.

---

### FP32 → INT4 (4-bit quantization)

- **Size reduction:** 8× from FP32, 4× from FP16.
- **Quality loss:** ~3–8% drop in cosine recall (est.). Noticeable but potentially acceptable for remote memories.
- **How it works:** similar to INT8 but with only 16 levels. Requires per-vector or per-group calibration. NF4 (Normal Float 4, from QLoRA/bitsandbytes) is a non-uniform quantization that better fits the typical Gaussian distribution of embedding values.
- **Runtime cost for compression:** similar to INT8; slightly more careful calibration needed.
- **Decompression cost:** higher per-element cost than INT8 due to NF4 lookup tables.

**Decision for somnus:** INT4 for the deepest cold-tier archive — very old or rarely-cued content. Not for the primary cold tier. A two-level cold tier (INT8 near-cold, INT4 far-cold) may be worth modeling in Phase 5+.

---

### Product Quantization (PQ)

Compress vectors by splitting them into `m` sub-vectors and quantizing each sub-vector independently with a shared codebook of `k` centroids.

- **Size:** `m × log₂(k)` bits per vector. Typical: m=8 sub-vectors, k=256 centroids → 8 bytes per vector, regardless of dimension.
- **Quality loss:** depends on m. For 384-dim, m=8 → each sub-vector = 48 floats; high-quality PQ with m=16 gives better recall. Expected recall@1 drop: ~5–15% (est.).
- **Requires training:** k-means over the dataset to build codebooks. One-time cost per region.
- **Runtime cost (compression):** one codebook lookup per sub-vector per vector. Fast after codebooks are trained.

**Decision for somnus:** PQ is the right tool for the cold tier when vectors number in the tens of millions and storage is the constraint. Not needed until Phase 5+.

---

## Structural / pruning compression

### Cognitive-layer network pruning (warm-tier model compression)

The regional concept area networks (encoder + decoder of the SAE, or the k-WTA linear layer) can themselves be pruned to reduce their parameter count.

**Structured pruning (remove units/neurons):**
- Identify units with low lifetime activation frequency (dead or near-dead units) and remove their row/column from the weight matrix.
- Impact: smaller network → faster forward pass → lower VRAM during sleep operations.
- Recovery strategy: before removing a unit, ensure any activated concept it represents has been re-coded into remaining units (requires a pass during sleep).
- Cost: one forward pass over recent episodes to measure unit activations. O(n_episodes × n_units).

**Magnitude pruning (remove individual weights):**
- Zero out individual weights below a threshold. Results in sparse weight matrices.
- Benefit: marginal — sparse BLAS is not faster than dense for the matrix sizes we use (<10M params per region).
- Not worth the complexity for Phase 1–4.

---

## Knowledge distillation (semantic compression)

Train a smaller "student" cognitive network to match the outputs of the larger "teacher" network.

**Use case in somnus:** when a regional concept area has grown large via cascade-correlation (many added units over time), periodically distill it into a more compact network during sleep.

**Cost:**
- Requires access to replay data (episodes) to generate teacher outputs.
- Training the student: ~10–100 GPU hours for a network of ~10M parameters (est.). Too expensive to do frequently; a quarterly or annual operation.
- This is a Phase 5+ mechanism — premature until the network has grown substantially.

**Alternative: compression by regularization (SHY-inspired homeostatic downscaling).**
- Instead of explicit distillation, apply an L2 weight decay toward zero during NREM-like sleep.
- Weights reinforced by recent replay resist decay; unreinforced weights shrink.
- This is the homeostatic downscaling mechanism already in the design — it is a soft form of compression that continuously prunes unused structure.
- Cost: one multiply-add per weight per sleep step. For a 10M-param region: ~40 MB of weights; one downscaling step takes <1 ms on GPU.

---

## Compression pipeline during sleep

Proposed NREM-like sleep compression sequence:

1. **Identify content to compress:** select all warm-tier vectors last accessed more than T sleep cycles ago (or below a salience threshold). These are candidates for cold spill.
2. **Quantize to INT8:** one GPU pass over candidate vectors.
3. **Serialize to disk:** write quantized vectors + scale factors + metadata to region-sharded files.
4. **Remove from warm-tier HNSW index:** logical deletion or tombstone; periodic index rebuild during long sleep windows.
5. **Homeostatic downscaling:** apply L2 decay to cognitive-layer weights not reinforced by recent replay. Compute: O(n_params), <1 ms per region.
6. **Optionally rebuild HNSW index** (if tombstone accumulation exceeds a threshold): O(n_remaining × log n) — do this infrequently.

---

## Budget analysis

**Warm-tier capacity (56 GB RAM, vectors at FP16):**
- 384-dim FP16: 768 bytes/vector. 56 GB → ~73M vectors.
- With HNSW overhead (~2×): ~36M vectors.

**Cold-tier capacity (disk, vectors at INT8):**
- 384-dim INT8 + scale (389 bytes/vector). For 1 TB disk: ~2.6B vectors.
- Product-quantized (8 bytes/vector): effectively unlimited for any expected scale.

**Compression runtime during sleep (est.):**
- INT8 quantize 1M vectors: <5 s on GPU.
- Write 1M INT8 vectors to disk (~389 MB): ~0.5–2 s on NVMe SSD.
- Total per-sleep overhead for 1M vectors spilled: <10 s. Negligible within a sleep window.

The bottleneck is not compression — it is rebuilding the HNSW index after deletions. Index rebuild for 10M vectors: ~10–30 min. Strategy: defer rebuilds to long sleep windows; use tombstones for short ones.

---

## Open questions → future ADR

1. INT8 vs. FP16 as the warm-tier storage format? FP16 is simpler and 2× smaller than FP32; INT8 halves that again. For a 56 GB warm tier, FP16 is probably sufficient — INT8 adds complexity for a benefit we may not need.
2. Per-vector vs. per-cluster INT8 scale? Per-vector is more accurate but doubles the metadata overhead.
3. HNSW tombstones vs. periodic rebuilds: what threshold triggers a rebuild (10% deleted? 30%)?
4. When to trigger cognitive-layer distillation (if ever) vs. relying on homeostatic downscaling alone?

---

## Sources

- Dettmers et al., 2022. "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale." arXiv:2208.07339.
- Dettmers et al., 2023. "QLoRA: Efficient Finetuning of Quantized LLMs." arXiv:2305.14314. (NF4 quantization format.)
- Jégou, Douze, & Schmid, 2011. "Product Quantization for Nearest Neighbor Search." *IEEE TPAMI*. (PQ for vector compression.)
- Han, Pool, Tran, & Dally, 2015. "Learning Both Weights and Connections for Efficient Neural Networks." NeurIPS. (Magnitude pruning.)
- Hinton, Vinyals, & Dean, 2015. "Distilling the Knowledge in a Neural Network." arXiv:1503.02531.
- Tononi & Cirelli, 2014. "Sleep and the price of plasticity." *Neuron*. (Homeostatic downscaling inspiration.)
- OpenSearch FP16 scalar quantization post — practical INT8/FP16 HNSW memory reduction benchmarks: opensearch.org/blog/optimizing-opensearch-with-fp16-quantization/

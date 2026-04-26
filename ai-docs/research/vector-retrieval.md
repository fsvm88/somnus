# Vector retrieval

Research note for somnus. Covers the recollection layer's retrieval mechanism: approximate nearest-neighbor (ANN) indexes, exact search, and associative-memory (Hopfield) approaches. All numbers are **engineering estimates**.

See `architecture.md` "Memory hierarchy" for the recollection layer design. The recollection layer is the cross-tier index: given a partial cue (from current perception or during sleep), it identifies relevant memories anywhere in the warm or cold tiers and materializes them into warmer tiers.

---

## What we need

At query time (during wake or sleep):
- Input: a cue vector (embedding from current perception or partial recall).
- Output: the top-k most semantically similar stored vectors, plus their metadata (timestamps, region, episode ID).
- Latency: wake queries should complete in <5 ms; sleep queries can be slower.
- Scale: initially thousands to millions of vectors. Long-term: potentially hundreds of millions in cold storage.

Writes (during sleep consolidation):
- Insert new vectors (semantic memory updates after replay).
- Optionally delete or merge near-duplicate vectors (compression step).
- Batch inserts during sleep are acceptable; inserts during wake should be avoided (deferred to sleep).

---

## Options

### Flat / exact search

Brute-force inner-product or cosine similarity over all stored vectors. Can be BLAS-accelerated.

**Memory:**
- Zero overhead beyond the raw vector store.
- For 384-dim FP16 vectors: 768 bytes/vector. 1M vectors = 768 MB RAM.

**Latency:**
- O(n × d) per query, BLAS-accelerated: ~10 ms for 100K vectors at dim=512 on CPU (est.); GPU-accelerated FAISS flat index: ~1 ms for 1M vectors.
- Does not scale past ~10M vectors at interactive latency.

**Tradeoffs:**
- Perfect recall (100%) — finds the exact nearest neighbor.
- No build step, no hyperparameters, trivially updatable (append-only).
- Becomes unacceptably slow for large archives (cold tier), acceptable for hot/warm tiers at modest scale.
- Good starting point for Phase 1–2 before scale demands ANN.

**Rust options:** `usearch` (USearch library), `hnsw_rs` flat mode, custom BLAS over `ndarray`/`faer`.

---

### HNSW (Hierarchical Navigable Small Worlds)

Graph-based ANN algorithm. Builds a multi-layer proximity graph; queries navigate from coarse to fine layers. The standard high-quality ANN choice.

**Memory:**
- Formula: approximately `1.1 × (vector_bytes + 8 × M)` per vector, where M is the max number of outgoing edges per node (typically 16–64).
- For 384-dim FP16 (768 bytes), M=16: 1.1 × (768 + 128) ≈ 986 bytes/vector — roughly 28% overhead over raw vectors.
- For 384-dim FP32 (1536 bytes), M=16: 1.1 × (1536 + 128) ≈ 1830 bytes/vector — 19% overhead.
- 1M vectors @ 384-dim FP16: ~986 MB total index (raw + graph), fits comfortably in RAM.
- 10M vectors @ 384-dim FP16: ~9.4 GB — still fits in 56 GB RAM.
- 100M vectors: ~94 GB — must live on disk (cold tier).

**Latency:**
- Query: O(log n) graph traversal. Empirically: ~0.1–2 ms per query on CPU for millions of vectors at 95%+ recall (est. from published SIFT1M benchmarks).
- Build: O(n log n); ~10–30 minutes for 10M vectors on CPU (est.). Not incremental by default — new insertions require graph re-linking which is O(M × log n) per insert.
- Insertion throughput: ~10K–100K inserts/second depending on M and hardware.

**Tradeoffs:**
- High recall (~95–99% at top-1) with sub-millisecond queries — the Pareto-optimal ANN choice for most workloads.
- Memory overhead (~2× raw vectors including the graph).
- Not great for streaming inserts at high rate — each insert modifies the graph. Fine for sleep-cycle batch inserts.
- Hyperparameters: M (graph connectivity, default 16), ef_construction (build quality, default 200), ef_search (query quality, tunable at runtime). Higher M → better recall, more memory.
- No native deletion — logical tombstones or periodic index rebuild required.

**Rust options:** `hnsw_rs`, `instant-distance`, `usearch` (best-maintained, supports FP16 natively).

**Recommendation for warm tier:** HNSW is the right choice for the semantic memory recollection index once vectors exceed ~500K. Use flat search in Phase 1–2 and switch when needed.

---

### IVF-Flat / IVF-PQ (Inverted File Index)

Cluster-based ANN: divide the space into `nlist` Voronoi cells at training time; at query time, search only the top-`nprobe` cells.

**Memory:**
- IVF-Flat: ~1× raw vector size (just the vectors, organized into clusters). Lower overhead than HNSW.
- IVF-PQ: vector storage replaced by product-quantized codes — ~8–32 bytes/vector regardless of dimension. Extreme compression at the cost of recall.
- For 384-dim FP32 → IVF-PQ with 8 bytes/vector: 192× size reduction; recall drops to ~80–90%.

**Latency:**
- Faster than HNSW at massive scale (>100M vectors) for the same recall.
- Requires a training step (k-means over the dataset) before use — not available until you have enough data.

**Tradeoffs:**
- Requires a one-time training step — not suitable for early phases with small data.
- IVF-PQ is the standard choice for cold-tier archives where extreme compression is needed.
- Less dynamic than HNSW: adding new vectors outside known clusters degrades performance.

**Role in somnus:** IVF-PQ (or its equivalents) is the right tool for the cold-tier archive once it grows past 10M vectors. Not needed until Phase 4+.

---

### Modern Hopfield networks

Continuous Hopfield networks (Ramsauer et al., 2020) recast as an attention-like memory: storing patterns as "memories", retrieval via iterative update rule.

**Memory:**
- Patterns stored explicitly: O(N × d) — same as flat storage.
- No graph structure overhead.
- Capacity: modern Hopfield networks have exponential capacity in theory (~exp(d/2) patterns), but in practice limited to patterns that are sufficiently separated.

**Latency:**
- Single-step retrieval (softmax attention form): O(N × d) per query — same as flat search but differentiable.
- Multi-step iterative update: 3–10× the single-step cost for convergence.

**Tradeoffs:**
- Fully differentiable — can be trained end-to-end with the rest of the network (recollection + cognitive layers as one joint system).
- Does not require a separate index build step.
- Does not scale as well as HNSW at millions of vectors — the single-step retrieval remains O(N × d).
- The exponential capacity theorem is asymptotic; practical capacity with noisy patterns is much lower.
- Natural fit for associative recall in the binding area (retrieve a full episode given a partial cue) — the differentiability is the key advantage over HNSW.

**Role in somnus:** candidate for the warm-tier recollection sub-component, specifically for associative completion (given partial episode → fill in rest). Not a replacement for HNSW as a scalable cross-tier index, but complementary. Worth a Phase 3+ experiment.

---

## Hybrid: HNSW for scale, Hopfield for associative completion

The two are complementary:

1. **HNSW** indexes the warm + cold tier by region × concept: fast O(log n) lookup of the k nearest episodes to a query embedding.
2. **Modern Hopfield** sits inside the binding area: given the top-k candidates from HNSW, does associative completion (fills in missing modalities, resolves partial cues).

This separates the scale problem (HNSW handles millions) from the completion problem (Hopfield handles associative fill-in over the top-k retrieved).

---

## Budget analysis

Hardware: 56 GB usable RAM (warm tier), ~14 GB VRAM (hot tier), disk (cold tier).

| Scale        | HNSW FP16 (384-dim, M=16) | HNSW FP32 (384-dim, M=16) | Fits in RAM? |
|--------------|--------------------------|--------------------------|--------------|
| 100K vectors | ~99 MB                   | ~183 MB                  | Yes          |
| 1M vectors   | ~986 MB                  | ~1.83 GB                 | Yes          |
| 10M vectors  | ~9.4 GB                  | ~18.3 GB                 | Yes (FP16)   |
| 50M vectors  | ~47 GB                   | OOM                      | Tight FP16   |
| 100M vectors | ~94 GB                   | OOM                      | No → disk    |

At 384-dim FP16 with M=16, the warm tier HNSW fits up to ~50M vectors in 56 GB RAM. For somnus's expected scale through Phase 4 (hundreds of thousands to low millions of concepts), memory is not the binding constraint. Phase 5+ may need cold-tier IVF-PQ.

---

## Open questions → future ADR

1. Start with flat search (zero complexity) or HNSW from day one? Flat is simpler but swapping later requires rebuilding the index.
2. Which crate: `usearch` (supports FP16 natively, fast C++ core) vs. `hnsw_rs` (pure-ish Rust, slower but fewer FFI dependencies) vs. `instant-distance`?
3. When to introduce the Hopfield associative completion layer — Phase 3 (episodic memory) or Phase 4 (semantic memory)?
4. Deletion strategy: logical tombstones (simple) vs. periodic index rebuild (cleaner recall) for the warm tier.

---

## Sources

- Malkov & Yashunin, 2018. "Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs." *IEEE TPAMI*. arXiv:1603.09320.
- Ramsauer et al., 2020. "Hopfield Networks is All You Need." arXiv:2008.02217.
- HNSW memory formula: documented in multiple ANN library implementations; formula `1.1 × (vector_bytes + 8M)` is a practical approximation consistent with hnswlib source.
- ANN Benchmarks project: ann-benchmarks.com — reference for recall/latency Pareto fronts.
- USearch repository: github.com/unum-cloud/usearch
- `hnsw_rs` crate: crates.io/crates/hnsw_rs

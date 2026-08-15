# Vector retrieval

> **Contents:** ANN options for the recollection layer — flat search, HNSW (✓ committed, hnsw_rs), IVF-PQ, modern Hopfield networks. Memory/query latency by scale; persistence (dump/load, SMNWARM magic); warm==cold union property.
> **When to read:** Scaling the warm HNSW index; debugging slow restart times; considering a cold-tier ANN index.

Research note covering the recollection layer's retrieval mechanism: ANN indexes, exact search, associative-memory (Hopfield) approaches. All numbers are **engineering estimates**. See `architecture.md` "Memory hierarchy" — recollection is the cross-tier index: given a partial cue, it finds relevant memories anywhere in warm/cold and materializes them into warmer tiers.

---

## What we need

**Query:** cue vector in, top-k similar stored vectors + metadata (timestamps/region/episode ID) out. Latency: wake <5ms, sleep can be slower. Scale: thousands-to-millions initially, potentially hundreds of millions in cold storage long-term. **Writes** (sleep consolidation): insert new vectors, optionally delete/merge near-duplicates; batch inserts during sleep are fine, wake-time inserts should be avoided (deferred to sleep).

---

## Options

**Flat/exact search.** Brute-force inner-product/cosine, BLAS-accelerated. Zero overhead beyond the raw store (384-dim FP16: 768B/vector, 1M=768MB RAM). O(n×d)/query — ~10ms for 100K vectors @ dim=512 CPU (est.), GPU FAISS flat ~1ms for 1M; doesn't scale past ~10M at interactive latency. 100% recall, no build step, no hyperparameters, trivially appendable. Good Phase 1-2 starting point before scale demands ANN. Rust: `usearch`, `hnsw_rs` flat mode, custom BLAS over `ndarray`/`faer`.

**HNSW (Hierarchical Navigable Small Worlds).** Multi-layer proximity graph, coarse-to-fine query navigation — the standard high-quality ANN choice. Memory ≈ `1.1×(vector_bytes+8M)` per vector (M=max edges/node, typ. 16-64): 384-dim FP16, M=16 → ~986B/vector (~28% overhead); FP32 → ~1830B (~19%). 1M@FP16 ≈ 986MB, 10M ≈ 9.4GB (fits 56GB RAM), 100M ≈ 94GB (needs disk). Query O(log n), ~0.1-2ms/query at 95%+ recall (est. from SIFT1M benchmarks); build O(n log n), ~10-30min for 10M vectors (est.), not incremental by default (each insert is O(M log n) graph re-linking); insertion throughput ~10K-100K/s. High recall (~95-99% top-1) with sub-ms queries — Pareto-optimal for most workloads; ~2× memory overhead including the graph; poor fit for high-rate streaming inserts (fine for sleep-cycle batches); hyperparameters M/ef_construction/ef_search; no native deletion (tombstones or rebuild). Rust: `hnsw_rs`, `instant-distance`, `usearch` (best-maintained, native FP16). **Recommended for warm tier once vectors exceed ~500K** — flat search until then.

**IVF-Flat/IVF-PQ.** Cluster-based: `nlist` Voronoi cells at training time, search top-`nprobe` cells at query time. IVF-Flat ≈ 1× raw size (lower overhead than HNSW). IVF-PQ replaces storage with product-quantized codes (~8-32B/vector regardless of dim — 384-dim FP32 → 8B/vector is a 192× reduction, recall drops to ~80-90%). Faster than HNSW at massive scale (>100M) for matched recall; needs a one-time k-means training step (not available with little data); less dynamic than HNSW (new vectors outside known clusters degrade). **Right tool for the cold-tier archive past 10M vectors** — not needed until Phase 4+.

**Modern Hopfield networks.** Continuous Hopfield (Ramsauer et al. 2020) as attention-like memory — store patterns, retrieve via iterative update. Memory O(N×d), same as flat, no graph overhead; theoretical exponential capacity (~exp(d/2)) but practically limited by pattern separation. Single-step retrieval (softmax attention form) O(N×d)/query, same as flat but differentiable; multi-step iterative update costs 3-10× single-step. Fully differentiable (trainable end-to-end with recollection+cognitive layers as one system), no separate index build — but doesn't scale like HNSW at millions of vectors, and the exponential-capacity theorem is asymptotic (much lower in practice with noisy patterns). Natural fit for associative recall in the binding area (partial cue → full episode) — differentiability is the key advantage over HNSW. **Candidate for warm-tier associative completion**, complementary to (not a replacement for) HNSW's scale role — worth a Phase 3+ experiment.

---

## Hybrid: HNSW for scale, Hopfield for associative completion

Complementary: **HNSW** indexes warm+cold by region×concept, O(log n) lookup of the k nearest episodes. **Modern Hopfield**, inside the binding area, does associative completion over HNSW's top-k candidates (fills missing modalities, resolves partial cues). Separates the scale problem (HNSW, millions) from the completion problem (Hopfield, top-k fill-in).

---

## Budget analysis

Hardware: 56GB usable RAM (warm), ~14GB VRAM (hot), disk (cold).

| Scale | HNSW FP16 (384-dim, M=16) | HNSW FP32 (384-dim, M=16) | Fits in RAM? |
|---|---|---|---|
| 100K vectors | ~99MB | ~183MB | Yes |
| 1M vectors | ~986MB | ~1.83GB | Yes |
| 10M vectors | ~9.4GB | ~18.3GB | Yes (FP16) |
| 50M vectors | ~47GB | OOM | Tight FP16 |
| 100M vectors | ~94GB | OOM | No → disk |

At 384-dim FP16/M=16, warm HNSW fits up to ~50M vectors in 56GB RAM — memory isn't the binding constraint through Phase 4 (hundreds of thousands to low millions of concepts). Phase 5+ may need cold-tier IVF-PQ.

---

## Open questions → future ADR

1. Flat search (zero complexity) or HNSW from day one? Flat is simpler but swapping later means rebuilding.
2. Crate: `usearch` (native FP16, fast C++ core) vs. `hnsw_rs` (pure-ish Rust, slower, fewer FFI deps) vs. `instant-distance`?
3. When to introduce the Hopfield associative completion layer — Phase 3 (episodic) or Phase 4 (semantic)?
4. Deletion strategy: logical tombstones (simple) vs. periodic rebuild (cleaner recall) for warm?

---

## Sources

- Malkov & Yashunin, 2018. "Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs." *IEEE TPAMI*. arXiv:1603.09320.
- Ramsauer et al., 2020. "Hopfield Networks is All You Need." arXiv:2008.02217.
- HNSW memory formula: `1.1×(vector_bytes+8M)`, a practical approximation consistent with hnswlib source.
- ANN Benchmarks project: ann-benchmarks.com.
- USearch: github.com/unum-cloud/usearch. `hnsw_rs`: crates.io/crates/hnsw_rs.

# Growth mechanisms

> **Contents:** Cascade-Correlation, GNG, online cloning — mechanisms for adding units to a sparse region. Covers cost per unit added, growth trigger detection (gist recall rate < 0.5), and interaction with duty-cycle boosting.  
> **When to read:** Implementing or tuning growth/pruning (grow_region, prune_region); understanding why cascade-correlation-style growth was chosen.

Research note for somnus. Covers how the regional concept areas grow capacity when behavioral saturation is detected during REM-like sleep. All numbers are **engineering estimates**.

See `architecture.md` "Sleep stages" (REM-like: "Growth") and `continuous-learning.md` (cascade-correlation). The growth mechanism operationalizes vision commitment 5 (bottom-up growth + behavioral saturation).

---

## The problem: fixed capacity fails on open-ended streams

A fixed-size network trained on a continuous stream will either:
1. Overfit to recent experience (forgetting older concepts), or
2. Underfit by spreading capacity too thin across all seen concepts.

Growth solves this by adding representational capacity exactly when and where it is needed — triggered by measured failure to recall, local to the failing region.

The trigger condition from `architecture.md`: "If a meaningful share of recent episodes in a region fail gist-level recall, that region's semantic memory grows."

---

## Cascade-Correlation (primary mechanism)

Fahlman & Lebiere, 1990. The core idea:

1. Train the current network on available experience until it stalls (loss stops improving).
2. Train a pool of **candidate units** to maximize their correlation with the network's current *residual error* — the signal the existing network cannot explain.
3. Add the single best-correlating candidate unit to the network. Its input weights are **frozen** after insertion.
4. Retrain only the output weights to incorporate the new unit.
5. Repeat until residual is small enough.

**Why cascade-correlation for somnus:**
- No task IDs required. Growth is purely driven by prediction residual.
- Added units are trained on residual, not overwriting existing representations.
- Frozen input weights of added units preserve the sparse coding of previously learned patterns.
- Incremental: one unit at a time. Growth can be partial (add some units, check if residual improves, stop when good enough).

**Compute cost per unit added:**

| Step | Cost |
|------|------|
| Train candidates (C candidates, E epochs, batch size B) | O(C × E × B × d) forward + backward passes |
| Select best candidate (correlation with residual) | O(C × n_examples) |
| Add unit and retrain outputs | O(E × B × d) forward + backward passes |

Practical estimate for a small region (d=384 input, n_existing=500 units, C=20 candidates):
- Candidate training (20 candidates × 100 epochs × 1000 examples × 384 dim): ~minutes on GPU.
- Output weight fine-tuning: ~seconds.
- **Total per unit added: ~1–10 min on GPU (est.)** depending on dataset size and number of candidates.

Adding 10 units per sleep cycle: ~10–100 min. This is a coarse estimate; in practice cascade-correlation converges faster on well-structured data.

**Growth rate:** the system should add only as many units as needed per sleep cycle. A reasonable heuristic: grow until gist-recall rate in the region exceeds a threshold (e.g., 80%), then stop.

---

## Growing Neural Gas (GNG) — alternative for the recollection layer

Fritzke, 1994. A topology-preserving competitive learning network that grows by inserting nodes where the input distribution is not well-covered.

**Key mechanism:**
- Maintain a graph of nodes (prototype vectors) and edges (proximity relations).
- On each input: find the two nearest nodes (winner + runner-up); move winner toward input; accumulate error on winner; insert a new node between winner and its highest-error neighbor if error exceeds a threshold.
- Remove edges older than a max age.

**Compute cost:** O(n_nodes) per update for the nearest-node search (exact), plus edge maintenance. For n_nodes=1000: negligible. Scales quadratically for naive nearest-node search, but most GNG implementations bound n_nodes.

**Use case in somnus:** not the primary growth mechanism for regional concept areas (those use cascade-correlation for supervised-residual growth). GNG is interesting for the **recollection layer**: maintaining a topology over semantic memory that reflects which concepts are close in perceptual space, updating as new episodes arrive. The competitive Hebbian edge structure could drive the HNSW index incrementally.

**Tradeoff vs. cascade-correlation:** GNG is unsupervised (no labels or reconstruction error needed); cascade-correlation requires a reconstruction target (the residual). For Phase 1–2 where we may not have a good reconstruction signal yet, GNG is simpler.

---

## Online gradient-based expansion (alternative to cascade-correlation)

Instead of cascade-correlation's alternating freeze-train scheme, train a network where the architecture is dynamic: periodically clone an existing unit and add the clone to the layer with small random perturbation on its weights.

**Pros:** simpler to implement with standard autograd (no frozen-weight bookkeeping); compatible with `candle` or `burn` without custom ops.
**Cons:** no theoretical guarantee that new units learn novel features; the clone may duplicate existing representations. Requires careful regularization to force differentiation.

**Practical notes:**
- Works in practice for small growth rates (~5% of existing capacity per sleep cycle).
- Less efficient than cascade-correlation — many clones may not contribute meaningfully.
- A hybrid: clone + train on residual is essentially cascade-correlation with soft input-weight freezing.

---

## Interaction with sparse coding

Growth works naturally with k-WTA or SAE sparse codes:

- **k-WTA:** after adding n_new units, k must increase slightly to maintain the same sparsity ratio. If n_existing=500, k=25 (5%), and we add 50 units → n_new=550, k_new=28. Small adjustment.
- **SAE:** the new units extend the dictionary. Their weights are initialized near zero; the L1 penalty ensures they only activate when the existing units fail to reconstruct the input.
- **Dead unit risk:** new units in k-WTA are initially never active (not yet trained). Boosting must include them to give them learning signal.

---

## Growth trigger detection

The gist-level self-probe during REM-like sleep requires measuring per-region recall quality:

1. For each recent episode, run the region's encoder → decode via the semantic memory → compare decoded output to original encoder output at gist level (e.g., cosine similarity > threshold ≈ 0.7).
2. Compute the fraction of episodes that fail (similarity < threshold) — the **saturation rate**.
3. If saturation rate exceeds a target (e.g., > 20% fail), trigger growth for that region.

**Compute cost of self-probe:**
- Forward pass through encoder + decoder per episode: ~1–5 ms/episode (est., small networks).
- For 10K recent episodes: ~10–50 s. Acceptable within a sleep window.

---

## Unit initialization strategies

When adding a new unit via cascade-correlation:

- **Random init:** standard Kaiming or Xavier initialization. Works but requires more training epochs to correlate with residual.
- **Residual-guided init:** initialize the new unit's input weights as the input feature vector with the highest residual — a greedy start that reduces training epochs.
- **Prototype-based init:** initialize the new unit's weights as the centroid of episodes in the failing cluster (k-means on failing episodes). Semantically meaningful starting point; fewer training epochs needed.

Prototype-based initialization is the best option when episodes are clusterable (which they should be if regions have some structure). Cost: one k-means pass over failing episodes — O(n_fail × d × k_clusters × n_iter), typically <1 min.

---

## Budget analysis

Per sleep cycle growth for one region:

| Activity                          | Estimated cost          |
|-----------------------------------|-------------------------|
| Self-probe (10K recent episodes)  | ~10–50 s GPU            |
| Prototype init (100 failing epis) | <1 min CPU              |
| Candidate training (20 cands)     | ~5–30 min GPU           |
| Best-candidate selection          | ~1 min CPU              |
| Output weight retraining          | ~2–5 min GPU            |
| **Total per unit added**          | **~10–40 min GPU**      |

For 4 regions adding 1–3 units per sleep cycle: ~40–160 min total growth time. Sleep windows should budget for this — a 2–4 hour sleep cycle with growth is realistic.

Growth is by far the most expensive sleep operation. Compression and replay are seconds to minutes; growth is tens of minutes.

---

## Open questions → future ADR

1. Cascade-correlation vs. online cloning vs. SAE dictionary expansion: which to implement in Phase 3 (first growth-enabled sleep cycle)?
2. How many candidate units to train per growth event? Trade-off between selection quality (more candidates → better unit) and compute (more candidates → more training time).
3. Growth granularity: single unit at a time (cascade-correlation style) vs. batch of N units simultaneously? Batch is faster but may cause more interference.
4. How to handle growth across multiple regions simultaneously — sequential (one at a time) or parallel (different GPU streams)?

---

## Sources

- Fahlman & Lebiere, 1990. "The Cascade-Correlation Learning Architecture." NIPS.
- Fritzke, 1994. "A Growing Neural Gas Network Learns Topologies." NIPS.
- Yoon et al., 2018. "Lifelong Learning with Dynamically Expandable Networks." ICLR. (Task-aware growth, not used directly, but informative on growth regularization.)
- Rae et al., 2018. "Fast Parametric Learning with Activation Memorization." ICML. (Related: unit-level memory mechanisms.)

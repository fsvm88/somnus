# Growth mechanisms

> **Contents:** Cascade-Correlation, GNG, online cloning — mechanisms for adding units to a sparse region. Covers cost per unit added, growth trigger detection (gist recall rate < 0.5), and interaction with duty-cycle boosting.
> **When to read:** Implementing or tuning growth/pruning (grow_region, prune_region); understanding why cascade-correlation-style growth was chosen.

Research note covering how regional concept areas grow capacity on behavioral saturation, detected during REM-like sleep. All numbers are **engineering estimates**. See `architecture.md` "Sleep stages" (REM: "Growth") and `continuous-learning.md` — operationalizes vision commitment 5 (bottom-up growth + behavioral saturation).

---

## The problem: fixed capacity fails on open-ended streams

A fixed-size network either overfits to recent experience (forgetting older concepts) or underfits by spreading capacity too thin. Growth adds capacity exactly when and where needed, triggered by measured recall failure, local to the failing region. Trigger (`architecture.md`): "if a meaningful share of recent episodes in a region fail gist-level recall, that region's semantic memory grows."

---

## Cascade-Correlation (primary mechanism)

Fahlman & Lebiere, 1990: train the current network to stall → train candidate units to maximize correlation with the current residual error → add the best-correlating candidate, freeze its input weights → retrain only output weights → repeat until residual is small enough.

**Why for somnus:** no task IDs needed (purely residual-driven); new units train on residual, not overwriting existing representations; frozen input weights preserve prior sparse coding; incremental (partial growth, stop when good enough).

**Compute cost per unit** (est., d=384 input, n_existing=500, C=20 candidates): candidate training O(C×E×B×d) ≈ minutes on GPU; selection O(C×n_examples); output retrain seconds. **Total ~1-10min/unit**, 10 units/cycle ≈ 10-100min. Growth rate heuristic: grow until region gist-recall exceeds threshold (e.g. 80%), then stop.

## Growing Neural Gas (GNG) — alternative for the recollection layer

Fritzke, 1994: topology-preserving competitive learning, inserts nodes where input distribution isn't well-covered — graph of prototype nodes + proximity edges; on each input, move winner toward it, accumulate error, insert a node between winner and its highest-error neighbor past a threshold; prune old edges. O(n_nodes)/update. Not the primary regional growth mechanism (that's cascade-correlation, supervised-residual); interesting for the **recollection layer** — an unsupervised topology over semantic memory (no reconstruction target needed, simpler than CC where a good reconstruction signal doesn't yet exist).

## Online gradient-based expansion (alternative to CC)

Periodically clone an existing unit + small random perturbation, instead of CC's freeze-train scheme. Simpler with standard autograd, no frozen-weight bookkeeping. But no guarantee clones learn novel features (may duplicate); needs careful regularization. Works at small growth rates (~5%/cycle) but less efficient than CC — many clones may not contribute. Clone+train-on-residual is essentially CC with soft input-weight freezing.

## Interaction with sparse coding

**k-WTA:** k must increase slightly with n_units to preserve sparsity ratio (n=500,k=25→n=550,k=28). **SAE:** new units extend the dictionary, near-zero init, L1 penalty ensures they only activate when existing units fail. **Dead-unit risk:** new k-WTA units start never-active — boosting must include them for learning signal.

## Growth trigger detection

Per episode: encoder→decode via semantic memory→compare to original at gist level (cosine similarity > ~0.7 threshold). Saturation rate = fraction failing; trigger growth if it exceeds a target (e.g. >20%). Self-probe cost: ~1-5ms/episode, ~10-50s for 10K episodes — acceptable in a sleep window.

## Unit initialization strategies

**Random init** (Kaiming/Xavier) — simplest, needs more epochs to correlate with residual. **Residual-guided init** — seed with the highest-residual input feature vector, greedy start. **Prototype-based init** — seed with the centroid of failing episodes (k-means) — semantically meaningful, fewest epochs needed; best when episodes are clusterable. Cost: one k-means pass over failing episodes, typically <1min.

## Budget analysis (pre-implementation estimate)

| Activity | Estimated cost |
|---|---|
| Self-probe (10K episodes) | ~10-50s GPU |
| Prototype init (100 failing) | <1min CPU |
| Candidate training (20 cands) | ~5-30min GPU |
| Best-candidate selection | ~1min CPU |
| Output weight retraining | ~2-5min GPU |
| **Total per unit added** | **~10-40min GPU** |

4 regions × 1-3 units/cycle ≈ 40-160min total — a 2-4h sleep cycle with growth seemed realistic under this estimate. Growth was projected as by far the most expensive sleep op (replay/compression = seconds-minutes). **This did not hold** — see below.

---

## Empirical validation (2026-07-09 to 2026-07-11): actual implementation vs. the estimates above

**The implemented mechanism (`grow_region_seeded`, ADR 0024 §2c) diverges from the cascade-correlation sketch above.** Not candidate-pool training against residual — a **seeded weight-tensor resize**: existing matrices grow in place, new units are seeded (not correlation-trained), trigger is `within_session_gist_rate` with a 3-cycle cooldown backstop, not "train candidates, pick best." Consequence: the "~10-40min GPU/unit" estimate never applied — growth is closer to instantaneous (resize+seed); the observed ~20-27s/cycle wall-clock is dominated by perception/replay/NREM, not growth. The cost table above is kept as the pre-implementation estimate it was, not corrected — the ADR is the authoritative record of what shipped and why the simpler approach was chosen.

**Two characterization pilots** (not the full 10x campaign — see gaps.md), both `SESSIONS=15 CYCLES=5 CAPACITY=8192` (75 cycles), isolated dirs, from n=1024 baseline:

| | Pilot 1 (2026-07-09) | Pilot 2 (2026-07-11) |
|---|---|---|
| Corpus | 10 clips (5× h3h3, 5× shorts), looped | 145 clips (5× h3h3, 84× shorts, 56× podcasts), far less repetition |
| Growth events | 16 | 16 |
| Cadence | ~1/4.7 cycles, gate-limited (cooldown floor) | 4-7 cycle gaps, same gate-limited pattern |
| Final region sizes | `[2048,1984,1984,1984,1984]` (uneven) | `[2048,2048,2048,2048,2048]` (uniform) |
| `per_region_mse` trend | Improved with growth (region 0: 0.127→0.007-0.017) | Mean ~0.0135 (first 10)→~0.0039 (last 10) |
| Plateau? | No — one quiet session (s9), growth resumed s10c1 | No — matches pilot 1's cadence/MSE trend |
| Wall-clock | 21.8s/cycle avg, ~1.7GB disk | 25.9s/cycle avg (~19% slower, plausibly larger source files), 32min total |

**Why pilot 2:** pilot 1's tight 10-clip loop left open whether the observed cadence/improving MSE were genuine capacity/signal effects or memorization of the repeated 10 clips. Pilot 2 reruns the identical protocol against a corpus 14.5× larger and structurally diverse (podcast long-form speech added specifically for this, `download_podcasts.sh`) to pressure-test that risk before the full-scale campaign.

**Result: no plateau, cadence/quality trend match pilot 1 almost exactly** — confirms growth is gate-limited (bounded by cooldown), not signal-limited; corpus diversity doesn't change the behavior. This was the last open caveat before the full 10x campaign (~9,660 units/region, ~135 events, extrapolated ~633 cycles/~3.9h at the observed rate — see gaps.md for actual campaign status, which diverged further from this estimate).

---

## Open questions → future ADR

1. Cascade-correlation vs. online cloning vs. SAE dictionary expansion for the first growth-enabled sleep cycle?
2. How many candidate units per growth event — selection quality vs. compute tradeoff?
3. Growth granularity: single unit (CC-style) vs. batch of N — batch is faster but may cause more interference?
4. Multi-region growth: sequential or parallel (different GPU streams)?

---

## Sources

- Fahlman & Lebiere, 1990. "The Cascade-Correlation Learning Architecture." NIPS.
- Fritzke, 1994. "A Growing Neural Gas Network Learns Topologies." NIPS.
- Yoon et al., 2018. "Lifelong Learning with Dynamically Expandable Networks." ICLR.
- Rae et al., 2018. "Fast Parametric Learning with Activation Memorization." ICML.

---
status: accepted
date: 2026-06-14
deferred:
  - id: s5-startup-queryable-priors
    what: Query available RAM and benchmark HNSW latency at startup to auto-seed --max-warm-records and --recall-ef as bootstrap priors
    when: Trigger met (S1+S2 validated 2026-06-14); ready to implement
  - id: s6-adaptive-safety-ceilings
    what: Widen --max-wake-secs ceiling to 3× rolling mean wake duration; complement the wakefulness pressure homeostat
    when: Trigger met (S3 validated 2026-06-14); ready to implement
  - id: s7-throttle-adaptation
    what: Adapt --wake-reconstruct-interval and --gain-update-every from reconstruction quality trend; no clean signal exists yet
    when: After Phase 2 controllers stable and wake reconstruction quality metrics are reliable per-step
---

# ADR 0024 — Neuromodulatory self-regulation: derived signals replace fixed thresholds

**Status:** accepted  
**Date:** 2026-06-14

> **Decision:** Replace fixed-threshold CLI knobs with internal controllers driven by signals derived from the system's own runtime state. Introduce two layers — signal design (what to measure) and control law (how to regulate) — and stage implementation by signal cleanliness. Phase 1: adaptive salience gate + settle-rate modulation. Phase 2: wakefulness pressure homeostat. Growth controller explicitly deferred; prerequisite documented.  
> **Keywords:** neuromodulation, homeostasis, EWMA controller, salience gate, settle-rate, wakefulness pressure, prior-beta, growth controller, within-session gist, self-regulation, bootstrap prior

---

## Context

The system has accumulated ~40 CLI flags. Most started as one-time calibration
choices that calcified into permanent knobs:

- `--sleep-trigger-fraction 0.8` — chosen once for a 4096-capacity setup, never revisited.
- `--settle-gate 0.85` — arbitrary prior-cosine threshold; unknown whether 0.85
  is ever the right value once the model has converged.
- `WAKE_RECONSTRUCT_SALIENCE_THRESHOLD = 0.8` — hardcoded constant; unverifiable
  from first principles how well it tracks actual novelty in different content types.
- `--growth-cooldown 3` — a band-aid over contaminated signal (see Growth section).
- `--prior-beta 0.0` — manually enabled after convergence; no signal tells you *when*.

A brain-inspired system should carry its own regulatory state and self-adjust.
Crucially, this is a **control-systems argument**, not a biological one: a fixed
threshold is a constant-gain controller with no feedback; an EWMA-based threshold
is a first-order adaptive controller with well-understood stability properties.
The biological analogy (neuromodulators, arousal, homeostasis) is a useful
vocabulary, not a design warrant.

**Two architectural layers must be kept separate:**

1. **Signal design**: does this measurement track the thing we care about? Is it
   bounded, low-variance, defined at cold start, available within the control loop?
2. **Control law**: given a clean signal, what is the decision rule, setpoint,
   deadband, and stability guard?

Conflating them (designing controller and signal simultaneously) is what caused
the growth runaway: `gist_recall_rate` is a plausible controller input, but it
wasn't a clean signal — it collapsed due to cross-session NREM embedding drift,
independent of whether the region needed capacity.

### Long-clip campaign findings (2026-06-13)

5 sessions × 3 cycles, CAPACITY=4096, 49k warm records at end:

- c3 gist declined monotonically: 0.714 → 0.746 → 0.634 → 0.36 → 0.22.
- c2 gist collapsed by s4–s5 (0.0, 0.0).
- The decline tracks index size + cumulative model drift, not capacity need.

This confirms that `gist_recall_rate` is **not yet a clean signal for growth
control**. The cooldown (ADR 0023) rate-limits the damage; it does not fix the
underlying measurement.

---

## Three-bucket classification of all flags

Not every knob is a candidate for internal regulation. Forcing a learning signal
onto a structural or perf knob just adds noise. Three buckets:

### Bucket R — Regulation candidates

These gate learning or memory decisions at runtime; the optimal value changes
as the system state changes. Good candidates for derived-signal controllers.

| Flag | Default | Gates |
|---|---|---|
| `WAKE_RECONSTRUCT_SALIENCE_THRESHOLD` (const) | 0.8 | wake_reconstruct firing |
| `--nrem-salience-threshold` | 0 | NREM interrupt: pending novel embeddings |
| `--sleep-trigger-fraction` | 0.8 | sleep onset |
| `--max-wake-secs` | 600 | sleep onset (safety valve) |
| `--settle-gate` | 0.85 | settle firing (prior_cosine gate) |
| `--settle-every` | 1 | settle firing (throttle) |
| `--prior-beta` | 0.01 | top-down prior blend (base; S4 scales to 0.05) |
| `--perf-sleep-threshold` | 0.0 | early sleep on EWMA prior_cosine drop |
| `--growth-cooldown` + `GIST_GROWTH_THRESHOLD` | 3 / 0.85 | region growth |

### Bucket P — Bootstrap priors

These are set once at session start from architectural decisions. They become the
**cold-start prior** for the controller; once a controller is live, the CLI value
initialises the controller's estimate, not the threshold directly.

| Flag | Default | Meaning |
|---|---|---|
| `--episodic-capacity` | 4096 | hot buffer size |
| `--working-memory-size` | 0 | ring size |
| `--settle-steps` | 0 | whether settling is on at all |
| `--settle-beta` | 0.5 | settle blend weight (architectural) |
| `--replay-batch-size` | 128 | NREM batch size |
| `--language-oversample` | 0 | curriculum design |
| `--nrem-min-traces` | 0 | replay floor |
| `--nrem-loss-target` | 0.0 | NREM exit quality |

### Bucket S-query — Startup-queryable (will become bootstrap priors)

These can be seeded from the environment at process startup rather than hardcoded.
They do not regulate during a session; they become better-informed initial priors.

| Flag | Default | Startup query |
|---|---|---|
| `--max-warm-records` | 0 | `free_ram_bytes × 0.35 / RECORD_BYTES` (sysinfo crate) |
| `--recall-ef` | 64 | Benchmark 500-record synthetic HNSW; binary-search ef to hit <2 ms/query |
| `--max-cold-records` | 0 | `free_disk_bytes × 0.20 / RECORD_BYTES` |

These flags become `--init-max-warm-records`, `--init-recall-ef`, `--init-max-cold-records`
once the startup-query logic is implemented. Deferred — see front-matter.

### Bucket S-safety — Safety valves (adaptive ceiling, fixed floor)

Hard minima that can never be violated — they are the emergency stops for when internal
controllers misbehave. Once S3 (wakefulness pressure) is validated, the ceiling can
widen to 3× the historical average: `max_wake_secs_effective = max(floor, 3 × mean_wake_secs_rolling)`.
This makes them self-widening without removing the floor protection. Deferred — see front-matter.

| Flag | Floor | Adaptive ceiling (deferred) |
|---|---|---|
| `--max-wake-secs` | 600 s | `max(600, 3 × mean_wake_secs_rolling)` |
| `--max-sleep-secs` | 0 (disabled) | `max(0, 3 × mean_sleep_secs_rolling)` |
| `--max-cycles` | 0 | test harness — permanent manual |

### Bucket S-flags — Feature flags (all-on defaults)

These select which subsystems are active. All should default to `true` in normal
operation. The current `false` defaults are historical accidents from staged rollout.

| Flag | Current default | Correct default |
|---|---|---|
| `--enable-language` | false | **true** |
| `--enable-action-decoder` | false | **true** |
| `--wake-recon-async` | false | **true** (validated in A/B, 2026-06-12) |
| `--enable-gain-mod` | true | true ✓ |
| `--migraphx` | true | true ✓ |

Flipping these is a `main.rs` change alongside this ADR, not just documentation.
Goal: `somnus --inputs <file>` works without specifying subsystem flags.

### Bucket S-throttle — Throttle knobs (deferred — no clean signal yet)

These tune the frequency of operations whose optimal rate depends on signals that do
not yet exist or are not reliable. Listed in the front-matter `deferred:` section.

| Flag | Default | Potential future signal |
|---|---|---|
| `--wake-reconstruct-interval` | 5 | Wake MSE Δ/step trend — fire more when stagnant |
| `--gain-update-every` | 20 | Variance of `gain[i]` over N episodes — reduce if stable |
| `--top-k-recall` | 5 | Recall diversity ratio (unique recalled episodes / top-k) |

### Bucket S-ops — Structural / ops (permanent manual)

These trade throughput for accuracy, select hardware, or configure observability.
No internal learning signal can substitute for hardware knowledge or user intent.

| Flag | Default | Why permanent |
|---|---|---|
| `--video-stride` | 1 | Content-rate tradeoff (camera fps / encode budget) |
| `--replay-batch-size` | 128 | BLAS batch size (hardware-dependent) |
| `--max-tokens` | 128 | LM budget (user intent, not system state) |
| `--language-oversample` | 0 | Curriculum design (experimenter's call) |
| `--nrem-min-traces` | 0 | Safety replay floor |
| `--asr-min-chars` | 20 | Noise filter (content-dependent) |
| paths, session-id, ort-profiling, inputs | — | Ops / observability |

---

## Decision: implementation sequence

Signal cleanliness governs sequencing, not knob priority.

| Phase | Controller | Signal clean? | Implements |
|---|---|---|---|
| **1a** | Adaptive salience gate | Yes — L2 distance, well-observed | Replaces `WAKE_RECONSTRUCT_SALIENCE_THRESHOLD` + `--nrem-salience-threshold` |
| **1b** | Settle-rate modulation | Yes — prior_cosine already tracked | Replaces `--settle-gate` + (later) `--settle-every` |
| **2** | Wakefulness pressure homeostat | Mostly — safeties needed | Replaces `--sleep-trigger-fraction` + `--max-wake-secs` |
| **2b** | Prior-beta auto-enable | Yes — prior_cosine mean | Replaces `--prior-beta` manual enable |
| **2c** | Growth controller — within-session signal | Yes (c2+); noisy at c1 | `within_session_gist_rate` primary; cross-session gist fallback at c1; cooldown retained |

---

## Signal and controller specifications

### Controller principles applying to all signals

- **Cold-start**: until `N_WARMUP` samples exist, the controller outputs the
  current CLI default (every flag becomes the initial prior). This ensures
  `controller off = today's behavior` — a free A/B baseline.
- **Observability**: every controller's state variable (signal EWMA, threshold,
  controller output) goes into the per-cycle NDJSON. No dark state.
- **Knobs become priors**: the CLI flag for each replaced knob is renamed with a
  `--init-` prefix and kept for cold-start seeding. It is not deleted.

---

### S1 — Adaptive salience gate (Phase 1a)

**Replaces:** `WAKE_RECONSTRUCT_SALIENCE_THRESHOLD = 0.8` (hardcoded const) and
the salience-count component of `--nrem-salience-threshold`.

**What it controls:** Whether an episode's warm-recall L2 distance is "novel
enough" to trigger `wake_reconstruct_step` and to count as a pending novel
embedding during NREM.

**Signal source:** `ep_salience` — warm-recall L2 distance, already accumulated
per cycle as `mean_salience_wake` and `min_salience_wake` (ADR 0022).
This is per-episode, real-valued, bounded (L2 ≥ 0), and available without
any new instrumentation.

**Signal properties:**
- Clean: L2 distance tracks actual novelty independent of model drift.
- Bounded below by 0 (exact recall), open above (~3–5 in practice).
- Sentinel `f32::MAX` already filtered in existing accumulation code.
- Well-defined at cold start (warm is empty → all episodes are novel by default).

**Controller:**

EWMA of per-episode salience:

```
μ_s[t] = α · ep_salience[t] + (1 - α) · μ_s[t-1]     α = 0.01  (per-episode)
σ_s[t] = α · |ep_salience[t] - μ_s[t]| + (1 - α) · σ_s[t-1]
```

Threshold:

```
τ_s[t] = μ_s[t] + k · σ_s[t]     k = 0.5  (above-median novelty)
τ_s[t] = clamp(τ_s[t], τ_min=0.3, τ_max=3.0)
```

`k=0.5` targets episodes 1 SD above the mean (roughly the top third of novel
content). Increase `k` to filter more aggressively; decrease to trigger on more.

**Operating range:**
- `μ_s`: expected 0.8–1.5 in normal operation (from A/B campaign observations).
- `σ_s`: 0.2–0.6 typical.
- `τ_s`: 0.9–2.1 expected range; clamp prevents collapse or runaway.

**Cold-start fallback:** `τ_s = 0.8` (current const) until 100 episodes sampled.

**NREM interaction:** The existing `--nrem-salience-threshold` count gate uses
a fixed L2 threshold to count "novel" pending embeddings. With S1 live, that
fixed threshold is replaced by `τ_s`. The count gate itself (`--nrem-salience-threshold`)
remains a manual tunable (Bucket S) since it is a throughput knob, not a
learning signal.

**Observability:**
```json
"sal_ewma_mean": 1.12,
"sal_ewma_std": 0.31,
"sal_threshold": 1.27
```

**Implementation note:** `μ_s` and `σ_s` are per-session EWMA state, reloaded
from `snapshot/sal_ewma.bin` at session start. Same persistence pattern as
`growth_cooldown.bin`.

**What would invalidate S1:** If `ep_salience` distribution shifts dramatically
with modality mix (e.g., pure-language runs have near-zero warm L2 because
language tokens are dense in embedding space), the EWMA mean will collapse and
all language episodes will be above threshold. Monitor `sal_threshold` vs
modality breakdown; if language runs break it, stratify by modality.

---

### S2 — Settle-rate modulation (Phase 1b)

**Replaces:** `--settle-gate 0.85` (prior_cosine threshold), and eventually
`--settle-every N` (throttle).

**What it controls:** Settling fires a predictive-coding correction loop per
episode, blending the output-head prediction into the encoder input.
It is expensive (~2–3 ms/ep) and should fire more when the model is uncertain
and less when prediction is already accurate.

**Signal source:** `prior_cosine` — per-episode cosine similarity between the
output-head prediction and the actual input. Already accumulated as
`prior_cosine_mean` per region per cycle. Clean, bounded ∈ [-1, 1], available
every episode.

**Controller:**

EWMA of prior_cosine (across all regions, global mean):

```
C[t] = α · prior_cosine_global[t] + (1 - α) · C[t-1]     α = 0.005  (per-episode)
```

Gate: fire settling when `prior_cosine_global < C[t] - Δ` where Δ = 0.1.

Effect: settling fires when the *current* prediction quality is below the
*recent average*. When the model is well-calibrated (predictions routinely good),
settling fires only on anomalous episodes. When the model is uncertain throughout
(new domain, post-growth NREM drift), it fires frequently.

**Operating range:**
- `C[t]`: expected 0.3–0.7 after initial convergence (per session-state data).
- Gate Δ = 0.1 is roughly "fire when 10% worse than average" — empirically
  this fires ~30–50% of episodes when convergence is incomplete, <10% when
  fully converged.

**Cold-start fallback:** `C[t] = 0.85` (current gate value) until 200 episodes
sampled. Controller output ≡ current behavior.

**Settle-every modulation (follow-on):** Once C[t] is consistently > 0.6 (well
converged globally), `settle-every` auto-increases to 2 (halve frequency).
Gate: `C[t] > 0.6` for `>= 3 consecutive cycles` → double `settle-every` (max 4).
`C[t]` drops below 0.4 → reset to `settle-every = 1`. One step at a time.

**Observability:**
```json
"settle_cosine_ewma": 0.54,
"settle_gate_effective": 0.44,
"settle_every_effective": 1
```

**What would invalidate S2:** If prior_cosine has high cycle-to-cycle variance
(it does across regions: c1 low, c3 high), using a global mean may set the gate
too high for weak regions. Mitigation: monitor per-region prior_cosine means;
if one region consistently 0.2 below global mean, stratify.

---

### S3 — Wakefulness pressure homeostat (Phase 2)

**Replaces:** `--sleep-trigger-fraction 0.8`, `--max-wake-secs 600`.

**What it controls:** When to transition from wake to sleep.

**Current mechanism:** Sleep when episodic buffer hits 80% capacity (3276/4096
episodes), hard limit at 600s. Neither reflects how much the system has *learned*
from the wake period — it fires purely on volume/time.

**Signal source:** A scalar accumulator P(t) ∈ [0, ∞).

Accumulation (per episode during wake):
```
P += novelty_weight(ep_salience[t]) / baseline_rate
```
where `novelty_weight(s) = max(0, s - τ_s)` (salience above S1 threshold —
i.e. genuinely novel content adds more pressure than familiar content).
`baseline_rate` normalises so that 1.0 added per episode when salience = 2×τ_s.

Dissipation (per NREM cycle):
```
P -= consolidation_quality(gist_recall_rate, nrem_loss_ratio)
```
where `consolidation_quality` ∈ [0, 0.5] per cycle.

Sleep fires when `P >= P_threshold = 1.0`.

**Setpoint:** The target is that a typical wake session (varied novelty) triggers
sleep after ~3000–4000 episodes. `baseline_rate` is tuned accordingly.

**Cold-start fallback:** Until 500 wake episodes have been observed (no reliable
estimate of novelty rate), fall back to the episodic-capacity fill trigger at
`--init-sleep-trigger-fraction` (default 0.8). P accumulates in parallel so the
controller is ready when warmup ends.

**Stability guards:**
- Hard minimum wake duration: 120 episodes (prevents sleep-on-startup).
- Hard maximum: `--max-wake-secs` remains as a safety valve (not removed).
- P is clamped to [0, 3.0] to prevent unbounded runaway (e.g. if all content
  is extremely novel indefinitely).

**Interaction with `--perf-sleep-threshold`:** The prior-cosine early-sleep
trigger fires when EWMA prior_cosine drops below threshold. With S3 live, this
becomes a P boost instead: a sudden prior_cosine drop adds a large impulse to P
(model destabilised by drift → pressure to consolidate sooner). The
`--perf-sleep-threshold` CLI becomes `--perf-sleep-pressure-boost` (a scalar
multiplier on the P impulse size).

**Observability:**
```json
"wakefulness_pressure": 0.73,
"wake_trigger": "pressure",
"wake_episodes_until_pressure_sleep": 412
```

**What would invalidate S3:** If novelty_weight is miscalibrated (all episodes
above τ_s → P accumulates too fast → too-frequent short sleeps), the NREM
consolidation signal is weak (gist contaminated), and P cannot dissipate. The
accumulator must be tunable via `baseline_rate` until the signal is reliable.
If it is unreliable, the capacity trigger remains active in parallel as fallback.

---

### S4 — Prior-beta scale-up controller (Phase 2b)

**Replaces:** `--prior-beta 0.0` manual enable.

**What it controls:** Weight β of the top-down prior blend during wake.
Previously set manually to 0 until "the model seemed converged".

**Design note (2026-06-14 amendment):** The original design used a toggle
(0.0 → 0.05), but `blend_with_prior` returns `None` cosine when
`prior_beta <= 0.0`, so the ring buffer never fills at cold start —
chicken-and-egg. The fix is a small non-zero base (`--prior-beta 0.01`):
prior cosines are always measured, the blend is negligible when predictions
are poor (1% of noise), and S4 scales up to 0.05 when predictions mature.

**Signal source:** `prior_cosine_mean` (global, rolling). Already emitted
per cycle.

**Controller:** 3-cycle ring buffer of per-cycle global prior_cosine_mean.

```
Scale up   (β = 0.05):  all 3 ring values > 0.65
Scale down (β = base):  any ring value    < 0.40
```

Base β is `--prior-beta` (default 0.01). Hysteresis gap (0.40–0.65)
prevents rapid oscillation. Only engages when `--prior-beta < 0.05`.

**Cold-start:** β = `--prior-beta` (0.01). Prior cosines are measured from
episode 1 (output heads produce predictions after the first episode).

**Observability:**
```json
"prior_cosine_mean": [0.71, 0.89, 0.62, 0.84, 0.33],
"prior_beta_effective": 0.05
```

**What would invalidate S4:** If prior_cosine stays below 0.65 indefinitely
due to model architecture or content properties (not lack of convergence),
scale-up never fires. Mitigation: expose `--prior-beta-threshold` as an
override. If the base 0.01 blend measurably degrades learning, reduce to
0.001 or add a flag.

---

### Growth controller — within-session signal (Phase 2c)

**Prerequisite resolved (2026-06-14):** within-session gist probe landed in
commit 7214d16. `SemanticStore::records_since(session_start_secs)` filters warm
by `timestamp_secs`, giving a clean per-session subset. `within_session_gist_rate`
is emitted in every per-cycle NDJSON record.

**Why the prerequisite was non-negotiable:** The long-clip campaign (2026-06-13)
showed c3 gist declining 0.71 → 0.22 across 5 sessions as warm grew to 49k
records, independent of whether any region needed capacity. Regulating growth on
`gist_recall_rate` means the controller reads "always poor" and never stops
growing. `within_session_gist_rate` removes this: only warm records from the
current session are used for both the k-means centroids and the nearest-neighbour
search, so drift from prior-session NREM weight updates cannot contaminate the
signal.

**Implemented signal priority (sleep.rs):**

```
quality_poor = match (within_session_gist_rate, gist_recall_rate) {
    (Some(ws), _)   => ws < GIST_GROWTH_THRESHOLD,   // c2+: clean within-session
    (None, Some(_)) => gist_poor,                      // c1: noisy cross-session fallback
    (None, None)    => per_region_mse[i] > GROWTH_THRESHOLD, // S1C1: no warm yet
}
```

`within_session_gist_rate` is always `None` at cycle 1 of every session (nothing
has spilled into warm from the current session yet). The cross-session gist
fallback at c1 is noisier, which is why the cooldown is retained as a backstop.

**Pruning gate** also switched to `within_session_gist_rate.or(gist_recall_rate)`.
`None` (c1) → no pruning, which is conservative and correct.

**Pressure dissipation** in main.rs uses the same `ws.or(gist)` preference,
so NREM consolidation quality drives wakefulness recovery with the cleaner signal
where available.

**Cooldown retained:** The target design in this ADR specified a plasticity
depletion/recovery model. The simpler approach — keep the existing cooldown but
switch to the cleaner signal — was chosen first because:
- c1 (ws=None) is still cross-session-gist-driven and remains noisy.
- The cooldown provides a backstop against spurious c1 growth cascades.
- The plasticity model adds complexity without measurable benefit until the c1
  signal is also solved (requires either tagging episodes with session IDs or
  accumulating within-session episodes in the episodic buffer across cycles).

**Observability:**
```json
"gist_recall_rate": 0.797,
"within_session_gist_rate": 0.906
```

---

## Full mapping table

| Knob (flag, default) | Decision gated | Replacement signal | Signal source | Controller / law | Setpoint / target | Operating range (signal) | Cold-start fallback | Stability guard | NDJSON field(s) | Expected contribution | Implementation phase |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `WAKE_RECONSTRUCT_SALIENCE_THRESHOLD = 0.8` | wake_reconstruct firing | Per-episode L2 salience EWMA (mean + std) | warm-recall L2, per wake episode | Threshold = μ_s + 0.5·σ_s; EWMA α=0.01 | ~top-40th percentile of novelty | μ_s ∈ [0.5, 2.0]; τ_s clamped [0.3, 3.0] | 0.8 (const) until N=100 | Clamp τ_s; sentinel filter | `sal_ewma_mean`, `sal_ewma_std`, `sal_threshold` | Adapts to content novelty; prevents calibration drift across clip types | **1a** |
| `--nrem-salience-threshold 0` | NREM interrupt count gate | Uses τ_s from S1 for the "is novel" classification | derived from S1 | Count pending episodes with L2 > τ_s; interrupt when count > N | — | — | 0 (no interrupt) | N remains manual (throughput knob) | `nrem_interrupt_count` | Correct novelty count for interrupt; N stays manual | **1a** (S1 prerequisite) |
| `--settle-gate 0.85` | settle firing (per episode) | Rolling prior_cosine EWMA C[t] | prior_cosine, per episode | Fire when prior_cosine < C[t] - 0.1; EWMA α=0.005 | Fires on below-average episodes | C[t] ∈ [0.0, 1.0]; expected 0.3–0.7 | C[t] = 0.85 until N=200 | Δ floor 0.05 | `settle_cosine_ewma`, `settle_gate_effective` | Settle fires when model is uncertain; idle when well-calibrated | **1b** |
| `--settle-every 1` | settle throttle | C[t] convergence trend (S2 EWMA) | derived from S2 | Increase settle-every by 1 when C[t] > 0.6 for 3 cycles; reset to 1 when C[t] < 0.4 | Auto-scale with convergence | settle_every ∈ [1, 4] | 1 (current) | Step-by-step; max 4 | `settle_every_effective` | Reduces settle overhead as model converges | **1b** (after S2 stable) |
| `--sleep-trigger-fraction 0.8` | sleep onset | Wakefulness pressure P (novelty accumulator) | ep_salience (above τ_s) per episode + NREM dissipation | P += novelty_weight/baseline_rate per ep; sleep when P ≥ 1.0 | P = 1.0 (target: ~3500 eps per sleep) | P ∈ [0, 3.0] clamped | Capacity fill at 0.8 until N=500 | Hard min 120 eps before sleep; P clamped | `wakefulness_pressure`, `wake_trigger` | Sleep driven by actual novelty load; long-familiar content doesn't trigger sleep | **2** |
| `--max-wake-secs 600` | sleep onset (safety valve) | — | — | Stays as hard wall-time safety valve | — | — | — | Cannot be replaced: safety is structural | `wake_secs` (existing) | No change; safety valve kept | stays manual (Bucket S) |
| `--perf-sleep-threshold 0.0` | early sleep on prior_cosine drop | P impulse on prior_cosine collapse | prior_cosine EWMA (S2) | Prior_cosine EWMA drop > 0.2 → P += 0.5 (boost) | — | impulse [0, 0.5] | 0.0 (disabled, no boost) | Boost capped at 0.5/cycle | `perf_sleep_boost` | Destabilisation triggers consolidation faster | **2** (requires S3 live) |
| `--prior-beta 0.01` | top-down prior blend weight | Rolling prior_cosine mean across cycles (3-cycle ring) | prior_cosine_mean, per cycle | Scale up β→0.05 when all 3 ring values > 0.65; scale down to base when any < 0.40 | Hysteresis band [0.40, 0.65] | P_c ∈ [0, 1] | 0.01 (base; cosines measured from ep 1) | Hysteresis; only engages when base < 0.05 | `prior_beta_effective`, `prior_cosine_mean` | Removes manual convergence judgement; base 0.01 avoids chicken-and-egg | **2b** |
| `--growth-cooldown 3` + `GIST_GROWTH_THRESHOLD` | region capacity growth | `within_session_gist_rate` primary; cross-session gist at c1; MSE at S1C1 | warm records filtered by session_start_secs | `ws < 0.85` → grow; `None` at c1 falls back to gist; cooldown backstop retained | — | ws ∈ [0, 1]; None at c1 of every session | cooldown=3 until c1 noise resolved | cooldown; None→no-prune at c1 | `gist_recall_rate`, `within_session_gist_rate` | Eliminates cross-session drift false positives (validated: S5C2 suppressed) | **2c** |

---

## Implementation notes

1. **Knob renaming convention:** Replaced knob `--foo` becomes `--init-foo` with
   identical default. Old `--foo` is kept as a deprecated alias for one release
   cycle, then removed. The `--init-` prefix makes explicit that it seeds the
   controller, not the threshold directly.

2. **Persistence:** Each controller's EWMA state is persisted in
   `snapshot/<controller>.bin` using the same u64-LE-bytes pattern as
   `growth_cooldown.bin`. Multiple EWMAs serialised as a flat `[f32; N]` array.

3. **A/B validation:** Each phase 1 controller is validated with a small
   fixed-seed run (2 sessions × 3 cycles) comparing `controller on` vs `controller off`
   (i.e. `--init-foo` only). Metrics of interest: `settle_every_effective`,
   `sal_threshold`, and overall `mean_output_mse_wake` trajectory.

4. **Observability discipline:** Every new controller state variable is emitted
   in the per-cycle NDJSON before the ADR is considered implemented. No dark
   state — per ADR 0022 discipline.

5. **Biology is analogy.** "Wakefulness pressure" is a scalar accumulator with
   a threshold; calling it neuromodulatory is a mnemonic. The control-systems
   justification is sufficient: it is a leaky-integrator homeostat with a known
   setpoint and stability bounds. No claim is made about neuronal fidelity.

---

## What would invalidate this decision

- If EWMA controllers prove slower to stabilise than fixed thresholds on
  short clips (< 500 episodes per session), the cold-start fallback ensures no
  regression, but the phase-1 controllers may need a stronger initialisation.
- If prior_cosine remains persistently low across all domains and modalities
  (not convergence-dependent), S2 and S4 have no useful dynamic range to exploit —
  they would always fire at the cold-start rate. At that point, fix the prior
  cosine signal first.
- If the wakefulness pressure baseline_rate is hard to calibrate across clip
  types (novelty distribution varies 10×), S3 may need per-clip-type priors.
  Fall back to capacity trigger and revisit.
- Once the within-session gist probe lands and growth controller is implemented,
  `--growth-cooldown` and `GIST_GROWTH_THRESHOLD` should be retired. If gist
  stays contaminated after the within-session fix, the growth controller must be
  redesigned around a different capacity signal (e.g. per-region MSE trend).

---

## Amendment — S4 chicken-and-egg fix + full validation (2026-06-14)

### Bug found during confirmatory run

The original S4 design gated on `args.prior_beta == 0.0` and started β at 0.
`blend_with_prior` in `somnus-perception` returns `(values, None)` cosine when
`prior_beta <= 0.0`. With prior cosines always `None`, the S4 ring buffer never
filled, S4 never triggered, and S2's settle EWMA (which tracks prior cosines
via `episode_cosine_mean`) also froze at cold start (0.85) indefinitely.

### Fix

Changed `--prior-beta` default from `0.0` to `0.01`. This is self-calibrating:
when predictions are poor, 1% blend barely touches the input; as predictions
improve, the same β contributes directionally. Prior cosines are always measured
from episode 1 — no change to `blend_with_prior` required. S4 guard changed
from `args.prior_beta == 0.0` to `args.prior_beta < 0.05`; on scale-down, β
resets to `args.prior_beta` (base) rather than 0.0. Campaign script
`PRIOR_BETA` env var defaults to `0.01`.

### Confirmatory run results (2s × 3c, fresh memory, CAPACITY=4096)

| cycle | session | trigger | wake_eps | sal_threshold | settle_gate_dynamic | wakefulness_pressure | prior_beta_effective | prior_cosine (r0–r4) |
|-------|---------|---------|----------|--------------|--------------------|--------------------|---------------------|----------------------|
| 1 | 1 | high_water | 3276 | 0.800 | 0.638 | 0.000 | 0.010 | [0.517, 0.603, 0.496, 0.680, 0.292] |
| 2 | 1 | pressure | 500 | 1.023 | 0.576 | 2.796 | 0.010 | [0.653, 0.736, 0.506, 0.699, 0.209] |
| 3 | 1 | pressure | 120 | 0.825 | 0.618 | 2.611 | 0.010 | [null, null, 0.667, 0.836, null] |
| 1 | 2 | pressure | 287 | 0.345 | 0.697 | 0.732 | 0.010 | [0.713, 0.889, 0.620, 0.843, 0.328] |
| 2 | 2 | pressure | 120 | 0.424 | 0.567 | 2.504 | 0.010 | [0.733, 0.935, 0.526, 0.739, null] |
| 3 | 2 | pressure | 120 | 0.519 | 0.593 | 2.645 | **0.050** | [null, null, 0.628, 0.770, null] |

**S1** ✅ `sal_threshold` adapts away from 0.8 cold-start from cycle 2 onward.  
**S2** ✅ `settle_gate_dynamic` live and tracking ~0.10 below `settle_ewma_mean`.  
**S3** ✅ Wakefulness pressure triggers sleep from cycle 2; 120-ep minimum guard active.  
**S4** ✅ `prior_beta_effective` scales up to 0.05 at S2C3 (3 consecutive cycles with active-region cosines > 0.65).

---

## Amendment — Growth controller implemented (2026-06-14)

### Within-session gist probe (prerequisite, commit 7214d16)

`SemanticStore::records_since(since_secs: f64)` filters warm by timestamp.
`run_sleep_cycle` takes `session_start_secs: f64` (captured at process start via
`wall_secs(&SystemTime::now())`). Within-session probe runs k-means + brute-force
NN on the filtered subset only — both centroids and neighbor pool are from
within-session records, so cross-session drift affects neither.

`within_session_gist_rate` is always `None` at c1 (nothing spilled yet this
session). The chart `gist_quality.png` shows both metrics over time.

### Validation run: 5s × 3c, CAPACITY=4096, fresh memory

| s | c | gist_recall | ws_gist | grew | notes |
|---|---|-------------|---------|------|-------|
| 1 | 1 | None | None | **Y** | S1C1: MSE-driven (warm empty) |
| 1 | 2 | 0.646 | 0.555 | — | cooldown |
| 1 | 3 | 0.930 | 0.969 | — | cooldown |
| 2 | 1 | 0.717 | None | — | cooldown |
| 2 | 2 | 0.852 | 0.781 | — | cooldown |
| 2 | 3 | 0.508 | 0.805 | **Y** | both < 0.85; growth real |
| 3 | 1 | 0.961 | None | — | gist high; no growth needed |
| 3 | 2 | 0.953 | 0.875 | — | |
| 3 | 3 | 0.750 | 0.883 | — | |
| 4 | 1 | 0.798 | None | **Y** | c1 gist fallback; MSE still elevated |
| 4 | 2 | 0.844 | 0.852 | — | cooldown |
| 4 | 3 | 0.984 | 1.000 | — | cooldown |
| 5 | 1 | 0.781 | None | — | cooldown |
| 5 | 2 | 0.797 | **0.906** | **Y (old)** | ← false positive eliminated by ws |
| 5 | 3 | 0.938 | 0.867 | — | |

**Key finding — S5C2:** Cross-session gist=0.797 fell below the 0.85 threshold
and triggered growth on all 5 regions. Within-session gist=0.906 correctly
identifies the quality as good. With ws-primary, this growth event is suppressed.

**S2C3** confirms the signal is not always conservative: gist=0.508, ws=0.805 —
both below threshold, both agree growth is needed. ws-primary still grows.

**S4C1** (c1 fallback): gist=0.798, ws=None → gist fallback fires, grew. Per-region
MSE at S4C1 was elevated (speech 0.435, visual 0.385), so growth was legitimate.
The cooldown absorbs the remaining c1 risk.

### Growth controller change (commit e30916c)

- `sleep.rs`: `quality_poor` uses `match (within_session_gist_rate, gist_recall_rate)`
- `sleep.rs`: `prune_ok` uses `within_session_gist_rate.or(gist_recall_rate)` — None → no pruning at c1
- `main.rs`: wakefulness pressure dissipation uses `ws.or(gist).unwrap_or(0.0)`
- Cooldown retained; plasticity model deferred pending c1 signal improvement

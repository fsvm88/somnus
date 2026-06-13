# ADR 0024 — Neuromodulatory self-regulation: derived signals replace fixed thresholds

**Status:** accepted  
**Date:** 2026-06-13

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
| `--prior-beta` | 0.0 | top-down prior blend |
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

### Bucket S — Structural / perf (stay manual)

These trade memory for accuracy, or time for throughput. No learning signal
should change them at runtime.

| Flag | Default | Meaning |
|---|---|---|
| `--recall-ef` | 64 | HNSW ef parameter |
| `--video-stride` | 1 | encoding throughput |
| `--max-warm-records` | 0 | RAM bound |
| `--max-cold-records` | 0 | RAM bound |
| `--max-sleep-secs` | 0 | wall-time safety valve |
| `--max-cycles` | 0 | test harness |
| `--wake-reconstruct-interval` | 5 | reconstruct throttle |
| `--wake-recon-async` | false | parallelism |
| `--gain-update-every` | 20 | perf |
| `--max-tokens` | 128 | LM budget |
| `--asr-min-chars` | 20 | noise filter |

---

## Decision: implementation sequence

Signal cleanliness governs sequencing, not knob priority.

| Phase | Controller | Signal clean? | Implements |
|---|---|---|---|
| **1a** | Adaptive salience gate | Yes — L2 distance, well-observed | Replaces `WAKE_RECONSTRUCT_SALIENCE_THRESHOLD` + `--nrem-salience-threshold` |
| **1b** | Settle-rate modulation | Yes — prior_cosine already tracked | Replaces `--settle-gate` + (later) `--settle-every` |
| **2** | Wakefulness pressure homeostat | Mostly — safeties needed | Replaces `--sleep-trigger-fraction` + `--max-wake-secs` |
| **2b** | Prior-beta auto-enable | Yes — prior_cosine mean | Replaces `--prior-beta` manual enable |
| **deferred** | Growth controller | No — gist contaminated | Requires within-session gist probe first |

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

### S4 — Prior-beta auto-enable (Phase 2b)

**Replaces:** `--prior-beta 0.0` manual enable.

**What it controls:** Weight β of the top-down prior blend during wake.
Currently enabled by hand after the model appears converged.

**Signal source:** `prior_cosine_mean` (global, rolling). Already emitted
per cycle.

**Controller:** EWMA-smoothed rolling mean P_c over the last 3 sleep cycles.

```
Enable (β = 0.05):  P_c > 0.65  for >= 3 consecutive cycles
Disable (β = 0.0):  P_c < 0.40
```

Hysteresis gap (0.40–0.65) prevents rapid toggling.

**Cold-start:** Start disabled (β = 0). Current behavior.

**Observability:**
```json
"prior_cosine_cycle_ewma": 0.61,
"prior_beta_effective": 0.05
```

**What would invalidate S4:** If prior_cosine stays below 0.65 indefinitely
due to model architecture or content properties (not lack of convergence),
auto-enable never fires. Mitigation: expose `--prior-beta-threshold` as an
override.

---

### Growth controller — Deferred

**Prerequisite (not optional):** within-session gist probe — restrict the REM
gist probe to only episodes encoded *in the current session's wake phase* (not
the full warm store). This removes the cross-session NREM embedding drift that
contaminates the current `gist_recall_rate` signal.

**Why deferred:** The long-clip campaign (2026-06-13) showed c3 gist declining
0.71 → 0.22 across 5 sessions as warm grew to 49k records, independent of
whether the region actually needed capacity. Regulating growth on this signal
means the controller reads "always low" and never stops growing — same runaway
as before the cooldown, but with an EWMA wrapper.

**Current safeguard:** `--growth-cooldown 3` (ADR 0023). This stays in place
until the within-session probe lands.

**Target design (post-prerequisite):**

Signal: `within_session_gist_rate` — gist recall restricted to episodes from the
current wake session. Expected properties: low variance (bounded by session
length, not index age), no cross-session drift, defined even at c1.

Controller: plasticity depletion/recovery model.

```
plasticity[t] = clamp(plasticity[t-1] - cost_per_growth + recovery_per_cycle, 0, 1.0)
```

Grow when `within_session_gist_rate < GIST_GROWTH_THRESHOLD` AND
`plasticity[t] > plasticity_floor = 0.3`.
After growth, `plasticity` decreases by `growth_cost = 0.7`, dissipating
linearly at `recovery_per_cycle = 0.25`. Effective refractory period: ~2.8
cycles (matching current cooldown) but with rate governed by internal state.

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
| `--prior-beta 0.0` | top-down prior blend weight | Rolling prior_cosine mean across cycles | prior_cosine_mean, per cycle | Enable β=0.05 when 3-cycle mean > 0.65; disable when < 0.40 | Hysteresis band [0.40, 0.65] | P_c ∈ [0, 1] | 0.0 (disabled) | Hysteresis prevents toggling | `prior_beta_effective`, `prior_cosine_cycle_ewma` | Removes manual convergence judgement | **2b** |
| `--growth-cooldown 3` + `GIST_GROWTH_THRESHOLD` | region capacity growth | within-session gist rate | gist restricted to current-session warm entries | Plasticity depletion/recovery model | — | plasticity ∈ [0, 1] | Current cooldown stays | plasticity_floor = 0.3 | `plasticity`, `within_session_gist` | Growth signal clean; runaway fixed at root | **deferred** |

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

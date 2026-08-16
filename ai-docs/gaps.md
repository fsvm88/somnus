# Gaps, candidates, and open work

> **Contents:** Open, undecided items only — algorithm/encoder/LLM upgrade candidates awaiting a decision, and known incompleteness with no chosen fix yet. Items that reached a decision (accepted/deployed/adopted, or rejected/not-pursued/null-result) live in [`engineering-log.md`](engineering-log.md), verbatim, under the same section numbers. Pruned to open-only 2026-08-15 (was §0-§37; see `engineering-log.md`'s header for what moved and why). §38 (Qwen3 KV-cache generation speedup) resolved and removed from this file 2026-08-15; full record at `engineering-log.md` §38. §41 (change-gated video encoder) rejected as a null-result and removed 2026-08-16; full record at `engineering-log.md` §43.
> **When to read:** Picking the next item to try, or checking whether something is still undecided.
> **Numbering:** section numbers are stable identifiers, not sequential — they're cited by ADRs, `architecture.md`, `roadmap.md`, and `session-state.md`. Gaps in the sequence (e.g. no §1-§3, §5-§6) mean that number's item moved to `engineering-log.md`, not that it never existed. New items get the next unused number across both files (next: §45).

---

## 4. Algorithm: LISTA Top-K unrolled encoder

**Source:** arXiv:2511.10575, Nov 2025. 3-5 unrolled ISTA iterations (learned step size + straight-through Top-K gate) instead of single-shot k-WTA — converges to sparser, better-reconstructed codes at the same k. All-5-regions change. Each step is matmul+bias+TopK+multiply — standard ONNX ops, static graph, no CUDA. **Effort: Medium.** Deferred.

---

## 7. Visual encoder: SigLIP2-ViT-B/16

**Source:** Google DeepMind, Feb 2025, arXiv:2502.14786. 86M params, ImageNet zero-shot 79.1% vs SigLIP 76.7%. Language-grounded (CLIP-style) vs DINOv2's supervised-probe-only features. 768-dim (up from 384) — cold-starts visual region k-WTA input + binding projection. MIGraphX likely fine (standard ViT attention). This is deferred item K (Phase 7+) — raise when cross-modal grounding (item N) becomes a target. **Effort: Medium.**

---

## 10. Language encoder: dedicated text embedding model

Current input: Qwen2.5-0.5B embed table (896-dim mean-pooled) — **stale**: production LM is Qwen3 (1024-dim) since ADR 0036; this item's own migration note already anticipated that. Key constraint: `--lang-use-lm`'s LM hidden states are the training signal behind the 4× p_lang improvement — switching the input encoder and switching the LM dim must move together.

| Model | Params | Dim | ONNX | MTEB | Notes |
|---|---|---|---|---|---|
| nomic-embed-text-v1.5 | 137M | 768 (Matryoshka) | confirmed | 62.39 | 8192-token context |
| modernbert-embed-base | 149M | 768 (Matryoshka→256) | confirmed (raw) | — | ModernBERT backbone |
| BGE-small-en-v1.5 | 33M | 384 | confirmed | low-60s | smallest |

**Recommended:** only migrate alongside a future LM upgrade. Deferred.

---

## 11. LLM: Gemma 3 1B IT

**Source:** Google DeepMind, March 2025, `onnx-community/gemma-3-1b-it-ONNX`. 1B, better benchmarks than Qwen2.5-0.5B (current production LM is Qwen3-0.6B, ADR 0036 — still a plausible upgrade target, comparison not redone against it). **Caveat:** community ONNX uses onnxruntime-genai's opaque KV-cache, not ADR 0034's static prefill/decode split — reproducing that split is high effort; verify ScatterElements absence before committing to GPU. **Effort: Medium-High.** Lower priority than closing §38/§39 on the current LM. Deferred.

---

## 15. Crate hygiene

- **usearch 2.25.3** (vs `hnsw_rs`): built-in INT8/INT4 quantization, AVX-512/NEON SIMD, concurrent inserts, recall@10 at parity with hnswlib/FAISS. Migration cost: rewrite `somnus-memory`'s HNSW layer. Raise when warm-tier footprint becomes a concern.
- **burn-ndarray**: deprecation path toward `burn-flex` in 0.22 (~late summer 2026). Watch, no action yet.
- **ffmpeg-the-third**: actively maintained `ffmpeg-next` fork, may have surpassed it. Check next time the media pipeline is touched.

(Dead-dep removal — candle-core/candle-nn — is done; see `engineering-log.md` §15's prior text in git history if needed.)

---

## 23. Future: native Rust ONNX graph construction 🔮 SCOPED, NOT STARTED

Raised 2026-07-12 while fixing an export-script bug (`engineering-log.md` §22). Would eliminate the Python3+onnx+onnxruntime subprocess round-trip and solve dynamic shapes for free (Rust already knows every shape exactly). **Why not now:** no mature "build ONNX graphs from Rust" crate — would mean hand-constructing `ModelProto`/`GraphProto`/`NodeProto` replicating everything the Python script does (GRU cell weight transpose, multi-head attention, the MIGraphX ScatterElements two-graph-split workaround). Real correctness risk, genuinely open-ended — a dedicated session, not a fast-follow. **Wins if done:** removes the Python runtime dependency, the per-cycle subprocess+export overhead, and the weights-manifest.json round-trip. **Trigger:** none set — raise when export-script maintenance recurs or profiling shows the subprocess overhead is itself meaningful.

---

## §33 — BindingArea weights never trained — decision (b) implemented, three pilots FAIL, no path chosen

**Confirmed 2026-08-14:** the binding area's ~1.72M params (5 region projections, W_q/W_k/W_v, GRU, temporal_proj) have never received a gradient — since ADR 0008. All training paths (`replay_step*`, `wake_reconstruct_step*`, `fused_replay_wake_step_batch`, ADR 0012 proj_loss) consume episodes as stored `Tensor::from_data` — the graph stops at the episode boundary; only inference call sites (`try_bind`, `settle_episode`, `binding_only`) ever call `binding.forward`. ADR 0008 anticipated this ("will be revisited in Phase 6") — the revisit never happened; corrected in place.

**Concrete consequence:** `grow_region`/`grow_region_seeded` append zero-init rows to the binding projection with a "gradients will shape them over time" comment — since no gradients ever flow, grown units' binding rows stay exactly zero forever. At 9.3× growth, ~89% of each region's units contribute to SAE reconstruction but nothing to episodes/memory/recall. Campaign stayed healthy because encoders/decoders/heads adapt around the fixed random binder — `per_region_mse` doesn't gate on binding.

**Options:** (a) accept/document (free, gives up ADR 0008's whole point). **(b) live-episode head training** — recompute episodes live via burn binding during NREM, train heads against regional targets on them, gradients flow into binding. (c) temporal prediction loss (GRU-specific, complements b).

**Decision (2026-08-15): (b)**, with (c) sequenced as a separate follow-up. **ADR 0041**: opt-in `--binding-replay` NREM pass; deliberately does NOT train the GRU (the `None`-context pass excludes it — that's (c)'s job). **Implemented & verified (`4f7aa1f`):** `forward_tracked` ≡ `forward(None)`, stateless; grown units' zero-init rows move once active; public-API test pins never-active rows staying exactly zero.

**Isolated A/B pilot (`engineering-log.md` §35, 2026-08-15):** mechanism validated — fix engages in production state (47-96% of grown rows nonzero in 10 cycles vs. 0 control) — but the predicted episode-drift recall regression materialized (`gist_recall_rate` collapses to 0.23-0.45 vs control's 0.90-0.99) and NREM cost rose 34.5%. **Not adopted.**

**Repeat pilot at step-budget cap N=1 (`engineering-log.md` §36, 2026-08-15):** wall-clock did improve (−24.8%) but quality got *worse* — 10× same-cycle MSE on fired cycles, plus a new salience/settle cascade starving 6/20 cycles of replay traces. **Also not adopted.** Step-count capping abandoned as the lever.

**Third pilot, candidate (a) — loss-scale reduction (`engineering-log.md` §44, 2026-08-15):** `--binding-replay-loss-scale 0.1` (new flag, step-count cap left at its default of 1). **Worse than both prior pilots** — `mean_output_mse` +440.6%, every region's `per_region_mse` up 25-81%, the same salience/settle cascade signature as §36 (`traces_replayed` −49%, `settle_iters_sum`/`settle_trigger_count` +39%) but with roughly 4x the quality damage at 10x *lower* gradient magnitude. **Also not adopted.** Loss-magnitude scaling abandoned as the lever — the opposite of what the theory predicted (a weaker update made things worse, not better), suggesting a partially-scaled update adds noise without enough signal to converge, and that the salience/settle cascade is a structural consequence of live-episode training itself, not a dosage problem.

**Open — no path chosen. Two untried candidates remain, (c) now favored:** (b) fuse into `fused_replay_wake_step_batch`'s existing optimizer step instead of a separate pass — doesn't address the cascade, lower priority given (a)'s result; **(c) decouple the salience/settle cascade directly** (a control-side clamp on wake-triggering reacting to NREM-quality swings) — favored, since the cascade (not step count, not loss magnitude) is the one symptom common to all three failed pilots so far. Full pilot numbers: `engineering-log.md` §35/§36/§44. ADR: `decisions/0041-live-episode-binding-training.md`.

---

## §39 — Qwen3 LoRA fine-tuning (ADR 0033's natural follow-on) never run

Same orphaned-infrastructure story as §38, same source commit (2026-06-27): `scripts/train_lora_qwen3.py` exists, ported from ADR 0033's Qwen2.5 script, but its own commit message says it's blocked ("current blocker: projection weights from Qwen2.5 don't match Qwen3 dims... regenerate projection weights from Qwen3 model, future campaign run"). That campaign run never happened — `memory/lora_output/` has never existed in git history, and a checksum/size/date comparison of `qwen3_lm.onnx` (4,517,046 bytes, 2026-07-05) vs. `qwen3_lm_base.onnx` (5,248,472 bytes, 2026-06-27) shows two genuinely different files with the later one's date matching the ADR 0035/0036 Qwen3-architecture-class-fix session, not a LoRA merge timestamp. **Production Qwen3 generation is running on frozen pretrained weights, never fine-tuned on somnus's own episode-prefix distribution.**

This is CLAUDE.md's deferred item **L** — its one-line description ("LoRA fine-tuning") didn't previously distinguish "never started" from "started, half-built, then orphaned by an unrelated architecture swap"; `CLAUDE.md` and `session-state.md` corrected 2026-08-15 to state this precisely. Found: `engineering-log.md` §37.4 (2026-08-15 documentation audit).

**Progress 2026-08-15:** fixed a live correctness bug found during recon — `--export-proj-weights`'s `w1_len` was hardcoded to the Qwen2.5-era `640*4480` for the `--linear-proj` control-arm path (`crates/somnus/src/lib.rs`), silently wrong under Qwen3's `5*1024=5120`; now computed from `action::N_REGIONS*action::LM_HIDDEN`. The default (non-linear/MLP) path was never affected. Also cleaned up leftover Qwen2.5 defaults in `scripts/train_lora_qwen3.py` (`--model-id`, `--output-dir`, description string, wrapper class name, a stale comment) so the script matches its own documented usage.

**Proj-weights half unblocked 2026-08-15:** `campaign_growth10x` was idle between sessions (session 82 exited cleanly at its cycle cap; no process running) at the time this was tried, so `somnus --action --export-proj-weights memory/proj_weights_qwen3 --memory-dir memory` was run directly against the live snapshot — safe, since `--export-proj-weights` only reads `snapshot/decoder_weights.bin` and writes to a new subdirectory, touching nothing the campaign's own sessions read or write. Produced correctly-shaped weights (`w1.bin` 655,360 f32 `[640,1024]`, `w2.bin` 5,242,880 f32 `[1024,5120]`) from the campaign's actual trained decoder state — output at `memory/proj_weights_qwen3/` (i.e. `/data/somnus/memory/proj_weights_qwen3/`, gitignored runtime data). If `campaign_growth10x` is running when this is next attempted, copy `snapshot/` first rather than reading it live.

**Still blocking a real LoRA run — training data:** `--export-training-data` has never been enabled on any `campaign_growth10x` session (`EXTRA_ARGS=` empty in every `params.txt` entry checked), so no `(episode, text)` JSONL has accumulated. Two options, neither attempted this pass (would mean either modifying the live campaign's next-session launch command or a separate non-training `--action` pass against a copy of `memory/warm` — both are decisions about the primary research asset, left for the user): (a) add `--export-training-data <path>` to `EXTRA_ARGS` starting with the campaign's next session (its warm store already has ASR-derived text-bearing records from all 82 sessions so far per ADR 0033's ~20-40% estimate, so this could also plausibly be retried retroactively against the existing warm store, though that hasn't been confirmed to work); (b) same flag on a short standalone `--action`-only run against a copy.

With training data in hand, `peft`/`transformers` are already installed in `.venv/` (from the §38 work) and `train_lora_qwen3.py` should be runnable — ADR 0033's original Qwen2.5 run took ~20 minutes CPU for 1544 pairs/3 epochs, a similar order of magnitude expected here.

---

## §40 — Whisper `decode_greedy` intermittent ORT panic: root cause undiagnosed

`decode_greedy` intermittently panics inside ORT ("GetElementType is not implemented"), input-correlated, not yet reproduced on demand. **Mitigated 2026-07-05** via `catch_unwind` in the ASR thread (process survives, cycle's ASR output dropped).

**Instrumented 2026-08-15** (crates/somnus-metrics, crates/somnus/src/lib.rs): `AsrError` (returned `Err`) and the new `AsrPanic` (actual panic) counters are now split — they were previously conflated under one `AsrError` counter, which made root-cause triage from the metrics file alone impossible. Two new Core counters, `AsrPostPanicSuccess`/`AsrPostPanicFailure`, fire on every ASR decode *after* the first `AsrPanic` in a process session — directly answers the open poisoning question (does the `Session` recover fully, or silently degrade) the next time a panic actually fires, without needing to force a reproduction. Note "poisoning" here is not Rust `Mutex` poisoning (no `Mutex` is involved — the `Session` is exclusively owned by the single ASR thread); it means ORT-internal (FFI-side) state possibly left inconsistent after an unwind through `Session::run`. Also added: the panicking input (`encoder_hidden`, the raw Whisper encoder output) is now dumped to `<memory_dir>/asr_panic_captures/panic_input_<ts>.bin` on every panic, so the next occurrence leaves a reproducible input instead of vanishing with the dropped cycle.

**Still open:** the poisoning question itself and the root cause — both need a natural panic recurrence to observe (can't be forced on demand). No decision on priority beyond the instrumentation landing.

---

## §42 — ADR 0038 (hierarchical k-WTA encoder scoring): prerequisite 3 — resolved with a caveat, ADR stays `proposed`

ADR 0038 picks hierarchical/coarse-to-fine WTA over MoE-routing and approximate sampling for encoder-side scaling (the k-WTA encoder is O(input_dim×n_units) and can't be sparsified the way `engineering-log.md` §25/§30 sparsified binding). Three prerequisites were named before implementation; 1 and 2 were already answered (`engineering-log.md` §28/§29 — live scoring cost is small/not dominant at current scale; doesn't matter yet at honeybee-order target).

**Prerequisite 3 — numerical validation — done 2026-08-15.** Built a minimal block-local scorer (`somnus-perception::region::hierarchical_validation`, validation-only, not wired into any production path) and validated it against the invariants `select_top_k`'s global competition guarantees: k/n ratio preserved, indices in-range, duty-cycle bounded, newly-grown units immediately eligible — all confirmed via unit/property tests (`region.rs`'s `hierarchical_validation_tests` module, including a proptest over randomized shapes).

**Caveat surfaced by the validation (real finding, not just a pass/fail):** a naive port of `select_top_k`'s per-unit-only duty-cycle boosting silently starves chronically-low-scoring *blocks* — a block rarely chosen at the coarse stage never gets any of its units selected no matter how boosted they are individually, since the per-unit boost only applies once a block is already selected. The fix needed is a **second, block-level duty-cycle boost** applied at the coarse block-selection stage (same constants, `BOOST_STRENGTH`/`DUTY_CYCLE_ALPHA`, tracked per-block instead of per-unit). With that fix in place, recovery from a moderate, persistent score disadvantage does happen — but only on the same ~1000+-round timescale the production `DUTY_CYCLE_ALPHA=1/1000` already implies (confirmed empirically: a 500-round test showed zero recovery, a 4000-round run did). This bound — duty-cycle boosting can resolve a gap up to roughly `BOOST_STRENGTH × target_dc`, not an arbitrarily large one — already applies to the existing single-stage encoder, but block-local routing adds a *second* place it has to be applied correctly (block selection, not just unit selection), which a straightforward port would silently miss.

**Not validated — still open:** `per_region_mse` trend at growth-campaign scale, the way `engineering-log.md` §16's pilots validated the current global-competition encoder. The property tests validate the selection mechanism's invariants in isolation; confirming reconstruction quality holds up under block-local routing at real campaign scale needs an actual growth-campaign-style run, not attempted here.

**ADR stays `proposed`, not closed:** all three prerequisites are now resolved, but 1/2 remain "doesn't matter yet, not wrong direction" (`brain-scale-comparison.md` frames the project around eventually closing the scale gap via exactly this kind of technique), and prerequisite 3's campaign-scale piece is still open. **Trigger to revisit implementation:** the campaign target moves past honeybee-order, or the encoder/binding cost balance shifts back toward the encoder mattering (binding just got cheaper via `engineering-log.md` §30, pushing further from urgency, not closer).

---

> **Note:** the `#` column matches the `##`/`§` section number above, not chronological/priority order. Numbers not listed here are closed — see `engineering-log.md`.

| # | Item | Effort | Impact | Status |
|---|---|---|---|---|
| §4 | **LISTA unrolled encoder** | Med | Sparse coding quality | Deferred |
| §7 | **SigLIP2-ViT-B/16** | Med | Visual (Phase 7+ deferred) | Deferred |
| §10 | **Dedicated text embedding model** | Med | Language encoder | Deferred |
| §11 | **Gemma 3 1B IT** | Med–High | Generation (lower priority than closing §38/§39) | Deferred |
| §15 | **Crate hygiene** (usearch, burn-ndarray watch, ffmpeg-the-third) | Low–Med | HNSW footprint / maintenance | Deferred until raised |
| 🔮 §23 | **Native Rust ONNX graph construction** | High | Removes Python runtime dependency | Scoped, not started |
| 🔍 §33 | **BindingArea weights never trained** | Med–High | Episode quality / temporal context / grown-unit participation | Decision (b), ADR 0041 implemented; three pilots FAIL (drift+cost, then worse instability, then worse still at lower loss scale); (c) favored among 2 remaining candidates |
| §39 | **Qwen3 LoRA fine-tuning** (ADR 0033 follow-on) | Med (a campaign run) | Generation grounded in somnus's own distribution | Blocked infra never run; needs a priority decision |
| §40 | **Whisper `decode_greedy` panic root cause** | Low–Med (diagnosis first) | ASR reliability under prolonged unattended runs | Mitigated (catch_unwind); instrumented 2026-08-15 (AsrPanic/AsrPostPanic* counters + input capture); poisoning question + root cause still open, pending natural recurrence |
| §42 | **ADR 0038 prerequisite 3 + revisit trigger** | — | Informs hierarchical WTA priority | All 3 prereqs resolved 2026-08-15 (3 with a caveat — see above); ADR stays `proposed`, campaign-scale `per_region_mse` check still open |

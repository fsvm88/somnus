# Session state — somnus

> **Last updated:** 2026-05-16
> **Last commit on main:** 1d96599 — *Warm HNSW persistence: rebuild from cold archive on restore*
> **Branch:** `main` (tracking `origin/main` on Codeberg, SHA-256 repo)

Canonical record of where the conversation arc on somnus stands. Read end-to-end when resuming a session.

## Where we are

The **vision discussion is complete and folded into docs.** All six defining commitments are stable:

1. **Continuous learning** — no fixed train/inference split; persistence across sessions; owns catastrophic forgetting.
2. **Sleep and wake** — two-stage cycle (NREM-like for replay/homeostasis/compression; REM-like for generative replay/self-probe/growth); behavioral triggers (exhaustion, wake-perf) with schedule fallback; adaptive duration with cap; salience-gated interruptibility.
3. **Multimodal sensory input** — audio + video as primary; text bypass as a debug/research channel.
4. **Brain-inspired architecture** — regional specialization (audio-speech, audio-env, visual, symbolic) + integrative/binding area; bidirectional region↔integrative paths for cross-modal top-down priors.
5. **Bottom-up growth** — starts as a "baby brain"; expands via cascade-correlation-style unit addition when behavioral saturation (gist-level recall failure) triggers during REM-like sleep.
6. **Grounded concepts** — every concept anchored to perceptual exemplars; continuous embeddings now, with discretization (VQ-style) deferred to Phase 3+.

Operating constraints: single consumer machine, ~14 GB GPU VRAM (16 GB minus desktop overhead) + ~56 GB system RAM (of 64 GB) + disk for cold long-term memory.

The high-level **architecture** in `ai-docs/architecture.md` is stable end-to-end:

- Frozen pretrained encoders (visual + dual-purpose audio + text-bypass) emit **multi-source** continuous representations.
- Regional concept areas (sparsely coded continuous embeddings) per modality and concept domain.
- Integrative / binding area composes regional outputs into unified episodes; **bidirectional** with the regions for cross-modal top-down priors (McGurk-style disambiguation).
- Three-tier memory hierarchy:
  - **Hot** (VRAM) — episodic memory, recent unified episodes, source of replay.
  - **Warm** (system RAM) — semantic memory, consolidated, sparsely coded, growable, region-sharded.
  - **Cold** (disk) — long-term archive, compressed, sharded by region × concept.
- **Recollection layer** — cross-tier index pulling cold memories back into warm/hot when cued.
- **Sleep cycle** runs replay + self-probe + growth + compression across the two stages.

## What's been done in the repo (excerpt — see git log for full history)

- Project skeleton: `ai-docs/` (vision, principles, glossary, architecture, tech-stack, roadmap), `ai-docs/research/` and `ai-docs/decisions/` with conventions, `CLAUDE.md` for behavioral instructions, `.gitignore`.
- Git initialized as SHA-256 repo on `main`, pushed to Codeberg.
- `.claude/mind.mv2` (memvid project memory) is tracked; lock files are not.
- Vision and architecture sharpened across multiple commits through a structured Q1/Q2/Q3 + sub-question conversation.
- Full engineering-tradeoffs research pass (Option B — engineering estimates). Nine new notes + two supplement sections. All notes indexed in `ai-docs/research/README.md`.
  - `encoders.md` — Whisper/CLAP/HuBERT/DINOv2/SigLIP; VRAM, latency, pairing recommendations.
  - `ml-runtimes.md` — ort/candle/burn/tch-rs; split runtime (ort for frozen inference, candle for cognitive training).
  - `vector-retrieval.md` — flat/HNSW/IVF-PQ/Hopfield; memory formula; budget by scale.
  - `sparse-coding.md` — k-WTA/top-k/SAE; dead-unit problem; interaction with growth.
  - `memory-compression.md` — FP16/INT8/INT4/PQ; warm→cold pipeline; time estimates.
  - `growth-mechanisms.md` — cascade-correlation; ~10–40 min GPU per unit added; dominates sleep budget.
  - `replay-strategies.md` — uniform/PER/reservoir/generative; episodic buffer sizing (12M episodes in VRAM).
  - `multimodal-binding.md` — concatenation→gated fusion→cross-attention upgrade path.
  - `predictive-coding.md` — Friston FEP; partial adoption (SAE reconstruction error = prediction error).
  - `continuous-learning.md` — supplemented with hardware budget section.
  - `sleep-and-consolidation.md` — supplemented with sleep time/memory budget estimates.
- Phase 1 and Phase 2 architectural design. Four ADRs:
  - `ai-docs/decisions/0001-rust-workspace-layout.md` — crate structure: `crates/somnus-sensory` (lib) + `crates/somnus` (binary) for Phase 1; reserved names for Phase 2–5 crates; no premature traits.
  - `ai-docs/decisions/0002-phase1-sensory-types.md` — **fully spec'd and audit-complete.**
  - `ai-docs/decisions/0003-phase2-encoder-selection.md` — **accepted.** Whisper-tiny encoder-only (384-dim speech), CLAP-HTSAT-unfused (512-dim env), DINOv2-S/14 (384-dim visual). HF revisions pinned. ONNX artifacts in `.gitignore`'d `models/`, exported via `scripts/export_models.py`.
  - `ai-docs/decisions/0004-phase2-encoding-types.md` — **accepted (corrected).** `ort` 2.0.0-rc.12 + `load-dynamic` against system ORT 1.24.4. Exact mel-spectrogram preprocessing formulas verified by parity tests. CLAP correction: `rand_trunc` mode, Slaney scale, fmin=50 Hz (not HTK/fusion as originally stated). ADR updated with implementation notes (LOGSTEP, f64 pipeline, float-space resize).
- **Phase 2 encoding crate implemented and fully tested.** `crates/somnus-encoding`: `WhisperMel`, `ClapMel`, `frame_to_tensor` (DINOv2 image), `AudioEncoder`, `VideoEncoder`. 7 parity fixture tests pass within 1e-4 absolute vs. HuggingFace Python reference. `scripts/gen_mel_fixtures.py` generates fixtures. ADR 0004 updated with corrected CLAP parameters and implementation notes.
- **Phase 2 wiring complete.** `crates/somnus/src/main.rs` wired end-to-end: ORT initialised from system library at startup; `--models-dir` CLI arg (default `models/`) with up-front file existence checks; `AudioProcessor` (lazy resampler init via `SincResampler` from `somnus-encoding::preprocess::resample`, 480k-sample accumulation for both 16 kHz and 48 kHz, `encode_speech` + `encode_env`); `VideoEncoder` task per frame; embedding shape + latency logged via `tracing::info!`. All 31 workspace tests pass.
- **Phase 4 memory hierarchy complete.** `crates/somnus-memory`: `EpisodicBuffer` (VecDeque ring), `SemanticStore` (hnsw_rs HNSW, L2, eager insert), `ColdArchive` (INT8 quantised append-only binary, memmap2), `RecollectionLayer` (warm query live; promote stubbed). `somnus` binary updated with `--episodic-capacity`, `--memory-dir`, `--top-k`; memory metrics on 10s tick. 44 workspace tests pass. ADR 0006 written.
- **Phase 5 sleep cycle complete.** NREM replay trains k-WTA encoder + SAE decoder via AdamW (straight-through estimator); homeostatic decay after replay; hot→cold spill + `episodic.clear()`; REM self-probe measures per-region MSE and triggers `grow_region` (Tensor::cat + Var rebuild) when MSE exceeds threshold; atomic snapshot (episodic binary + safetensors weights); snapshot restore on startup; HNSW rebuilt from episodic records on restore (not persisted — lifetime constraints). Sleep interruptible via `AtomicBool`; replay checkpoint preserved across interrupts. Growth is session-local in Phase 5 (shape mismatch on restore). 51 workspace tests pass. ADR 0007 written. Binary smoke test confirms: multiple sleep cycles fire, loss decreases (1.81 → 1.37 across 3 cycles on a short clip), snapshot written, restored correctly on next run.
- **Phase 1 sensory pipeline implemented and fully tested.** Crate layout: `crates/somnus-sensory` (library) + `crates/somnus` (binary). All modules pass `cargo fmt`, `cargo clippy -D warnings`, `cargo test`. 22 tests total (13 unit, 9 integration).
  - `frame.rs` — `Timestamp`, `AudioFrame` (frame_count, duration_secs), `PixelFormat`, `VideoFrame` (bytes_per_pixel).
  - `error.rs` — `SensorError` (11 variants via thiserror), `SensorResult`.
  - `metrics.rs` — `SensorMetrics` (AtomicU64), `MetricsSnapshot` (drop_rate).
  - `audio.rs` — `AudioCapture`: cpal live mic capture, 10-format `make_cb!` dispatch, drop-oldest backpressure.
  - `video.rs` — `VideoCapture`: nokhwa pull-model webcam; `unsafe impl Send` (single dedicated thread, documented safety).
  - `media_file.rs` — `MediaFile`/`MediaFileStreams`: single decode thread, ffmpeg-next 8, nested-scope stream borrows, channel-layout fallback, `scaling::Context` built inside thread (it is `!Send`), flush sequences, `send_drop_oldest` helper.
  - `main.rs` — clap CLI (`--file`, `--mic`, `--webcam`), tokio async runtime, `spawn_blocking` bridges, `select!` loop with 10s metrics tick.
  - `tests/media_file_it.rs` — 9 integration tests: error path, video geometry, frame count, pixel invariants, audio metadata, sample finiteness, total duration, and metrics accounting for both streams. Each test synthesizes its own MP4 via `ffmpeg` CLI.
  - **ADR deviations (not in ADR 0002):** `unsafe impl Send for VideoCapture` (nokhwa Camera is !Send); `scaling::Context` moved into decode thread (ffmpeg-next scaler is !Send, unlike the resampler).

## Where to pick up

**Phase 5 is complete (2026-05-10).** All 51 workspace tests pass. Smoke test confirms the full sleep cycle works end-to-end.

**Also done (2026-05-10):**
- Phase 5 exit criterion verified: 16-trace buffer accumulates enough samples; speech/visual/symbolic grow 1024→1088 in first sleep cycle; loss decreases cycle-over-cycle.
- Durable growth: `region_sizes.bin` (4 × u64 LE) saved to snapshot dir; `load_weights_with_sizes` pre-grows layers, loads safetensors, calls `rebuild_linears()` (which re-reads all weights from VarMap since `Var::set` does not update Tensor clones in-place — also fixes silent no-op bug in the prior `load_weights`). Verified across two runs.

**Also done (2026-05-10):**
- Cross-attention binding (ADR 0008): replaced concat+linear(4096→512) with single-head self-attention over 4 region tokens. Per-region projections N_UNITS→128, Q/K/V [128,128], flatten [4,128]→[512]. 573K params vs old 2.1M. Input clamped to N_UNITS via to_dense_n (growth-invariant). Old snapshots incompatible (binding.weight key replaced by 7 keys).

**Also done (2026-05-11):**
- SemanticStore warm consolidation: sleep now calls recall.warm.insert per spilled record; warm tier is populated within session but not across restarts (HNSW lifetime constraint, ADR 0007)
- Salience-gated interruption: pending_embs Arc<AtomicUsize> tracks embeddings produced by encoder tasks but not yet consumed; reset to 0 at sleep entry; sleep checks count > SALIENCE_THRESHOLD=16 after each replay step and exits early if true

**Also done (2026-05-11):**
- Neuromodulation: `nrem_lr()` in `sleep.rs` — cosine annealing (LR_MAX=1e-3 → LR_MIN=1e-5) × loss-trend gate (full/half/quarter lr for improving/plateau/worsening loss). EWMA α=0.3. `set_learning_rate` / `learning_rate` wired on `PerceptionProcessor` (delegates to AdamW, no moment reset). Base lr restored after NREM so `grow_region` optimizer reset starts clean.

**Also done (2026-05-11):**
- Recollection during wake: `SemanticStore::query_with_dist` + `RecollectionLayer::query_warm_with_dist` return `(f32 dist, &EpisodeRecord)`. `record_episode` always queries warm memory; logs top-1 id, timestamp, L2 distance at INFO. If top-1 L2 < RECALL_BLEND_THRESHOLD (50.0), blends α=0.1 of retrieved into stored episode (intentional feedback loop — recall shapes future recall). Threshold needs calibration once real distances are observed from a session with warm data.

**Also done (2026-05-11):**
- Output reconstruction channel (ADR 0009): four `linear_no_bias` heads (EPISODE_DIM→region_dim) per modality. `wake_reconstruct_step` called once per episode; MSE loss backprops through heads and binding area. `output_mse` logged at INFO. 4 new weight tensors in VarMap (output_heads.{0,1,2,3}.weight). `rebuild_linears` extended. NREM integration deferred (would need episode vector re-derivation during replay). 52 tests pass.

**Also done (2026-05-11):**
- Bidirectional top-down priors (ADR 0010): `prior_predictions: [Option<Vec<f32>>; 4]` + `prior_beta: f32` on `PerceptionProcessor`. After each episode, output heads run in inference mode; predictions stored as priors. Next ingest call blends raw encoder input with prior via `(1-β)*raw + β*prior`. β defaults to 0.0; enable with `--prior-beta` after output_mse converges. Raw encoder inputs still stored in `latest_inputs` for unbiased training targets. Logs `prior_cosine` at INFO when β > 0. 53 tests pass.

**Also done (2026-05-11):**
- Live observability run on `test.mp4` (models already exported). `output_mse` drops 4.03→3.32→2.44→1.81 across the first four episodes — output heads learn within-session. Clip too short to trigger sleep (needs ~3277 episodes at 80% high-water). Recollection doesn't fire on fresh run (no warm memory). `prior_cosine` silent (β=0.0 default). Calibration targets: `RECALL_BLEND_THRESHOLD=50.0` and `prior_beta` need a multi-session run.
- NREM output-head replay: `sleep.rs` NREM loop now also calls `wake_reconstruct_step(episode_vec, regional)` after each `replay_step`. Stored episode_vec is a constant (untracked tensor), so gradients flow only through output head weights. `mean_output_mse` logged at NREM complete and in `SleepStats`.
- Warm-recall growth gate: REM probe snapshots `pre_spill_warm_len` before spill; queries warm with each episode vector if `pre_spill_warm_len > 0`. Mean L2 logged as `warm_recall_l2`. Growth now requires both SAE MSE > 0.1 AND warm recall poor (L2 > 100.0 or no cross-session data). `WARM_RECALL_GROWTH_THRESHOLD=100.0` needs calibration.

**Also done (2026-05-11):**
- Codebase review: two latent bugs identified and fixed.
  - `pending_embs.fetch_sub()` could underflow to `usize::MAX` when sleep resets the counter to 0 mid-flight. Both consume sites (audio + video arms in `main.rs`) now use `fetch_update` + `saturating_sub(1)`.
  - `replay_checkpoint` in the NREM loop only advanced inside the `Ok` arm. A broken trace would be retried on every sleep restart. Moved increment outside the match so errored traces are always skipped.

**Also done (2026-05-11):**
- Predictive-coding settling loop (ADR 0011): `KWtaLayer::forward_no_dc` skips duty-cycle update; `settle_episode` iterates up to `settle_steps` times — runs output heads on current episode, blends predictions with raw inputs via `settle_beta`, re-runs k-WTA (no DC update), rebinds, logs L2 delta at INFO, stops when delta < SETTLE_EPS=1.0 or steps exhausted. All three `ingest_*` methods call it when episode is produced. `set_settle_steps` / `set_settle_beta` public setters. `--settle-steps` / `--settle-beta` CLI args (defaults 0 / 0.5). Startup WARN when settle_steps > 0 and no snapshot found. 54 tests pass.

**Also done (2026-05-12):**
- Action loop designed and documented (ADR 0012): region-token projection (linear 512→4×LM_HIDDEN) + frozen Qwen2.5-0.5B. Dual function: (a) per-episode LLM self-attention over 4 region tokens → per-region gain multipliers applied to k-WTA inputs of next episode; (b) full autoregressive decode on stdin query → stdout JSON. Text-bypass targets stored on EpisodeRecord train the projection during NREM (separate LR 1e-4, not subject to neuromodulation schedule). Architecture doc and decisions README updated.

**Also done (2026-05-12):**
- ADR 0012 end-to-end verified. Fixed four issues found during smoke test:
  - Qwen2.5-0.5B must be exported with `torch_dtype=float32` (bf16 default causes ORT dtype mismatch)
  - tracing subscriber redirected to stderr (was corrupting stdout JSON)
  - File sources use blocking `forward_frames_blocking` (prevents Whisper buffer starvation under CPU load)
  - Pre-sleep query drain via `try_recv()` loop at top of main loop (prevents query starvation when episodes accumulate rapidly)
- Observed: query → ~2.4s inference → 25-token JSON response; output is repetitive (expected — projection is randomly initialized, no training yet)
- Gain updates fire per-episode with varied values in [0.5, 2.0] range across 4 regions ✓

**Also done (2026-05-15):**
- Calibration run completed. Two bugs fixed and two thresholds set from real data:
  - **Bug fix**: warm recall probe was running AFTER hot→cold spill, so HNSW always found exact matches (L2=0.0). Fixed by moving probe before spill.
  - **Bug fix**: `forward_frames_blocking` had no return value; changed to return `bool` (true=source exhausted, false=downstream closed) so the file-loop thread can distinguish normal EOF from disconnection.
  - **`--file-loop` flag added** to `main.rs`: restarts MediaFile in-process on EOF, keeping warm memory alive across iterations. Enables multi-cycle calibration from a single short clip.
  - **`RECALL_BLEND_THRESHOLD`**: 50.0 → **5.5**. Calibrated from 933 wake-time recall measurements (same-content looped video): p95=4.1, max=5.3. Threshold 5.5 blends familiar content, suppresses noise from dissimilar episodes.
  - **`WARM_RECALL_GROWTH_THRESHOLD`**: 100.0 → **15.0**. Calibrated from 17 sleep cycles: same-content warm L2 ranged 1.6–3.5 (median 2.3, max 3.5). Threshold 15.0 gives ×4 safety margin; genuinely novel content expected at L2 >> 5.
  - `prior_beta` and `settle_steps` remain 0.0/0 — output_mse is stable (~0.013–0.020) but dataset is too homogeneous (one looped video) to calibrate these safely.

**Also done (2026-05-16):**
- Warm HNSW persistence across restarts. Root cause: sleep calls `episodic.clear()` before `save_snapshot`, so the snapshot's episodic buffer is empty; `load_snapshot` rebuilt warm from that empty buffer → zero cross-session recall. Fix: `load_snapshot` now opens the cold archive and inserts all historically spilled records first, then any remaining hot-buffer records. Cold records have lower IDs by construction so no ID collisions occur in normal operation. New test `warm_rebuilt_from_cold_after_hot_cleared` verifies the fix. ADR 0007 updated: documents the rebuild-from-cold approach and explains why native `hnsw_rs` file persistence was not used (lifetime/unsafe complexity; acceptable for Phase 5 cold-archive sizes). 58 workspace tests pass.

**Next (queued):**
- Remaining Phase 6+ items: sample-level audio mixing, language as additional modality.
- Re-calibrate `prior_beta` and `settle_steps` once diverse multi-video content is available.
- True cross-session threshold validation (now unblocked: warm HNSW persists across restarts).

## Open architectural questions

The full list lives at the bottom of `ai-docs/architecture.md`. Salient unresolved items:

- Initial regional set (audio-speech + symbolic for Phase 1, or all four from the start?)
- Encoder specifics (single multi-task audio encoder vs. two specialized; specific model choices for visual)
- Growth mechanism details (granularity, sparse-code integration)
- Recollection-layer implementation (vector ANN, Hopfield, hybrid)
- Compression strategy (quantization, pruning, distillation, layered)
- Sparse-coding scheme (k-WTA, top-k, learned penalty)
- Sleep-cycle time budget (NREM/REM split)
- Action loop (whether the system has one in early phases)
- Scheduling (async / dedicated threads / event loop)
- Tensor representation (single shared type vs. per-subsystem)

## How to resume

When the user says "resume SOMNUS prompt" (or close variation):

1. Read `CLAUDE.md` (already auto-loaded) for behavior.
2. Read this file end-to-end.
3. Skim `ai-docs/vision.md` and `ai-docs/architecture.md` to refresh the integrated picture.
4. Skim research notes in `ai-docs/research/` if the user's prompt references them.
5. Check `git log --oneline -10` for recent commit messages — they summarize each substantive change.
6. Greet briefly with "where we left off", then either ask for direction or proceed with the queued next step.

## When to update this file

At a logical break point (after a substantive set of edits has been committed), refresh:

- "Last updated" date
- "Last commit on main" hash and subject
- "What's been done in the repo"
- "Where to pick up"

Keep the rest stable unless the project picture itself shifts.

# Session state — somnus

> **Last updated:** 2026-05-18 (Phases 10b, 10c, 10d, 11a, 11b, 12a, 12b, 12c complete)
> **Last commit on main:** (see git log) — Phase 12c: regional unit pruning
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
- Warm HNSW persistence across restarts. Root cause: sleep calls `episodic.clear()` before `save_snapshot`, so the snapshot's episodic buffer is empty; `load_snapshot` rebuilt warm from that empty buffer → zero cross-session recall. Fix: `load_snapshot` now opens the cold archive and inserts all historically spilled records first, then any remaining hot-buffer records. Cold records have lower IDs by construction so no ID collisions occur in normal operation. New test `warm_rebuilt_from_cold_after_hot_cleared` verifies the fix. ADR 0007 updated.

- **ID collision bug fixed**: when episodic buffer is empty (post-sleep), `load_episodic` set `next_id=1`, causing every new session to reuse IDs from the cold archive (discovered during validation run: 4611 cold records → only 1782 unique in warm sidecar). Fix: `save_snapshot` now writes `next_id.bin` (8B u64 LE); `load_snapshot` restores `next_id` as `max(file_value, max_cold_id+1, hot_counter)`. Verified: session 2 shows `next_id=1300` continuing from session 1.

- **Cross-session threshold validation complete** (clean 2-session experiment, test.mp4 × 2):
  - Session 2 startup: `warm HNSW rebuilt: cold_records=1300 hot_records=0 total=1300 next_id=1300` ✓
  - Session 2 `semantic_len=1300` matches cold archive size exactly ✓ (no ID collisions)
  - Cross-session `warm_recall_l2` on first complete sleep cycle: **1.40** — well below calibrated `WARM_RECALL_GROWTH_THRESHOLD=15.0` ✓
  - All subsequent cycles: 1.0–3.5 (consistent with within-session range from calibration) ✓
  - Growth correctly suppressed on all cycles (familiar content recognized cross-session) ✓
  - Quantization noise (INT8 cold → dequant) does NOT push L2 past 5.5 ✓
  - **Verdict: both calibrated thresholds hold cross-session. RECALL_BLEND_THRESHOLD=5.5 and WARM_RECALL_GROWTH_THRESHOLD=15.0 are confirmed stable.**

**Also done (2026-05-16) — prior_beta baseline calibration:**
- Ran `--prior-beta 0.1 --settle-steps 3` on existing test.mp4 memory (2600 cold records, weights backed up to `memory/snapshot/weights.safetensors.bak`).
- **prior_cosine results** (n=1029 episodes):
  - visual: mean=0.9886, min=0.8930 (dominant modality, 751 samples)
  - speech: mean=0.9879, min=0.9589 (55 samples)
  - symbolic: mean=0.9879, min=0.9582 (55 samples)
  - env: mean=0.8055, min=0.4576 (168 samples, most stochastic)
  - Overall mean=0.9586. Output heads confirmed working — near-perfect prediction on familiar content.
- **settle delta convergence** (SETTLE_EPS=1.0, 3 steps, iter=0 mean delta=2.006):
  - 35.0% converge at iter=0, 26.6% at iter=1, 5.7% at iter=2, **32.7% still not converged after 3 steps**.
  - Conclusion: familiar content has prior_cosine ≈ 0.99 so settling has little marginal benefit. The real test for settle_steps requires novel content where prior_cosine is 0.3–0.5.
- **warm_recall_l2**: 1.5–3.8 across all sleep cycles — within calibrated thresholds. ✓
- **output_mse**: mean=0.0412, range 0.007–0.533.
- **β=0.1 confirmed safe to use**; calibration on novel content still needed to set final value and settle_steps.

**Also done (2026-05-17):**
- Two latent bugs fixed in file-mode exit:
  - `audio_cb_tx`/`video_cb_tx` originals were never dropped after spawning one-shot file forwarding threads, so the crossbeam→tokio bridge tasks looped forever and embedding channels never closed.
  - Main `select!` loop had no exit path when both emb channels closed; only broke on Ctrl-C. Added `audio_emb_closed`/`video_emb_closed` guards + a `file_mode && both_closed` break after the select.
  - These bugs caused the binary to hang indefinitely after file EOF. Now exits naturally with "file source exhausted" log.

- Novel-content calibration complete (4 clips: talking_head, nature, animation, urban):
  - **prior_cosine on novel content**: visual 0.71–0.89 (vs 0.99 familiar), env 0.39–0.49 (vs 0.90 familiar). Novelty detection works — env is the sharpest discriminator.
  - **SETTLE_EPS recalibrated**: 1.0 → **4.0**. Original value was unreachable; familiar content has mean delta ~3.8 at iter=0 (and diverges slightly with more steps); novel content (talking_head) has mean 6.1→3.5 over 3 steps. 4.0 allows early exit on the lower tail of novel content.
  - **Settling behavior**: 42% delta reduction iter0→2 on talking_head (novel), but -7% on familiar (settling is slightly counterproductive when prior is near-perfect). Settling is useful for novel content, neutral-to-harmful for familiar. A prior_cosine-gated settle (only settle when prior_cosine < ~0.85) would be cleaner — deferred.
  - **settle_steps=3** kept as "3 fixed passes of top-down prior feedback" rather than a convergence criterion.
  - **prior_beta=0.1** confirmed safe across all novel clips.

**Also done (2026-05-17) — Phase 6 Language region (Stage A scaffolding):**
- Language added as the 5th region: 896-dim Qwen2.5 embeddings, k-WTA layer + SAE decoder, output head (640→896).
- EPISODE_DIM 512→640 (5 regions × D_PROJ=128); COLD_DIM 512→640; magic bytes bumped (SMNCOLD1, SMNHOT01) to reject old archives.
- All [4]-wide arrays promoted to [5] throughout: regional, gain, prior_predictions, output_heads, region_sizes, per_region_mse.
- `try_bind()` zero-pads language SparseActivation when region 4 absent — episodes still form from regions 0–3 alone.
- `ingest_language()` public entry point in PerceptionProcessor.
- `save/load_region_sizes` updated to 5×u64 LE (40 B); old 4-region files return None.
- Prior_cosine-gated settling implemented: SETTLE_COSINE_GATE=0.85, only settle when prior_cosine < 0.85.
- Deleted incompatible memory/ directory; new session will start fresh.
- All 59 workspace tests pass; clippy clean.

**Also done (2026-05-17) — Phase 6 Stage A wiring:**
- `LangEmbedder` struct in `main.rs`: loads `qwen25_embed_table.safetensors` + `qwen25_tokenizer.json`; `embed()` tokenizes text via Qwen2.5 BPE, mean-pools token embeddings → 896-dim vec.
- `--language` CLI flag; language model file existence checked at startup.
- Single unified stdin reader: routes each line to `lang_tx` (if `--language`) and `query_tx` (if `--action`); combined use feeds both paths (text becomes region input AND action query text_target).
- Full episode lifecycle in the language arm: `ingest_language` → `wake_reconstruct_step` (when episode produced) → `record_episode` → gain update.
- Pre-sleep `lang_rx` drain mirrors the existing query drain.
- `lang_closed` guard prevents perpetual polling of the select arm when `--language` is off.
- 59 tests pass; clippy clean.

**Also done (2026-05-17) — Phase 6 Stage B (Whisper ASR → language region):**
- `scripts/export_models.py`: `export_whisper_decoder()` — `WhisperDecoderWrapper` (last-token-only LM head, ~10× speedup), exports `whisper_tiny_decoder.onnx` + `whisper_tokenizer.json`.
- `crates/somnus/src/whisper_asr.rs`: `WhisperDecoder::decode_greedy()` — SOT/EN/TRANSCRIBE/NO_TIMESTAMPS prefix, EOT stop, text filter `id < 50257`, tokenizer decode.
- `crates/somnus-encoding/src/audio.rs`: `encode_speech()` now returns `(AudioEmbedding, Vec<f32>)` — Vec is raw (1500×384) encoder hidden state for ASR.
- `main.rs`: `AudioProcessor` gains `whisper_dec: Option<WhisperDecoder>` + `lang_tx: Option<Sender<String>>`; ASR runs synchronously in audio encoder task after each speech embedding; `lang_tx` restructured (clone for stdin reader + clone for audio task + drop original).
- Soft failure: missing decoder ONNX → warn + ASR disabled; Stage A stdin ingestion unaffected.
- ADR 0013 written covering Stage A + Stage B.
- 59 tests pass; clippy clean.

**Also done (2026-05-17) — Phase 6 language calibration:**
- Export fix: `WhisperForConditionalGeneration` uses `proj_out` not `lm_head`; corrected in wrapper.
- Exported: `whisper_tiny_decoder.onnx` (118 MB), `whisper_tokenizer.json` (3.93 MB).
- Calibration run (`--language --prior-beta 0.1 --settle-steps 3`, talking_head.mp4 looped, 1 sleep cycle):
  - **ASR working**: 16 transcriptions, mean=114 chars/chunk (range 4–261) ✓
  - **Language ingest**: 16 events → 640-dim episodes ✓
  - **Language prior_cosine**: n=16, mean=0.027, min=−0.011, max=0.082 (near-zero; expected for random fresh head)
  - **Other regions**: visual=0.859, speech=0.896, symbolic=0.894 (familiar content)
  - **Sleep**: traces_replayed=409, mean_loss=0.040, warm_recall_l2=2.527 (within threshold) ✓
  - **Structural**: language ~4% of replay buffer (16/409). Two 30s speech windows per 70s file loop vs ~30 visual. Language output head trains slowly — expected; mirrors Phase 5 session-1 behavior for all regions.
  - **Verdict**: Stage B correct. Prior_cosine will rise toward 0.8–0.9 across sessions as language output head accumulates training.

**Also done (2026-05-17) — observability + settling gate:**
- `--log-file`, `--max-sleep-cycles`, `--metrics-file`, `--pid-file` CLI args added to `somnus`.
- `prior_cosines: [Option<f32>; 5]` added to `PerceptionOutput`; per-region cosines now flow to main for metrics.
- `asr_count` shared via `Arc<AtomicUsize>` between audio task and main; included in per-cycle NDJSON.
- `scripts/run.sh`: build release + `exec` binary — avoids cargo broken-pipe issue.
- `scripts/status.py`: process info (PID, uptime, CPU, RSS), cold archive stats, snapshot, last N sleep cycles.
- `sleep cycle complete` log now includes cycle number, wake_episodes, asr_count, lang_ingest_count.
- Per-region prior_cosine-gated settling: replaced stale `last_visual_cosine` field with per-call
  `cosine` parameter passed into `maybe_settle_and_update`. Each `ingest_*` now passes its own
  cosine(s); `ingest_speech` passes `min(speech_cos, symbolic_cos)`. More granular: env episode with
  env cosine=0.3 now correctly triggers settling even if the last visual cosine was 0.99.

**Also done (2026-05-17) — multi-session language calibration (3 sessions × 3 cycles):**
- Ran talking_head.mp4 looped with --language --prior-beta 0.1 --settle-steps 3 --max-sleep-cycles 3
  for 3 consecutive sessions, each using the same memory/ directory (weights accumulate across sessions).
- Language prior_cosine trajectory (full-replay cycles only):
    S1C2: 0.019  →  S2C2: 0.064  →  S3C2: 0.058
- Baseline was ~0.011 (random head); 0.064 after 2 full replay sessions = 3.4× above baseline. ✓
- Visual prior_cosine stable at 0.90 throughout (familiar content, expected). ✓
- No growth triggered (familiar content, warm_recall_l2 ≈ 2.0, well below 15.0). ✓
- Anomaly: S2C2 mse_sleep spiked to 1.92 (vs 0.08–0.11 elsewhere) — language head briefly
  destabilised. Resolved by S3C2 (0.058). Worth watching in future sessions.
- Salience-interrupt pattern: cycles 1 and 3 replay only 4–11 traces (file-loop fills buffer at CPU
  speed, pending_embs > SALIENCE_THRESHOLD). Expected file-mode behaviour.
- Projection: at ~90 ASR/3276-episode density (3%), full convergence to visual-level prior_cosine
  (~0.9) would take ~30+ sessions on this clip. In continuous mic use the rate would be much higher.
- Metrics written to memory/metrics.jsonl; status.py confirmed working.

**Also done (2026-05-17) — Phase 7: auto-grounding via language text targets (ADR 0014):**
- `ingest_language_line()` in `main.rs` now auto-sets `text_target` on every language-ingest episode
  (ASR transcription or stdin) when `--action --language` are both set. The text is tokenized via
  `ActionDecoder::tokenize()` and stored with `episodic.set_last_text_target(ids)`.
- Previously: `decoder_trainable_records = 0` in every session (projection trained only on manual
  queries, which nobody typed). Now: ~135–166 trainable records per cycle from ASR alone.
- ADR 0014 written covering the design, target placement rationale, and expected dynamics.
- NDJSON metrics bug fixed: `mean_proj_loss` and `decoder_trainable_records` were computed and
  logged but missing from the `serde_json::json!` block. Now included (commit 2148048).

**Phase 7 calibration results (memory/metrics_p7s2.jsonl, 2 sessions × 3 cycles):**
- S1C1: traces=3276 asr=166 trainable=166 proj_loss=0.09858 (first cycle, random-ish projection)
- S1C2: interrupted (traces=12), proj_loss=None
- S1C3: traces=3268 trainable=141 proj_loss=0.02169 (4.5× reduction within session 1 ✓)
- S2C1: traces=3276 trainable=151 proj_loss=1160.5 ← **spike; projection diverged between sessions**
- S2C2: interrupted (traces=20), proj_loss=0.08434
- S2C3: traces=3260 trainable=135 proj_loss=0.07824 (recovered)
- Session 3 was killed mid-run (SIGTERM), no data written.

**Key observation on S2C1 spike:** Projection loss jumped from 0.022 (S1C3) to 1160 at the start
of session 2, then recovered to ~0.08 by cycle 3. Likely cause: projection overfit to the session-1
episode distribution; session-2 episodes (rebuilt from a larger cold archive) look different and the
projection weight norms exploded. AdamW's second-moment normalization means loss-scaling is NOT an
effective gradient clip — advised against it. `proj_loss_max` added to diagnose on next session.

**Also done (2026-05-17) — Phase 8: generation evaluation + diagnostics (ADR 0015):**
- `--query TEXT` CLI arg: one-shot generation after first episode; used for non-interactive evaluation.
- `proj_loss_max: Option<f32>` added to `SleepStats` and NDJSON metrics (distinguishes sustained shift
  from catastrophic-step spikes — helps post-mortem the S2C1 anomaly on next run).
- `proj_weight_norm()` on `ActionDecoder`; logged after NREM completes.
- Repetition penalty 1.3 in `generate()`: `penalized[id] /= 1.3` for each previously generated token.
  Eliminates degenerate echo loops in greedy decoding.
- hnsw_rs 0.3.4 has `println!(" setting number of points… ")` in its source — pollutes stdout. Known
  issue; work around with `grep '"type":...'` in evaluation scripts.

**Generation evaluation results (~700 training examples, proj_loss ≈ 0.078):**
- Pre-penalty: all 3 queries produced degenerate loops (echo or n-gram repetition).
- Post-penalty (repetition_penalty=1.3): coherent, grammatical, topically relevant to query — but
  NOT grounded to clip content (expected; language prior_cosine ≈ 0.05 vs. 0.9 goal).
  - "what is being said?" → discourse about saying/doing (query-guided, no clip signal)
  - "describe what you observe" → discourse about observation and change
  - "hello" → "John Doe" book-author narrative (consistent across runs → region prefix IS doing something)
- Projection convergence expected to take 30+ sessions on one clip. On live mic: much faster.

**Also done (2026-05-18) — projection spike investigation + cosine loss fix:**
- P8S1 (old MSE loss): C3 diverged catastrophically (proj_mean=1.77e12, ratio max/mean=58.85×).
  Root cause: C2 trained projection to high precision (proj_mean=0.004); C3's new episode
  distribution caused large gradients; AdamW's stale v_t (≈ g_C2²) amplified C3 gradient →
  explosion. Same mechanism as S2C1 spike in Phase 7.
- Fix: switched `projection_step` loss from raw MSE to **cosine alignment loss** — normalise
  both `proj_mean` and `text_mean` to unit vectors before computing MSE. Loss bounded [0,4];
  magnitude-invariant; prevents gradient explosion regardless of AdamW moment state.
- Implementation: `proj_mean / ||proj_mean||` kept in graph (via `broadcast_div` on scalar norm
  tensor) for correct gradient flow. `text_mean` normalised as scalar (frozen, no grad needed).
- P8S2 (cosine loss): C1 proj_mean=0.0011, C3 proj_mean=0.0007, max/mean ≤ 3.3× throughout.
  No divergence. Fix confirmed.
- Also fixed: tracing default log level → INFO when RUST_LOG unset (so --log-file captures logs).
- Cosine loss scale: ~0.001–0.002 at current training level (vs 0.004–0.016 with MSE).
  Comparable to 2*(1-cosine_sim); lower is better alignment.

**P8S2 language prior_cosine trajectory:**
- C1=0.023 → C2=0.011 (interrupted, 28 traces) → C3=0.040
- Slow upward trend confirmed. Target 0.1–0.2 expected within ~10 more sessions.

**Also done (2026-05-18) — multi-clip training:**
- `--file` promoted from `Option<PathBuf>` to `Vec<PathBuf>` (`num_args = 1..`); may be specified
  multiple times on the command line.
- In `--file-loop` mode the thread now cycles through all files in order, restarting from the
  first when the last finishes. Logs `file loop: opening <path>` at each rotation.
- Without `--file-loop` the audio and video threads each stream all files in sequence then close
  their channels, triggering the existing clean-exit logic.
- Single-file usage (`--file foo.mp4`) is unchanged.
- 59 tests pass; clippy clean.

**Also done (2026-05-18) — multi-clip calibration session 1 (metrics_multiclip_c1.jsonl):**
- Rotation: talking_head → urban → animation → nature (loop), 3 sleep cycles.
- All 3 cycles ran to completion with no salience interruptions (4-clip pacing = evener episode
  production vs. tight single-clip loop). All 3276 traces replayed each cycle.
- Projection cosine loss stable: 0.00115→0.00085→0.00078, proj_max ≤ 0.0026. No divergence. ✓
- Warm recall L2 4.16→3.98→2.44 (system recognises multi-clip content; well below threshold 15.0).
- No growth triggered (familiar content; region_sizes=[1024,1024,1152,1024,1024] from prior session).
- Key prior_cosine C3 values:
  - speech: ~0.94, symbolic: ~0.94 (stable familiar content)
  - visual: 0.795→0.766→0.865 (dipped on novelty, recovered by C3)
  - env: 0.502→0.516→0.657 (steadily adapting to diverse audio)
  - language: 0.021→0.019→0.043 (slow upward from baseline 0.011; 3.9× above baseline after session)

**Also done (2026-05-18) — Phase 9 (continuous operation):**
- **9a (schedule trigger):** sleep entry now logs trigger reason (`high_water` vs `schedule`),
  buffer occupancy, and elapsed wake_secs. `sleep_trigger` field added to NDJSON metrics.
  Validated: fired at 30s with buffer_len=803/4096 (19.6%), well below high_water mark.
- **9b (mic/webcam graceful failure):** `AudioCapture::start` and `VideoCapture::open` failures
  now warn+continue instead of hard-failing. Validated: mic warns gracefully
  (`AudioConfig(DeviceNotAvailable)`), system continues on file input; full sleep cycle completes.
- **9c (cold→warm promotion):** investigated and closed as non-issue. Warm is already kept in
  sync with cold at all times (startup rebuild + spill+insert during sleep). Roadmap updated.
  Real scaling concern (HNSW rebuild time at 164k records → ~3min) deferred to Phase 11.

**Also done (2026-05-18) — Phase 10a: wake-performance degradation trigger:**
- `--perf-sleep-threshold f64` CLI arg (default 0.0 = disabled). Requires `--prior-beta > 0`.
- Per-wake EWMA (α=0.1) of mean prior_cosine per episode. Baseline = mean of first 20 episodes.
- When EWMA drops > threshold below baseline → perf_sleep_flag = true → trigger at next sleep check.
- Trigger reason = `"perf_degradation"` in log + NDJSON.
- EWMA state reset at each sleep entry alongside other per-cycle counters.
- Validated: 5× talking_head establishes baseline=0.566; novel noise clip tanks EWMA to 0.505;
  `trigger="perf_degradation"` fires at wake_secs=22, buffer_len=38/500. ✓

**Also done (2026-05-18) — Phase 10b: Adaptive sleep duration:**
- `--sleep-loss-target f32` (default 0.0): NREM exits early when loss EWMA < target; REM + spill + snapshot still run (`interrupted=false`).
- `--sleep-max-secs u64` (default 0): sleep exits when wall time exceeds cap; treated as interrupted (no REM/spill/snapshot).
- `nrem_exit_reason` field in SleepStats and NDJSON: `"traces_exhausted"`, `"loss_target_met"`, `"time_budget"`, `"salience"`.
- Validated: target=2.0 on 29-trace buffer → exits after 2 traces (`nrem_exit="loss_target_met"`, `interrupted=false`). max-secs=5 on 55-trace buffer → exits after 4 traces (`nrem_exit="time_budget"`, `interrupted=true`).

**Also done (2026-05-18) — Phase 10c: Gist-level self-probe in REM:**
- k-means (k=32, Lloyd's, up to 2048-record sample) on warm episode vectors during REM.
- Each probe episode and its top-1 warm neighbor are assigned to nearest centroid.
- `gist_recall_rate` = fraction of probe episodes whose neighbor is in the same cluster.
- Activates when warm store ≥ GIST_K=32 records; falls back to SAE MSE when unavailable.
- Growth trigger uses gist when available: grow when `gist_recall_rate < 0.5`.
- `gist_recall_rate: Option<f32>` added to SleepStats and NDJSON metrics.
- Validated: cycle 2 shows `gist_recall_rate=0.932` on 41 warm records × 59 probe episodes; growth correctly suppressed.

**Also done (2026-05-18) — Phase 10d: Working memory:**
- `--working-memory-size usize` (default 0 = disabled).
- Fixed-capacity VecDeque of recent episode vectors on `PerceptionProcessor`.
- During binding: WM mean ([EPISODE_DIM]) reshaped to [5, D_PROJ], appended to K and V (Q stays [5, D_PROJ] → output fixed at 640 with no new weights).
- Settling loop uses `None` for WM context (settling is within-episode iteration, not temporal).
- Cleared at sleep entry (`clear_working_memory()`).
- Validated: model trained with WM=8 on talking_head clip; re-running WITH WM=8 vs WITHOUT (WM=0) on same snapshot — output_mse consistently 40–50% lower with WM (e.g., episode 6: 0.29 vs 0.58). Output heads learned to predict from WM-augmented episodes.

**Also done (2026-05-18) — Phase 11a: PQ compression for cold archive:**
- `crates/somnus-memory/src/pq.rs`: `PqCodebook` (train/encode/decode/serialize), `PqColdArchive` (create/open/append/iter/len), `convert_cold_to_pq`.
- Constants: PQ_M=8, PQ_K=256, PQ_SUB_DIM=80 (640/8). Per-record: 24 bytes vs 660 INT8 → 27.5× reduction.
- Codebook: 655 360 bytes, stored at PQ file head (magic SMNCOLDQ); trained on up to 8192 even-spaced samples, Lloyd's k-means 25 iters, deterministic init.
- INT8 ColdArchive unchanged as live write path; PQ is offline conversion only.
- 6 new tests pass (round-trip, serialise, archive, convert, bad_magic, constants); 18/18 total in somnus-memory.
- Roadmap 11a marked ✓.

**Also done (2026-05-18) — Phase 11b: REM creative recombination:**
- `PerceptionProcessor::synthesize_episode(&self, regional_a, regional_b, alpha)` — blends encoder inputs (α=0.5), runs through `layer_forward_settling` (no DC update) + binding (no WM context) → synthetic episode vector.
- `PerceptionProcessor::measure_output_mse_inference(&self, episode, regional)` — inference-mode output-head MSE without backward step.
- REM creative recombination block in `sleep.rs` (after growth, before `episodic.clear`):
  - Constants: `REM_SYNTH_ALPHA=0.5`, `REM_SYNTH_MIN_L2=2.0`, `REM_SYNTH_MAX_N=8`.
  - Samples real episodes for inference-mode baseline MSE (`mean_real_inference_mse`).
  - Pairs with L2 > 2.0 synthesized; output MSE averaged over both source regionals → `mean_synthetic_output_mse`.
  - All three new fields in `SleepStats` + NDJSON metrics.
- Roadmap 11b marked ✓.

**Also done (2026-05-18) — Phase 12 complete:**
- **12a (eval harness)**: `scripts/eval_metrics.py` — 4 PNGs from a single invocation; matplotlib+numpy only; missing-field-safe; validated on all 24 existing metrics lines.
- **12b (HNSW persistence)**: `SemanticStore::dump` / `load_warm`; `Box::leak` lifetime resolution; bincode sidecar; `warm/version` tag; fast-path in `load_snapshot`; dump called at end of every sleep cycle. 21/21 somnus-memory tests pass.
- **12c (regional pruning)**: `PerceptionProcessor::prune_region` — L1 magnitude selection; grown units above N_UNITS=1024 only; `--prune-fraction` (default 0.0) + `--prune-min-grown` (default 64) CLI args; trigger when `gist_recall_rate ≥ 0.8`; optimizer reset after prune; `regions_pruned` + `total_units_pruned` in SleepStats + NDJSON.
- All three new ADRs (0016–0018) written; roadmap Phase 12 marked ✓.
- Full workspace builds clean; clippy clean; all tests pass.

**Next (queued): No specific next phase queued. Candidates:**
- More calibration runs to test 12b restart speedup and 12c trigger in practice.
- Phase 9b (real mic/webcam) — deferred until hardware available.
- Any new phase the user proposes.

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

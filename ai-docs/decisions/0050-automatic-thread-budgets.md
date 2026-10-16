# 0050. Automatic thread budgets: rayon = cores/2, encoder ORT sessions = cores/8

- **Status:** implemented 2026-10-17
- **Date:** 2026-10-17
- **Closes:** gaps.md Q14 and the thread-contention half of R5.

## Context

After the burn-flex migration (ADR 0049) the wake loop had four independent thread pools all
defaulting to "every core" (32 here): burn-flex's global rayon pool, the audio/video encoder ONNX
Runtime sessions (`somnus-encoding`, ORT default intra-op = all cores each), the perception/ASR ORT
sessions (already capped at cores/2 by `onnx_thread_budget`), and the main tokio/worker threads.
Measured in the live pipeline against isolated runs of the same code: `encode_speech` 1.08 s/call
vs ~54 ms, `encode_env` 450 ms vs ~70 ms, `encode_video` 48 ms vs ~25 ms (CPU EP). Mel
preprocessing is only 34 ms / 24 ms, so the cost was oversubscription, not the work. The doc
comment on `onnx_thread_budget` claimed half the cores were reserved for burn; nothing sized
rayon's pool.

## Decision

- `--rayon-threads N`: size of the global rayon pool. `0` (default) = auto = `cores / 2`.
- `--encoder-ort-threads N`: intra-op threads (inter-op 1) for each encoder session, applied via
  `somnus_encoding::set_ort_intra_threads` before the sessions are built. `0` = auto =
  `max(2, cores / 8)`.
- Explicit values override; `--rayon-threads 32 --encoder-ort-threads 32` reproduces the old
  behaviour on this machine. Both are logged at startup.

Alternatives considered: leave ORT/rayon at defaults (status quo, measured slow); one shared pool
(not possible across ORT and rayon); a smaller rayon pool (cores/4 = 8 starved burn: no gain over
the default, see pilot).

## Evidence (restored campaign snapshot, `clips/animation.mp4`, 2 sleep cycles, seed 7)

Manual sweep, cycle-1 wall time: default/default 470 s; rayon 16 / ORT default 363 s;
**rayon 16 / ORT 4: 234 s**; rayon 8 / ORT 4: 466 s. Re-run with the shipped auto defaults vs the
legacy all-cores setting: wall **336 s vs 448 s**; `ingest_bind` 38 vs 65 ms, perception ingest
43 vs 70 ms, `encode_video` 68 vs 104 ms, `encode_env` 356 vs 755 ms, `asr_decode` 338 vs 965 ms,
NREM 25.2 vs 29.3 s. Per-region MSE and the episode count were equivalent (wake is episode-count
driven).

## What would invalidate this

Run-to-run variance is large (the best manual run, 234 s, was not reproduced by the auto setting's
336 s), so the headline gain is "20-50 %", not a point estimate; one machine, one clip. A different
core count changes the optimum (the formulas only scale it). `encode_speech` per-call time was not
better with 4 threads (Whisper-tiny's GPU/CPU split); it is ~30 calls per cycle against ~3000
video frames, so the net effect is positive, but a speech-heavy workload should be re-measured.
Isolated numbers (this machine): DINOv3-small is faster on the CPU EP (23 ms) than through the
MIGraphX EP (54 ms); Whisper-tiny is the opposite (20 ms GPU vs 67 ms CPU); CLAP is equal. Moving
the video encoder off MIGraphX is a separate, un-tried option.

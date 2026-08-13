# Tech stack

> **Contents:** Committed and candidate crates for ML (ort + burn), audio (cpal, ffmpeg-next), video (nokhwa), vector retrieval (hnsw_rs), storage, concurrency (tokio, crossbeam), and observability (tracing).  
> **When to read:** Adding a dependency; checking what's already committed; understanding why candle was not adopted (burn was).
>
> **Status:** candidates are noted as such; committed choices are marked **[committed]** and backed by an ADR or research note.

## Language

**Rust** for the entire system. See `vision.md` for rationale. Key consequence: the ML, audio, and video ecosystems are less mature than Python's, and we will sometimes accept ergonomic friction or wrap C/Python libs to get capability we need.

Order of preference when the ecosystem has a gap:

1. Wrap an existing crate.
2. FFI to a mature C / CUDA / Python library.
3. Build a minimal in-tree shim.
4. Write from scratch — only when the surface is small and well-understood.

## Numerical / ML core

**Split runtime — [committed], see `research/ml-runtimes.md`:**

- **`ort`** (ONNX Runtime bindings) — **[committed]** for frozen pretrained encoder inference (Phase 2+) and perception inference during wake (ADR 0023, GPU unification). Best inference latency; MIGraphX EP for AMD RDNA4/gfx1201.
- **`burn`** (NdArray backend, CPU) — **[committed]** for cognitive-layer training (NREM replay, wake_reconstruct, AdamW optimizer). Wake inference path migrated to ORT (ADR 0023); burn is now training-only. Originally `candle` was planned; `burn` was chosen instead for better autograd ergonomics and backend agnosticism.

Candidates not adopted:

- `candle` — originally planned for cognitive-layer training; `burn` chosen instead. Removed from `[workspace.dependencies]` in the 2026-08-12 dependency sweep.
- `tch-rs` — full libtorch; fallback if `burn` can't support a needed op.
- `dfdx` — pure-Rust autograd; slower-moving.

Lower-level math (candidates, no commitment yet):

- **`ndarray`** — N-dimensional arrays, host-side.
- **`half`** — `f16` / `bf16` types (needed for FP16 storage in cold tier).

## Audio

- **`cpal`** — **[committed]** real-time microphone capture (Phase 1). Cross-platform audio I/O; callback model bridged to crossbeam channel.
- **`ffmpeg-next = "9"`** — **[committed]** audio track decoding from media files (Phase 1, via `MediaFile`). Also the video decoder; one dep for both. Bumped 8→9 on 2026-08-12 to build against the locally installed ffmpeg 9.0.
- **`rubato = "5"`** — **[committed]** sample-rate resampling at encoding time (`somnus-encoding`'s `SincResampler`: media rate → 16 kHz speech path + env-encoder rate). Bumped 3→5 in the 2026-08-12 dependency sweep; rubato 5 silently stopped enforcing the chunk-length/valid-ratio contract, so `SincResampler` enforces it itself — pinned by tests, see gaps.md §31.
- **`hound`** — WAV read/write for offline experiments only.

`symphonia` (pure-Rust audio decoder) is superseded by `ffmpeg-next` for the file-input use case. Revisit only if we need a pure-Rust build without system ffmpeg.

## Video / camera

- **`nokhwa`** (feature: `input-native`) — **[committed]** live webcam capture (Phase 1). Cross-platform v4l2/AVFoundation/MSMF wrapper.
- **`ffmpeg-next = "9"`** — **[committed]** video frame decoding from media files (Phase 1, via `MediaFile`). Shared dep with audio above.
- `image`, `imageproc` — pure-Rust image ops; candidates for preprocessing in Phase 2.

System requirement: `libavcodec`, `libavformat`, `libavutil`, `libswresample`, `libswscale` (Arch: `pacman -S ffmpeg`).

## Vector retrieval

- **`hnsw_rs`** — **[committed]** HNSW index for warm semantic memory (Phase 4). Pure-Rust implementation; chosen over `usearch` for simpler FFI story (see ADR 0006). See `research/vector-retrieval.md`.
- Flat search (no crate) — Phase 1–3 before the index is needed.
- `memmap2` — candidate for cold-tier disk access (Phase 4+).

## Storage / persistence

Three-tier hierarchy is architecturally committed (`architecture.md`); crate choices are confirmed for Phase 4:

| Tier | Medium | Crate(s) |
|------|--------|-----------|
| Hot (episodic) | CPU RAM (Vec<f32> ring) | none — plain Rust Vec |
| Warm (semantic) | RAM | `hnsw_rs` (HNSW) + plain `Vec` |
| Cold (archive) | Disk | `memmap2` + custom binary format |

Embedded KV (`redb`, `fjall`, `sled`) and SQL (`rusqlite`, `sqlx`) are candidates only if the cold-tier format outgrows hand-rolled serialization.

**Serialization:** `bitcode` — **[committed]** for warm-tier HNSW sidecar (`warm/records.bin`). Replaced unmaintained `bincode` 2.x (RUSTSEC-2025-0141); migration in commit 57c87a1, WARM_VERSION bumped 1→2 for version gating (graceful cold rebuild on format change).

## Concurrency / runtime

- **`tokio`** — **[committed]** async runtime in the binary (Phase 1). `features = ["full"]` for signals, timers, and task scheduling.
- **`crossbeam-channel`** — **[committed]** bounded sync channels bridging cpal callbacks and decoder threads to the async runtime (Phase 1). Chosen for cloneable Sender + Receiver (enables drop-oldest pattern and multi-source fan-in).
- `std::thread` — for decoder and capture threads that must not be async.
- **`rayon`** — **[committed]** parallel warm-store HNSW rebuild (`parallel_insert_slice`, gaps.md §27, 16.7× speedup).

## Observability

- **`tracing`** + **`tracing-subscriber`** — **[committed]** structured logs and spans. `features = ["env-filter", "fmt"]`. No `println!` / `eprintln!` in production paths.

## Tooling

- **`cargo`** workspace — **[committed]** multi-crate layout defined in ADR 0001.
- **`thiserror`** — **[committed]** error types in library crates.
- **`anyhow`** — **[committed]** error handling in the binary.
- `cargo-nextest` — for faster test runs.
- `criterion` — benches when we have a latency target to hit.
- `proptest` — property tests for stateful logic (memory, learning rules).
- **`clap = "4"`** (derive) — **[committed]** CLI argument parsing in the binary.

## Python tooling (committed)

A Python venv lives at `.venv/` — used only for model export and fine-tuning scripts, not for runtime inference.

- **`peft`** + **`transformers`** + **`torch`** — LoRA fine-tuning (ADR 0033; written for Qwen2.5, base model since upgraded to Qwen3-0.6B per ADR 0036 — the LoRA work itself is deferred item L). `scripts/train_lora.py`, `scripts/export_models.py`.
- **`onnxruntime`** — KV-cache model quantization to INT8. `scripts/quantize_kv.py`, `scripts/export_kv_cache.py`.

## Things we will *not* do (yet)

- Hand-roll a tensor library or autograd engine.
- Roll our own audio/video codecs.
- Build distributed-systems infrastructure. This is a single-node system.
- Adopt a "Rust for everything" stance that excludes pretrained models from the Python ecosystem — we will use them via `ort` when sensible.

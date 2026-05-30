# Tech stack

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

- **`ort`** (ONNX Runtime bindings) — **[committed]** for frozen pretrained encoder inference (Phase 2+). Best inference latency; no training support.
- **`candle`** (HuggingFace) — **[committed]** for cognitive-layer training (Phase 3+). GPU via cuBLAS/cuDNN; actively maintained.

Candidates not adopted:

- `burn` — backend-agnostic; reconsider if `candle` CUDA support becomes a problem.
- `tch-rs` — full libtorch; fallback if `candle` can't support a needed op.
- `dfdx` — pure-Rust autograd; slower-moving.

Lower-level math (candidates, no commitment yet):

- **`ndarray`** — N-dimensional arrays, host-side.
- **`half`** — `f16` / `bf16` types (needed for FP16 storage in cold tier).

## Audio

- **`cpal`** — **[committed]** real-time microphone capture (Phase 1). Cross-platform audio I/O; callback model bridged to crossbeam channel.
- **`ffmpeg-next = "8"`** — **[committed]** audio track decoding from media files (Phase 1, via `MediaFile`). Also the video decoder; one dep for both.
- **`rubato`** — candidate for sample-rate resampling at encoding time (Phase 2), when audio must be normalized to 16 kHz for Whisper.
- **`hound`** — WAV read/write for offline experiments only.

`symphonia` (pure-Rust audio decoder) is superseded by `ffmpeg-next` for the file-input use case. Revisit only if we need a pure-Rust build without system ffmpeg.

## Video / camera

- **`nokhwa`** (feature: `input-native`) — **[committed]** live webcam capture (Phase 1). Cross-platform v4l2/AVFoundation/MSMF wrapper.
- **`ffmpeg-next = "8"`** — **[committed]** video frame decoding from media files (Phase 1, via `MediaFile`). Shared dep with audio above.
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
| Hot (episodic) | VRAM / candle tensors | `candle-core` |
| Warm (semantic) | RAM | `hnsw_rs` (HNSW) + plain `Vec` |
| Cold (archive) | Disk | `memmap2` + custom binary format |

Embedded KV (`redb`, `fjall`, `sled`) and SQL (`rusqlite`, `sqlx`) are candidates only if the cold-tier format outgrows hand-rolled serialization.

## Concurrency / runtime

- **`tokio`** — **[committed]** async runtime in the binary (Phase 1). `features = ["full"]` for signals, timers, and task scheduling.
- **`crossbeam-channel`** — **[committed]** bounded sync channels bridging cpal callbacks and decoder threads to the async runtime (Phase 1). Chosen for cloneable Sender + Receiver (enables drop-oldest pattern and multi-source fan-in).
- `std::thread` — for decoder and capture threads that must not be async.
- `rayon` — candidate for compute-heavy parallel work in sleep cycle (Phase 5+).

## Observability

- **`tracing`** + **`tracing-subscriber`** — **[committed]** structured logs and spans. `features = ["env-filter", "fmt"]`. No `println!` / `eprintln!` in production paths.

## Tooling

- **`cargo`** workspace — **[committed]** multi-crate layout defined in ADR 0001.
- **`thiserror`** — **[committed]** error types in library crates.
- **`anyhow`** — **[committed]** error handling in the binary.
- `cargo-nextest` — for faster test runs.
- `criterion` — benches when we have a latency target to hit.
- `proptest` — property tests for stateful logic (memory, learning rules).
- `clap` — candidate for CLI argument parsing in the binary (Phase 1; alternative: manual `std::env::args` parse).

## Things we will *not* do (yet)

- Hand-roll a tensor library or autograd engine.
- Roll our own audio/video codecs.
- Build distributed-systems infrastructure. This is a single-node system.
- Adopt a "Rust for everything" stance that excludes pretrained models from the Python ecosystem — we will use them via `ort` when sensible.

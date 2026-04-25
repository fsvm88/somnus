# Tech stack

> **Status:** initial survey. Choices listed here are *candidates* unless explicitly noted as committed; commitments are made in ADRs.

## Language

**Rust** for the entire system. See `vision.md` for rationale. Key consequence: the ML, audio, and video ecosystems are less mature than Python's, and we will sometimes accept ergonomic friction or wrap C/Python libs to get capability we need.

Order of preference when the ecosystem has a gap:

1. Wrap an existing crate.
2. FFI to a mature C / CUDA / Python library.
3. Build a minimal in-tree shim.
4. Write from scratch — only when the surface is small and well-understood.

## Numerical / ML core

Candidates for the tensor + autograd layer:

- **`candle`** (HuggingFace) — actively developed, decent API, GPU support via CUDA/Metal. Reasonable default.
- **`burn`** — modular, multi-backend (ndarray, wgpu, candle, libtorch), thoughtful design.
- **`tch-rs`** — libtorch bindings; mature, but ties us to a heavy native dependency and PyTorch ABI.
- **`ort`** — ONNX Runtime bindings; great for *inference* of pretrained models, not training.
- **`dfdx`** — pure-Rust, type-safe shapes, but slower-moving.

Likely path: start with `candle` or `burn` for prototypes, escalate to `tch-rs` if we need PyTorch parity, use `ort` for any pretrained inference (e.g., a vision/audio encoder we don't want to retrain).

Lower-level math:

- **`ndarray`** — N-dimensional arrays, host-side.
- **`nalgebra`** — general linear algebra.
- **`faer`** — high-performance dense linear algebra.

## Audio

- **`cpal`** — cross-platform audio I/O (microphone, speaker).
- **`symphonia`** — pure-Rust decoders for offline files.
- **`rubato`** — sample-rate conversion.
- **`hound`** — WAV read/write for offline experiments.
- Features: hand-rolled mel-spectrograms initially; pretrained encoders via `ort` later if needed.

## Video / camera

- **`nokhwa`** — cross-platform webcam capture.
- **`opencv`** (Rust bindings) — broad image/video processing; heavy native dep, accept only if we use it widely.
- **`ffmpeg-next`** — for offline video files.
- **`image`**, **`imageproc`** — pure-Rust image ops.

## Storage / persistence

Undecided. Candidates:

- **Embedded KV.** `redb` (active, pure-Rust), `fjall` (LSM, pure-Rust), `sled` (mature but stalled).
- **Embedded SQL.** `rusqlite` (sync), `sqlx` with SQLite (async).
- **Vector recall.** Roll our own ANN over a flat store initially, or use `hnsw_rs` / `instant-distance`.

The choice will be driven by what episodic and semantic memory actually need; deferred until we have a use case.

## Concurrency / runtime

- **`tokio`** for async I/O (sensory ingestion, telemetry sinks).
- Dedicated threads (`std::thread`) or **`rayon`** for compute-heavy subsystems where async adds noise without benefit.
- **`crossbeam`** channels for cross-subsystem messaging where async isn't needed.

## Observability

- **`tracing`** + **`tracing-subscriber`** for structured logs and spans.
- An in-process structured **observation log** for "the system observing itself" — design TBD; see `architecture.md`.

## Tooling

- **`cargo`** workspace once we have multiple crates.
- **`cargo-nextest`** for faster test runs.
- **`criterion`** for benches when we have a target.
- **`proptest`** for property tests on stateful logic (memory, learning rules).
- **`insta`** for snapshot tests where a stable output is useful.

## Things we will *not* do (yet)

- Hand-roll a tensor library or autograd engine.
- Roll our own audio/video codecs.
- Build distributed-systems infrastructure. This is a single-node system.
- Adopt a "Rust for everything" stance that excludes pretrained models from the Python ecosystem — we will use them via `ort` when sensible.

# Tech stack

> **Contents:** Committed and candidate crates for ML (ort + burn), audio (cpal, ffmpeg-next), video (nokhwa), vector retrieval (hnsw_rs), storage, concurrency (tokio, crossbeam), and observability (tracing).  
> **When to read:** Adding a dependency; checking what's already committed; understanding why candle was not adopted (burn was).
>
> **Status:** candidates are noted as such; committed choices are marked **[committed]** and backed by an ADR or research note.

## Language

**Rust** for the entire system (see `vision.md` for rationale) — the ML/audio/video ecosystems are less mature than Python's, so some ergonomic friction or C/Python wrapping is accepted. Gap-filling order of preference: (1) wrap an existing crate, (2) FFI to a mature C/CUDA/Python library, (3) a minimal in-tree shim, (4) write from scratch only when the surface is small and well-understood.

## Numerical / ML core

**Split runtime — [committed], see `research/ml-runtimes.md`:**

- **`ort`** (ONNX Runtime bindings) — **[committed]** frozen pretrained encoder inference (Phase 2+) and wake-time perception inference (ADR 0023, GPU unification). Best inference latency; MIGraphX EP for AMD RDNA4/gfx1201.
- **`burn`** (NdArray backend, CPU) — **[committed]** cognitive-layer training (NREM replay, wake_reconstruct, AdamW). Wake inference migrated to ORT (ADR 0023) — burn is training-only now. `candle` was originally planned; `burn` chosen instead for autograd ergonomics and backend agnosticism.

**Not adopted:** `candle` (originally planned, `burn` chosen instead — removed from `[workspace.dependencies]` in the 2026-08-12 sweep); `tch-rs` (full libtorch, fallback if `burn` can't support a needed op); `dfdx` (pure-Rust autograd, slower-moving).

**Lower-level math (candidates, no commitment):** `ndarray` (N-dimensional arrays, host-side); `half` (f16/bf16, needed for cold-tier FP16 storage).

## Audio

- **`cpal`** — **[committed]** real-time microphone capture (Phase 1), cross-platform, callback bridged to a crossbeam channel.
- **`ffmpeg-next = "9"`** — **[committed]** audio+video track decoding from media files (Phase 1, `MediaFile`, one dep for both). Bumped 8→9 on 2026-08-12 to build against the locally installed ffmpeg 9.0.
- **`rubato = "5"`** — **[committed]** sample-rate resampling at encoding time (`somnus-encoding`'s `SincResampler`). Bumped 3→5 in the 2026-08-12 sweep; rubato 5 silently stopped enforcing the chunk-length/valid-ratio contract, so `SincResampler` enforces it itself — pinned by tests, gaps-resolved.md §31.
- **`hound`** — WAV read/write, offline experiments only.

`symphonia` (pure-Rust decoder) is superseded by `ffmpeg-next` for file input — revisit only if a pure-Rust build without system ffmpeg is needed.

## Video / camera

- **`nokhwa`** (`input-native`) — **[committed]** live webcam capture (Phase 1), cross-platform v4l2/AVFoundation/MSMF wrapper.
- **`ffmpeg-next = "9"`** — **[committed]** video decoding from media files, shared dep with audio above.
- `image`, `imageproc` — pure-Rust image ops, Phase 2 preprocessing candidates.

System requirement: `libavcodec`/`libavformat`/`libavutil`/`libswresample`/`libswscale` (`pacman -S ffmpeg`).

## Vector retrieval

- **`hnsw_rs`** — **[committed]** HNSW index for warm semantic memory (Phase 4), pure-Rust, chosen over `usearch` for a simpler FFI story (ADR 0006). See `research/vector-retrieval.md`.
- Flat search (no crate) — Phase 1-3 before the index is needed.
- `memmap2` — cold-tier disk access candidate (Phase 4+).

## Storage / persistence

Three-tier hierarchy is architecturally committed (`architecture.md`); crate choices confirmed for Phase 4:

| Tier | Medium | Crate(s) |
|---|---|---|
| Hot (episodic) | CPU RAM (Vec<f32> ring) | none — plain Rust Vec |
| Warm (semantic) | RAM | `hnsw_rs` (HNSW) + plain `Vec` |
| Cold (archive) | Disk | `memmap2` + custom binary format |

Embedded KV (`redb`/`fjall`/`sled`) and SQL (`rusqlite`/`sqlx`) are candidates only if the cold-tier format outgrows hand-rolled serialization.

**Serialization:** `bitcode` — **[committed]**, warm-tier HNSW sidecar (`warm/records.bin`). Replaced unmaintained `bincode` 2.x (RUSTSEC-2025-0141); migration `57c87a1`, WARM_VERSION 1→2 (graceful cold rebuild on format change).

## Concurrency / runtime

- **`tokio`** — **[committed]** async runtime in the binary (Phase 1), `features=["full"]`.
- **`crossbeam-channel`** — **[committed]** bounded sync channels bridging cpal callbacks and decoder threads to the async runtime — cloneable Sender+Receiver enables drop-oldest and multi-source fan-in.
- `std::thread` — decoder/capture threads that must not be async.
- **`rayon`** — **[committed]** parallel warm-store HNSW rebuild (`parallel_insert_slice`, gaps-resolved.md §27, 16.7× speedup).

## Observability

- **`tracing`** + **`tracing-subscriber`** — **[committed]** structured logs/spans, `features=["env-filter","fmt"]`. No `println!`/`eprintln!` in production paths.

## Tooling

- **`cargo`** workspace — **[committed]**, ADR 0001. **`thiserror`** — **[committed]**, library error types. **`anyhow`** — **[committed]**, binary error handling. **`clap="4"`** (derive) — **[committed]** CLI parsing.
- `cargo-nextest` — faster test runs. `criterion` — benches once there's a latency target. `proptest` — property tests for stateful logic (memory, learning rules).

## Python tooling (committed)

A venv at `.venv/` — model export/fine-tuning scripts only, not runtime inference. **`peft`+`transformers`+`torch`** — LoRA fine-tuning (ADR 0033; written for Qwen2.5, base model since upgraded to Qwen3-0.6B per ADR 0036, the LoRA work itself is deferred item L). `scripts/train_lora.py`, `scripts/export_models.py`. **`onnxruntime`** — KV-cache INT8 quantization, `scripts/quantize_kv.py`, `scripts/export_kv_cache.py`.

## Things we will *not* do (yet)

Hand-roll a tensor library or autograd engine. Roll our own audio/video codecs. Build distributed-systems infrastructure (single-node system). Adopt a "Rust for everything" stance excluding pretrained models from the Python ecosystem — used via `ort` when sensible.

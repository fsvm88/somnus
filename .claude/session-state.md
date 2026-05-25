# Session state — somnus

> **Last updated:** 2026-05-25
> **Last commit on main:** a0a5939 — Migrate somnus-perception and ActionDecoder/LangEmbedder from candle to burn+wgpu
> **Branch:** `main`

## Status: burn+wgpu migration complete

All candle-core/candle-nn usage removed from the codebase. The workspace no longer depends on candle in `somnus-perception` or `somnus`.

### Completed this arc (Tasks 1–7)
- **T1**: Added burn 0.21 + wgpu + autodiff to workspace deps
- **T2**: region.rs → KWtaLayer with `#[derive(Module, Debug)]`, k-WTA forward, homeostatic duty-cycle tracking
- **T3**: sae.rs → SaeDecoder with mse_loss_batch
- **T4**: binding.rs → BindingArea with `activation::softmax`
- **T5**: lib.rs → PerceptionModel<B> + PerceptionProcessor (full rewrite, ~600 lines)
- **T6**: action.rs ActionDecoder projection → burn TrainB ProjModel; embed_table → Vec<f32> via safetensors; main.rs LangEmbedder → pure-Rust mean-pool
- **T7**: Removed candle from somnus-perception/Cargo.toml and somnus/Cargo.toml; added safetensors = "0.7"

### Build/test status (as of a0a5939)
- `cargo build --workspace` — clean
- `cargo clippy --workspace -- -D warnings` — clean
- `cargo test -p somnus-perception` — 12/12 pass (8 unit + 4 GPU integration)
- `cargo test --workspace` — all pass

## Key burn 0.21 API facts (for future sessions)
- `type TrainB = Autodiff<Wgpu>` — autodiff backend for training
- `#[derive(Module, Debug)]` on model structs — don't add explicit Clone (Module derive provides it)
- `#[module(skip)]` for non-tensor fields (duty_cycle: Vec<f32>, n_units: usize, region: RegionId)
- `PercOptim = OptimizerAdaptor<AdamW, PerceptionModel<TrainB>, TrainB>` — import path: `burn::optim::adaptor::OptimizerAdaptor`
- `Param::id` is a Copy field (not a method, no `.clone()` needed)
- `activation::softmax(tensor, dim)` — not `.softmax()` method on Tensor
- `save_file` consumes model → clone via `.as_ref().unwrap().clone()` first
- Take/replace pattern for optimizer step (burn value-semantics optimizer)
- `TensorData::new(vec, shape).to_vec::<f32>()` for CPU readback

## Next queued work
- Evaluate `campaign_speech_first` results: check asr_count/cycle, lang_cos trajectory
- Validate that GPU is actually being used during training (burn's wgpu backend should dispatch to ROCm/Vulkan on RX 9070 XT)
- Phase 9b (mic/webcam live capture) — pending
- Phase 11b (creative REM) — pending

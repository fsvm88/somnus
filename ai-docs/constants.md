# Constants — fast reference

> **Contents:** Every numeric constant and default that appears in `architecture.md`/`architecture.svg`, plus the neuromodulation (ADR 0024, S1-S7) formulas and defaults. Each entry cites its source file:line so this stays checkable, not just trusted.
> **When to read:** Need a dimension, threshold, or formula fast without grepping the crates. Also the reconciliation point when `architecture.svg` is updated — see "Keeping this in sync" below.
> **Kind column:** `const` = compile-time Rust constant, fixed until code changes. `CLI` = a `--flag` default, overridable per-run without a rebuild. `derived` = computed from other constants, shown with its formula.

## Keeping this in sync

CLAUDE.md requires `architecture.svg` to stay in sync with `architecture.md` whenever architecture or key parameters change. This file is the third leg of that sync: **whenever a constant in this table changes, update all three in the same commit** — the Rust source (authoritative), this file, and `architecture.svg` if the constant appears there (dimensions, region count, sparsity, memory-tier sizing, neuromodulation formulas). If a constant here doesn't need to appear in the diagram, it's still tracked here for the CLI/ADR reference value.

Verified against code 2026-08-16 (this file's creation pass). If a value here disagrees with a currently-running binary, trust the binary — re-verify and fix this file.

---

## Core dimensions

| Name | Value | Kind | Source | Note |
|---|---|---|---|---|
| `N_UNITS` | 1024 | const | `somnus-perception/src/region.rs:10` | Per-region k-WTA baseline width. Architectural baseline, **not current production size** — see "Current live scale" below. |
| `K_ACTIVE` | 20 | const | `somnus-perception/src/region.rs:11` | k-WTA active units at baseline (2% sparsity). |
| `EPISODE_DIM` | 640 | const | `somnus-perception/src/binding.rs:9` | Binding-area output episode vector width. Also `somnus/src/action.rs:36`, `somnus-memory/src/cold.rs:17` (`COLD_DIM`) — must stay identical across crates. |
| `D_PROJ` | 128 | const (derived: `EPISODE_DIM/5`) | `somnus-perception/src/binding.rs` (`assert!(5*D_PROJ==EPISODE_DIM)`) | Per-region projection width into the binding area. **Fixed, not scaled with N_UNITS** — flagged open question, `research/gpu-unification-design.md` §13. |
| `GRU_HIDDEN` | 256 | const | `somnus-perception/src/binding.rs:24`, `somnus-perception/src/onnx_session.rs:14` | Binding-area temporal-context GRU hidden size. |
| `N_HEADS` | 4 | const | `somnus-perception/src/binding.rs:17` | Cross-attention heads in the binding area. `D_HEAD = D_PROJ/N_HEADS = 32`. |
| `LANG_HIDDEN` | 1024 | const | `somnus-perception/src/lib.rs:149` | Qwen3-0.6B hidden size (896 for the retired Qwen2.5, ADR 0035/0036). Also `somnus/src/action.rs:34` (`LM_HIDDEN`), `somnus/src/lib.rs:698` (`LANG_DIM`) — all three must match. |
| `ENV_DIM` | 512 (960 with `mn10_env` feature) | const | `somnus-perception/src/lib.rs:155/157` | Audio-env region width: CLAP-HTSAT (default) vs. EfficientAT-mn10 (opt-in feature, rejected as default per ADR 0046). |
| `REGION_DIMS` | `[384, ENV_DIM, 384, 384, LANG_HIDDEN]` | derived | `somnus-perception/src/lib.rs:160` | Per-region input widths: audio-speech, audio-env, visual, symbolic, language. |
| `CROSS_MODAL_DIM` | `384+ENV_DIM+384+LANG_HIDDEN` (2296 default) | derived | `somnus-perception/src/lib.rs:164` | Concatenated cross-modal projection input width. |
| `N_REGIONS` | 5 | const | `somnus/src/action.rs:35`, `somnus/src/sleep.rs:227` (`N_EPISODE_REGIONS`) | audio-speech[0], audio-env[1], visual[2], symbolic[3], language[4]. |
| `WHISPER_ENC_DIM` | 384 | const | `somnus/src/whisper_asr.rs:13` | Whisper-tiny encoder output width. |

## Encoders (frozen, ORT/MIGraphX)

| Name | Value | Kind | Source | Note |
|---|---|---|---|---|
| Whisper-tiny sample window | 480,000 (30s @ 16kHz) | const | `somnus/src/lib.rs:553`, `somnus-encoding/src/audio.rs:42` | `WHISPER_SAMPLES`. |
| CLAP-HTSAT env window | 480,000 @ 48kHz | const | `somnus/src/lib.rs:557-559` | Default (non-`mn10_env`) path. |
| mn10 env window | 320,000 @ 32kHz | const | `somnus/src/lib.rs:561-563` | `mn10_env` feature path. |
| `WHISPER_ENC_SEQ` | 1500 | const | `somnus/src/whisper_asr.rs:12` | Whisper encoder sequence length. |
| Visual encoder | DINOv3-ViT-S, 384-dim | — | ADR 0045 | Replaced DINOv2-S/14 2026-07-05. |

## Sparse coding / k-WTA growth

| Name | Value | Kind | Source | Note |
|---|---|---|---|---|
| `scaled_k(n_units)` | `round(n_units × K_ACTIVE / N_UNITS)`, min 1 | derived (function) | `somnus-perception/src/region.rs:20-22` | Preserves the 2% sparsity ratio as regions grow (§17). |
| `BOOST_STRENGTH` | 1.0 | const | `somnus-perception/src/region.rs:13` | Duty-cycle boost magnitude. |
| `DUTY_CYCLE_ALPHA` | 1/1000 | const | `somnus-perception/src/region.rs:14` | EWMA rate for per-unit duty-cycle tracking — sets the ~1000+-round recovery timescale noted in gaps.md §42. |
| `GROWTH_BATCH_MIN` / `MAX` | 16 / 128 | const | `somnus/src/sleep.rs:36-37` | Novelty-proportional growth batch size (§19). |
| `GROWTH_THRESHOLD` | 0.1 | const | `somnus/src/sleep.rs:26` | |
| `GROWTH_MIN_SAMPLES` | 16 | const | `somnus/src/sleep.rs:27` | |
| `GIST_GROWTH_THRESHOLD` | 0.85 | const | `somnus/src/sleep.rs:240` | REM growth trigger: `within_session_gist_rate < 0.85`. |
| `--growth-cooldown` | 3 | CLI | `somnus/src/lib.rs` (`growth_cooldown`) | Cycles between growth events, same region. |
| `PRUNE_GIST_THRESHOLD` | 0.8 | const | `somnus/src/sleep.rs:242` | Pruning eligibility gate: `gist_recall_rate ≥ 0.8`. |
| `--prune-fraction` | **0.0 (disabled)** | CLI | `somnus/src/lib.rs:411` | ADR 0018's original default was 0.5; ADR 0026's A/B (env-region prior cosine regressed −0.065, no recovery) disabled it. `roadmap.md`'s Phase 12c text corrected 2026-08-16 to match. |
| `--prune-min-grown` | 64 | CLI | `somnus/src/lib.rs` (`prune_min_grown`) | Minimum grown units before pruning is eligible. |

## Binding area / action decoder

| Name | Value | Kind | Source | Note |
|---|---|---|---|---|
| `--working-memory-size` (WM ring) | **8** | CLI | `somnus/src/lib.rs:189-199` | Hardened 2026-08-16 (ADR 0047) — was 0/disabled; `architecture.svg` had already said "WM=8" ahead of the code, since that's the validated pilot value (roadmap.md §10d: WM=8 vs WM=0, output_mse 40-50% lower) every real campaign has always run at. `0` still disables the ring. |
| `--episodic-capacity` | 4096 | CLI | `somnus/src/lib.rs` (`episodic_capacity`) | Hot-tier ring buffer capacity. |
| `PROJ_HIDDEN` | 1024 | const | `somnus/src/action.rs:220` | MLP projection hidden width (ADR 0032, item M). |
| `KV_SIZE` / `N_KV_LAYERS` / `N_KV_HEADS` / `KV_HEAD_DIM` | 2048 / 24 / 2 / 64 | const | `somnus/src/action.rs:52-55` | Qwen3-0.6B KV-cache shape (ADR 0034). |
| `MAX_GEN_SEQ` | 1024 | const | `somnus/src/action.rs:46` | |
| `EOS_TOKEN_ID` | 151645 (`<\|im_end\|>`) | const | `somnus/src/action.rs:39` | Qwen3; was 151643 for the retired Qwen2.5. |

## Memory tiers

| Name | Value | Kind | Source | Note |
|---|---|---|---|---|
| `HNSW_M` | 16 | const | `somnus-memory/src/semantic.rs:15` | |
| `HNSW_MAX_LAYERS` | 16 | const | `somnus-memory/src/semantic.rs:17` | |
| `HNSW_EF_C` (construction ef) | 100 | const | `somnus-memory/src/semantic.rs:22` | |
| `--recall-ef` (search ef) | 16 | CLI | `somnus/src/lib.rs:202` | Overrides the struct's `SEARCH_EF=64` init constant (`semantic.rs:23`) at startup via `set_search_ef` — **16 is the effective production value**, not 64. Calibrated at scale per ADR 0024 S5. |
| `--max-warm-records` | **200000** | CLI | `somnus/src/lib.rs:201-208`, formula at `somnus/src/lib.rs:1105-1114` | Hardened 2026-08-16 (ADR 0047) — every real campaign has hardcoded this since 2026-06-17, ending the completed 10x campaign at 151,725 warm records (comfortably under cap). `0` opts back into the ADR 0024 S5 auto-derive formula: `free_ram_bytes×0.35/2500`, clamped `[1_000, 1_000_000]`. |
| `--max-cold-records` | 0 → auto: `free_disk_bytes×0.20/RECORD_BYTES` | CLI | `somnus/src/lib.rs:1117-1136` | `RECORD_BYTES=661` (`somnus-memory/src/cold.rs:34`). Falls back to unlimited (0) if the disk lookup fails (e.g. btrfs), per ADR 0024. |
| `PQ_M` / `PQ_K` / `PQ_SUB_DIM` | 8 / 256 / 80 | const | `somnus-memory/src/pq.rs:26-28` | Product-quantization compression, 27.5× reduction (roadmap.md 11a). |
| `RECORD_BYTES` (INT8 cold) | 661 | const | `somnus-memory/src/cold.rs:34` | 8(id)+8(ts)+4(salience)+640(vec)+1(source... )+... |

## Neuromodulation (ADR 0024, S1-S7)

All formulas verified against `ai-docs/decisions/0024-neuromodulatory-self-regulation.md` 2026-08-16 — the version below is the diagram-simplified form; the ADR has the exact clamped/EWMA-α form.

| # | Mechanism | Formula (diagram-simplified) | Defaults | Source |
|---|---|---|---|---|
| S1 | Adaptive salience gate | `τ_s = μ_L2 + 0.5·σ_L2`, clamped `[0.3, 3.0]` | cold-start `τ_s=0.8` until N=100; EWMA α=0.01 | ADR 0024 §S1, `somnus/src/lib.rs:2841` (`SAL_EWMA_COLD_START`) |
| S2 | Settle-rate modulation | fire settle when `prior_cosine < EWMA − 0.1` | cold-start `C=0.85` until N=200; EWMA α=0.005 | ADR 0024 §S2 |
| S3 | Wakefulness-pressure homeostat | `P += novelty/τ_s` per episode; sleep at `P≥1.0`; NREM dissipates `P −= consolidation_quality∈[0,0.5]` | `P_THRESHOLD=1.0`, `P_MAX=3.0`, cold-start capacity-fill trigger (`--sleep-high-water 0.8`) active until `sal_n≥500` (`P_COLD_START_N`) and `wake_episodes≥120` (`P_MIN_EPISODES`) | ADR 0024 §S3, `somnus/src/lib.rs:2844-2861` |
| S4 | Prior-beta scale-up | β: base→0.05 when 3-cycle-ring cosine all `>0.65`; hysteresis band `[0.40, 0.65]` | `--prior-beta` **CLI default 0.01** (fixed 2026-08-16 — was 0.0, see "CLI-default sweep" below) | ADR 0024 §S4, `somnus/src/lib.rs:154-158` |
| S5 | Startup-queryable priors | see Memory tiers table (`--max-warm-records`, `--max-cold-records`, `--recall-ef`) | — | ADR 0024 §S5 |
| S6 | Adaptive wake ceiling | `max_wake = max(floor, 3×wake_ewma_mu)` after N≥3 cycles | floor = `--max-wake-secs` default 600; EWMA α=0.3/cycle | ADR 0024 §S6 |
| S7 | Wake-interval throttle | stagnation (<1% `mean_output_mse_wake` improvement, 3 consecutive cycles) decrements `--wake-reconstruct-interval` toward 1 | CLI default 5 | ADR 0024 §S7, `somnus/src/lib.rs:238` |

## CLI-default sweep (2026-08-16) — resolved, ADR 0047

`--prior-beta`'s discrepancy (default said 0.0/disabled, but the one real production harness — `scripts/train_campaign.sh` — has unconditionally passed 0.01 since `a5d20c7`, 2026-06-14) turned out to be a genuine drift: `git log -- scripts/train_campaign.sh` shows the fix from ADR 0024's S4 amendment landed in the *script's* env-var default only, never in the Rust flag's own `default_value`. **Fixed 2026-08-16** — `--prior-beta` now defaults to 0.01 in code.

Prompted by that find, every CLI default was swept against every unconditional (non-env-var, non-A/B-arm) flag pass in `scripts/train_campaign.sh`. Four more same-shape candidates were found and, per explicit user instruction, all hardened into the default together — see **ADR 0047** for the full evidence/reasoning per flag:

| Flag | Old default | New default (2026-08-16) |
|---|---|---|
| `--working-memory-size` | 0 | **8** |
| `--settle-steps` | 0 | **1** |
| `--max-warm-records` | 0 (auto-derive) | **200000** |
| `--lang-use-lm` | false | **true** |

Checked and ruled out (legitimate campaign-specific tuning, not a stale default — left unchanged): `--video-stride` (ADR 0024 explicitly buckets this as "permanent manual, content-rate tradeoff," not something that should be auto-set), `--action-distill-only` (doc comment explicitly scopes it to "training campaigns where generation output is not needed" — a real mode switch, not a proven-better setting for the interactive/eval path too). `--language`/`--action`/`--action-gain`/`--onnx-migraphx`/`--wake-recon-async` already default `true` and are only passed for explicitness. `--replay-batch-size`, `--recall-ef`, `--wake-reconstruct-interval`, `--nrem-replay-epochs` all already matched their CLI defaults (the script comments say so explicitly for the latter two).

`cargo test -p somnus` (101 tests), `cargo clippy --all-targets -- -D warnings`, and `cargo fmt --check` all clean after all five default changes (prior-beta + these four).

## Current live scale (as of the 2026-08-15 10x campaign)

These are **not** architecture-defining constants — they're the current trained-model state, included here because `architecture.svg`'s sidebar shows them and they drift fast. Full detail: `engineering-log.md` §34.

| Name | Value | Note |
|---|---|---|
| Region sizes | `[9821]×5` | 101.7% of the 9,660-unit (honeybee-order) target, grown from `[1024]×5` baseline. |
| `scaled_k(9821)` | 192 | `round(9821×20/1024)`, per the `scaled_k` formula above. |
| Campaign span | 791 cycles / 188 growth events / 82 sessions | 2026-07-12 → 2026-08-15. |
| `mean_output_mse` | 0.0430 → 0.0112 (first-vs-last-10, −74.0%) | |

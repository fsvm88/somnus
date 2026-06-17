# Session state — somnus

> **Last updated:** 2026-06-17
> **Last commit on main:** 02a241a fix(campaign): raise CAPACITY default to 8192 and add --max-warm-records 200000
> **Branch:** `main`

## Status: Wiki-on-mixed complete. Next: implement B (--lang-use-lm) + A/B test.

---

## Completed this session (2026-06-17)

| Item | What changed |
|------|-------------|
| Wiki-on-mixed campaign | s31–40 × 2 cycles; MSE 0.049→0.028 (↓40%); p[4]/lang noisy, peak 0.564 (barely above 0.557 baseline); text injection NOT additive for p_lang |
| Cycle-2 starvation fix | `train_campaign.sh`: CAPACITY default 4096 → 8192; `--max-warm-records 200000` added to both branches |

---

## Wiki-on-mixed results (s31–40)

| s | c | lang | mse | p[4] lang | notes |
|---|---|------|-----|-----------|-------|
| 31 | 1 | 66 | 0.0682 | 0.217 | initial drop from 0.557 baseline |
| 31 | 2 | 27 | 0.0750 | 0.278 | |
| 32 | 1 | 70 | 0.0625 | 0.476 | |
| 32 | 2 | 21 | 0.0363 | 0.296 | |
| 33 | 1 | 46 | 0.0465 | 0.544 | |
| 34 | 1 | 47 | 0.0523 | 0.465 | |
| 35 | 1 | 73 | 0.0413 | 0.383 | |
| 36 | 1 | 49 | 0.0503 | 0.507 | |
| 37 | 1 | 77 | 0.0420 | 0.205 | |
| 38 | 1 | 99 | 0.0427 | 0.193 | |
| 39 | 1 | 81 | 0.0566 | **0.564** | new peak, barely above baseline |
| 40 | 1 | 89 | 0.0428 | 0.414 | |
| 40 | 2 | 18 | 0.0284 | 0.353 | MSE best; lang still noisy |

**Findings:** MSE improved 40% (0.049→0.028). p[4]/lang oscillated 0.193–0.564 with no cumulative trend. 0.557 H3H3 baseline was not reliably exceeded. Representation quality (embed-table mean-pool) is the next hypothesis.

---

## Completed previous session (2026-06-16)

| Item | What changed |
|------|-------------|
| extract_wiki.py table-leak fix | Strip lines starting with `\|` or `!` after `{|...\|}` removal; re-extracted 271k articles, 294 files, 297 MB |
| Profiling (running binary) | `perf record`: 97.45% in libonnxruntime.so, 1517% CPU, GPU at 3% util |
| MIGraphX LM investigation | `qwen25_lm.onnx` silently falls back to CPU: dynamic `seq_len` (Range/Shape ops) prevents MIGraphX static compilation. DINOv2 hits 72% GPU; qwen stays at 3% — confirmed via Python ORT + rocm-smi |
| Real bottleneck found | LM autoregressive generation ("pre-sleep drain") = 205 calls × ~14s = 2870s/session = 96% of wall time |
| `--action-distill-only` flag | New flag in `main.rs`: sets `qtx=None` so stdin text feeds `lang_rx` (distillation) only, not `query_rx` (generation). Hardened into `train_campaign.sh` |
| Exit code 120 fix | `train_campaign.sh`: whitelist exit 120 (pipefail picks up inject_text.py SIGPIPE when somnus exits cleanly) alongside 139 |
| Wiki campaign 2 killed | 30 sessions × 3 cycles; 1 session completed; cycles took 1000–1200s each due to generation |
| Wiki campaign 3 | 11 sessions × 2 cycles, TEXT_RATE=1, distill-only, cold start, `memory/wiki_campaign3/` |

---

## Wiki campaign 3 results (full)

| s | c | lang | mse | p_lang | p_vis |
|---|---|------|-----|--------|-------|
| 1 | 1 | 303 | 0.2265 | 0.202 | 0.665 |
| 1 | 2 | 84 | 0.1489 | 0.308 | 0.703 |
| 2 | 1 | 74 | 0.2569 | 0.183 | 0.628 |
| 2 | 2 | 15 | 0.2440 | 0.163 | 0.703 |
| 3 | 1 | 80 | 0.1943 | 0.248 | 0.674 |
| 3 | 2 | 31 | 0.1998 | 0.227 | 0.419 |
| 4 | 1 | 91 | 0.1975 | 0.192 | 0.729 |
| 4 | 2 | 10 | 0.1870 | 0.329 | 0.718 |
| 5 | 1 | 75 | 0.1405 | **0.504** | 0.769 |
| 5 | 2 | 14 | 0.1773 | 0.324 | 0.712 |
| 6 | 1 | 66 | 0.1343 | 0.357 | 0.805 |
| 6 | 2 | 13 | 0.2446 | 0.162 | 0.739 |
| 7 | 1 | 48 | 0.1102 | 0.428 | 0.820 |
| 7 | 2 | 15 | 0.2126 | 0.236 | 0.730 |
| 8 | 1 | 88 | 0.1328 | 0.208 | 0.795 |
| 8 | 2 | 11 | 0.1376 | 0.262 | 0.719 |
| 9 | 1 | 88 | 0.1248 | 0.311 | 0.827 |
| 9 | 2 | 22 | 0.1507 | 0.219 | 0.800 |
| 10 | 1 | 45 | 0.1112 | 0.300 | 0.827 |
| 10 | 2 | 11 | 0.2577 | 0.212 | 0.750 |
| 11 | 1 | — | — | — | — |
| 11 | 2 | — | 0.1018 | 0.106 | — |

**Key findings:**
- MSE: clear downward trend 0.2265 → 0.1018 (halved) — system is genuinely converging
- p_lang: peaked at 0.504 (s5c1), oscillated 0.183–0.504, no cumulative drift across sessions
- Cycle-2 starvation: lang_ingest drops from 66–303 (cycle 1) to 10–31 (cycle 2) — warm memory fills at capacity 4096, slowing HNSW, reducing time for text ingestion
- Episodes capped at 120/cycle from session 2 onward (capacity ceiling)
- p_vis healthy 0.628–0.827 throughout
- Mixed campaign baseline at s30: p_lang=0.557. Cold-start wiki campaign peaks at 0.504 by s5 but doesn't compound

---

## Queued next (top priority)

### 1. Implement B: --lang-use-lm flag (LM-hidden language embeddings)

**Why:** p_lang plateaued at 0.557 with embed-table mean-pool. LM forward-pass hidden states are contextualized — this ablation confirms/refutes whether representation quality is the bottleneck.

**What:** Replace `LangEmbedder.embed()` (static embed-table mean-pool) with `ActionDecoder.text_hidden_states()` (single LM forward pass, mean-pool over hidden states, L2-normalized). Same 896 dims, same interface — richer representation.

**How (concrete changes to `crates/somnus/src/main.rs`):**
1. Add `--lang-use-lm` bool flag to `Args`
2. Add `fn tokenize(&self, text: &str) -> Result<Vec<u32>>` to `LangEmbedder` (expose tokenization without embedding)
3. Modify `ingest_language_line` signature: add `lm_emb: Option<&[f32]>` param; when `Some`, skip `LangEmbedder.embed()` and use it directly
4. At both `ingest_language_line` call sites (wake loop, ~lines 1475 and 2152): if `args.lang_use_lm && decoder.is_some()`, call `LangEmbedder.tokenize()` → `decoder.text_hidden_states(ids)` → pass result as `lm_emb`
5. No changes to `action.rs` — `text_hidden_states()` already exists and is correct

**A/B validation:** fresh cold-start runs, 5 sessions × 2 cycles each:
- A: current (`--lang-use-lm` off) → p_lang baseline
- B: `--lang-use-lm` on → p_lang with LM embeddings
- If B > A: representation was the bottleneck; keep the flag, harden into train_campaign.sh
- If B ≈ A: data diversity is the limiter; defer further language work; proceed to D

### 2. Implement D: full differentiable SAE — defer until B result known

**Context:** Encoder weights ARE already being trained (gradient flows through masked z). The k-WTA mask acts as a straight-through estimator naturally. Two options:
- **(a) Add L1 coding term** to existing k-WTA sleep loss: `loss = mse_loss + λ * |sparse_z|.mean()`. Marginal win — cardinality is already fixed at K_ACTIVE=20, so L1 only shrinks active magnitudes.
- **(b) Replace k-WTA with ReLU+L1 encoder** (principled SAE). Binding uses `to_dense_n(N_UNITS)` → fixed-k NOT assumed downstream. Architecturally safe. Breaks duty-cycle boosting; would need L1 λ tuning to maintain ~20 active units.

**Decision rule:** If B shows p_lang improvement → representation matters → do D option (b). If B is flat → data is the limiter → D is unlikely to help p_lang either; reconsider priority.

---

## Completed this session (2026-06-17)

| Item | What changed |
|------|-------------|
| Wiki-on-mixed campaign | 10 sessions × 2 cycles (s31–40), `memory/mixed_campaign/campaign_20260617_122244/` |
| Cycle-2 starvation fix | `train_campaign.sh`: CAPACITY default 4096 → 8192; `--max-warm-records 200000` added to both branches |

---

## Completed previous session (2026-06-15)

| Item | What changed |
|------|-------------|
| Shorts download | `scripts/download_shorts.sh` — 84 clips, 144 min, 20 categories → `clips/shorts/` |
| Shorts-only campaign | 30 sessions, `memory/shorts_campaign/` — language region dead (0 ASR), output_mse 0.498 |
| H3 Podcast download | `scripts/download_h3h3.sh` — 5 episodes (32–215 min) → `clips/h3h3/` |
| Mixed campaign | 30 sessions, `memory/mixed_campaign/` — language region alive; output_mse 0.049; Visual prior_cosine 0.553→0.848 |
| Wikipedia dump | `data/wiki_dump/simplewiki-latest-pages-articles.xml.bz2` (334 MB) |
| Wikipedia extraction | `scripts/extract_wiki.py` (stdlib, no deps) → 272k articles, 319 files, 324 MB in `data/wiki_text/` |
| Wiki campaign (killed) | Started `memory/wiki_campaign/`; only 1 session in 17 min — TEXT_RATE=5 too fast; killed cleanly |

---

## ADR 0024 implementation sequence — all done

All P1/P2/S1–S7 items complete.

---

## Open issues / queued (ADR scan 2026-06-14/15)

### Tier 1 — trigger-gated (not yet met)

| # | Source | Item | Trigger |
|---|--------|------|---------|
| A | ADR 0006 | Cold archive O(1) promotion index | `promote()` called AND cold > 500k records |
| B | ADR 0019 / 0014 | Language last-token pooling | `language prior_cosine` plateaus AND bottleneck confirmed as input representation |
| C | ADR 0006 | Per-region sparse activation storage | Replay analysis shows episode vectors insufficient |

### Tier 2 — conditional / quality-gated

| # | Source | Item | Trigger |
|---|--------|------|---------|
| D | ADR 0005 | Full differentiable SAE | `output_mse` converges (0.1018 in wiki campaign — approaching) |
| E | ADR 0009 | Separate optimizer for output heads | Conflicting gradient directions observed |
| F | ADR 0010 | Post-k-WTA prior injection | Pre-k-WTA injection shows poor `prior_cosine` uptake |
| G | ADR 0010 | Learned per-region scalar gate | `prior_cosine` stable; per-region β tuning shows benefit |
| H | ADR 0018 | Activation-frequency pruning | Magnitude pruning false-positives; A/B required first |
| I | ADR 0018 | Knowledge distillation compression | Grown regions exceed 2× baseline |

### Tier 3 — model swaps / Phase 7+

| # | Source | Item |
|---|--------|------|
| J | ADR 0003 | Whisper-tiny → Whisper-base / WavLM-base |
| K | ADR 0003 | DINOv2-S → SigLIP-B |
| L | ADR 0012 | LoRA fine-tuning of Qwen2.5 |
| M | ADR 0012 | MLP projection (640→5×896) |
| N | ADR 0014 | Visual/audio region grounding |

---

## Campaign results summary

| Campaign | Sessions | output_mse (final) | p_lang (peak) | Notes |
|---|---|---|---|---|
| shorts-only | 30 | 0.498 | None (dead) | No ASR in short clips |
| mixed (shorts+H3) | 30 | 0.049 | 0.557 | Language alive; all regions growing |
| wiki campaign 2 | 1 | — | — | Killed; generation blocking (96% wall time) |
| wiki campaign 3 | 11 (cold) | 0.1018 | 0.504 | Distill-only; p_lang noisy, MSE converging |

---

## Key facts

- `--replay-batch-size 128` hardcoded in `train_campaign.sh`
- `--max-warm-records 200000` hardcoded in `train_campaign.sh` (fixes cycle-2 HNSW starvation)
- CAPACITY default is now 8192 (was 4096); overrideable via env var
- `--working-memory-size 8` hardcoded in `train_campaign.sh`
- `--action-distill-only` now hardcoded in `train_campaign.sh`
- `--asr-min-chars 5` for noisy-audio clips
- `--max-warm-records 200000` recommended for long runs — was NOT set in wiki campaign 3 (may explain cycle-2 starvation)
- Wake schedule trigger at 602s/cycle; N_UNITS=1024, K_ACTIVE=20, D_PROJ=128, EPISODE_DIM=640
- GRU_INPUT=640, GRU_HIDDEN=256, CROSS_MODAL_DIM=2176
- burn GRU: reset_after=True → ONNX linear_before_reset=1
- burn Linear weight: [n_in, n_out] (transpose of ONNX/PyTorch)
- DUTY_CYCLE_ALPHA=0.001, BOOST_STRENGTH=1.0, TARGET_DC=20/1024≈0.0195
- settle_beta=0.5, settle_every=1 (defaults)
- SEARCH_EF → `--recall-ef 16` (confirmed at scale)
- wake_reconstruct_interval=5 (default); S7 decrements toward 1 on MSE stagnation
- SAL_EWMA_COLD_START=0.8, α=0.01 for S1/S2 EWMAs
- S3: P_THRESHOLD=1.0, P_MAX=3.0, P_COLD_START_N=500, P_MIN_EPISODES=120
- S4: base=0.01 (default), scale target=0.05, enable threshold=0.65 (3 cycles), disable threshold=0.40
- `PRIOR_BETA=0.01` is now the campaign default
- S7: S7_MSE_ALPHA=0.3, stagnation threshold=1% improvement, consecutive=3, decrement step=1
- COLD_RECORD_BYTES=665 (8+8+4+640+1+4); SMNCOLD3; P1a uses 20% of free disk
- Mixed campaign memory: `memory/mixed_campaign/campaign_20260615_230556/` (best result so far, p_lang=0.557)
- Wikipedia corpus: `data/wiki_text/` (294 files, 297 MB, 271k Simple English articles, table-leak fixed)
- `qwen25_lm.onnx` runs on CPU only (dynamic seq_len via Range/Shape ops blocks MIGraphX)
- LM generation was 96% of campaign wall time — fixed via `--action-distill-only`
- Exit code 120 from pipefail when somnus exits cleanly — whitelisted in train_campaign.sh

---

## Performance knobs

| Flag | Default | Effect |
|---|---|---|
| `--settle-steps 0` | 0 | Disable settle entirely |
| `--settle-every N` | 1 | Subsample settle |
| `--settle-gate F` | 0.85 | Cold-start prior for S2 adaptive gate |
| `--recall-ef N` | 16 | Lower → faster HNSW queries |
| `--wake-recon-async` | true | Offload wake_reconstruct to worker |
| `--wake-reconstruct-interval N` | 5 | S7 adapts this downward on MSE stagnation |
| `--video-stride N` | 1 | Encode every Nth video frame |
| `--action-distill-only` | false | Skip generation, keep LM distillation |
| `--max-warm-records N` | unlimited | Cap warm memory to keep HNSW fast |

---

## MIGraphX constraint (hard)

Never expose ScatterElements or INT64 outputs. Two-graph split, topk/masking on CPU.
`qwen25_lm.onnx` also blocked (dynamic seq_len) — would need fixed-shape re-export + attention_mask input.

# Session state — somnus

> **Last updated:** 2026-06-21
> **Branch:** `main`

## Status: Task 1 (GPU generation via qwen25_lm_fixed.onnx) and Task 2 (MLP projection ADR 0032) both complete. Next: A/B to verify proj_loss improvement with MLP, then L (LoRA, trigger-gated).

---

## Completed this session (2026-06-21)

| Item | What changed | Result |
|------|-------------|--------|
| **GPU generation (Task 1)** | `qwen25_lm_fixed.onnx` exported at `[1, 1024, 896]` (no LM head); Rust extracts `hidden[seq_len-1]` and computes logits via `embed_table^T` dot. `--gpu-generation` flag gates it. ADR 0031. Commits c7f43b4, 332c7c1, 3d619ce. | Token-match A/B confirmed correctness. Timing: GPU 2.1s/token vs CPU 0.14s/token on gfx1201 (RDNA 4) — GPU is 14× slower due to immature MIGraphX support. `--gpu-generation` is opt-in; default uses CPU. ADR updated with honest timing. |
| **MLP projection (Task 2)** | `ProjModel` updated to 2-layer MLP: fc1 [640→1024] + ReLU + fc2 [1024→4480]. `proj_weights_flat()` returns [W1\|W2]. `gain_from_episode()` updated for 2-layer forward. Old `decoder_weights.bin` incompatible — graceful cold-start with ADR 0032 log message. ADR 0032. Commit f9e5faa. | proj_loss plateau at ~0.0007 (sessions 46–50) triggered M. MLP should reduce proj_loss further; evaluate over 3–5 sessions post-deploy via `mean_proj_loss`. Persistent hallucination is not expected to change — that's L's domain. |
| **Completed this session (2026-06-19–20)** | ORT log suppression (a874166), live test (R1+R2+R3 generating 42 tokens, complete hallucination as expected). | — |

---

## Completed this session (2026-06-17)

| Item | What changed | Result |
|------|-------------|--------|
| **B: --lang-use-lm** | `ActionDecoder.text_hidden_states()` wired into `ingest_language_line`; `--lang-use-lm` flag in `main.rs`; hardened into `train_campaign.sh` | A/B confirmed 4× p_lang improvement (0.557→0.85–0.91). Commit 7eea3ae. |
| **lm-on-mixed campaign** | s41–50 × 2 cycles continuing mixed_campaign memory | p[4] peaked 0.910 (s46c1), stable 0.80–0.91 from s42+; MSE 0.026–0.053 |
| **D: SAE L1 coding loss** | `--sae-l1-coeff` flag (default 0.0) in `main.rs`; `sae_l1_coeff` field + `set_sae_l1_coeff()` on `PerceptionProcessor`; L1 term in `fused_replay_wake_step_batch` NREM loop | A/B result: λ=0.1 made MSE 27% WORSE (control 0.315 vs treatment 0.398 avg s3–5). Item D closed. ADR 0025. |
| **J/K/L/M item review** | Checked per-region cosines at steady state (s43–50 mixed_campaign). Speech=0.973 (strongest), language=0.843, visual=0.885. J (Whisper-base) not applicable — speech is already best region. K not triggered (Phase 7+ grounding). L not triggered (proj_loss still declining: 0.0023→0.0007). | Items J, K, L, M all deferred. |
| **H: Magnitude pruning A/B** | Cold A/B inconclusive (regions only 1.2× baseline). Warm A/B (s51–53 from 50-session memory, regions at 2.7×): 2 thresholds failed — env cosine −0.065 (persistent, not transient); gist +0.069 (positive direction). Cascade: 4384→2192→1096→548 units pruned on consecutive cycles (no prune cooldown). warm_recall_l2 average PASS is timing artifact. **FAIL. ADR 0026 closed.** Pruning stays at --prune-fraction 0.0. |

---

## Campaign results summary

| Campaign | Sessions | output_mse (final) | p_lang (peak) | Notes |
|---|---|---|---|---|
| shorts-only | 30 | 0.498 | None (dead) | No ASR in short clips |
| mixed (shorts+H3) | s1–30 | 0.049 | 0.557 | Language alive; all regions growing |
| wiki-on-mixed | s31–40 | 0.028 | 0.564 | Text injection not additive for p_lang |
| lm-on-mixed | s41–50 | 0.026–0.053 | **0.910** | --lang-use-lm; shattered embed-table ceiling |

---

## Queued gaps (from 2026-06-17 architectural review)

Three concrete gaps identified in the architecture review. Ordered by severity:

| # | Gap | What to do | Priority |
|---|-----|------------|----------|
| **Q1** | Qwen LM on CPU during NREM | **Done + A/B confirmed (2026-06-18).** Exported `qwen25_text_hidden.onnx` — static `[1, 128, 896]`, no LM head — loaded via `migraphx_session`. `text_hidden_states()` uses GPU path when present; CPU fallback otherwise. A/B (5s×2c fresh memory, h3h3+shorts): proj_loss 0.0021 vs 0.0021 (+2.1%, PASS). | Medium |
| **R1** | Memory recall in generation | **Done (2026-06-18).** `generate()` now takes `recalled_episodes: &[Vec<f32>]`. Query handlers call `query_recalled_episodes()` to fetch top-k warm HNSW results (default k=3, `--recall-k` flag), filtered for self-match, reversed so closest-recalled is nearest to query tokens. Startup query gets empty recall. ADR 0028. | — |
| **R2** | Multi-turn conversation | **Done (2026-06-18).** `ConvTurn` ring buffer on `ActionDecoder`; `CONV_HISTORY_BUDGET=256` tokens; oldest-first eviction at push-time; EOS stripped from stored response_ids; cleared at sleep entry via `clear_conversation()`. Layout: `[recalled | current_episode | history | query]`. `tracing::info!` logs `history_turns`/`history_tokens`/`seq_len` per call. ADR 0029. Commit ae29a70. | — |
| **R3** | Introspection Phase 1 | **Done (2026-06-19).** Introspective queries (keyword-matched: "what did you hear", "recap", etc.) trigger injection of the 8 most recent `text_target` entries from the hot buffer, decoded and prepended as `[Recent perceived text: ...]` before calling `generate()`. `EpisodicBuffer::recent_text_targets`, `ActionDecoder::decode_ids`, `is_introspective_query`, `build_query_with_history`. Wake-local (hot buffer cleared at sleep). ADR 0030. Commit 615af1f. | — |
| **Q2** | Pruning disabled, regions grow monotonically | `--prune-fraction 0.0` after magnitude-based A/B failure. Grown units (>N_UNITS=1024) are clamped by `to_dense_n` — unreachable but accumulating. No pruning strategy yet works. Item I (knowledge distillation) is the correct next step once regions grow large enough to matter (trigger: any region > 2× baseline). | Low — not a current bottleneck |
| **Q3** | Top-down prior is weak in practice | Language region prior_cosine 0.843 at steady state. The McGurk-style disambiguation described in architecture.md is not yet realized. β=0.05 (via S4) is a soft uniform blend, not a reliable predictive-coding feedback loop. Would require much stronger per-region priors before disambiguation is useful. | Low — research frontier, no clear fix without better prior signal |

---

## Interaction capability queue (committed roadmap — 2026-06-17)

These three capabilities are required for the system to be useful as an interactive agent. They are not deferred; they are the next substantive development arc after the performance gaps. Ordered by dependency.

### R1 — Memory recall in generation

**What:** When a query arrives, retrieve top-k semantically relevant past episodes from the warm store and include their projections in the generation context, not just the current episode.

**Why this is the gap:** Currently generation only sees 5 region tokens from the *current* episode. Retrieved memories never reach the LLM. The recollection layer queries warm during wake (for reconstruction and gain), but that path is completely separate from the generation path.

**Mechanism:**
1. On stdin query, encode the query text through the language k-WTA to get a query episode vector.
2. Query HNSW top-k (k=3–5) from warm store using that vector.
3. For each recalled episode record, project its stored episode vector → 5 region tokens via the same `ActionDecoder` projection.
4. Prepend all recalled token sets before the query tokens: `[recall_0_tokens | recall_1_tokens | ... | query_tokens]`.
5. Autoregressive decode as before.

**Dependencies:** Projection must be trained well enough to produce coherent region tokens (proj_loss ≤ 0.001, or empirically coherent). R1 can be implemented now and enabled once projection converges.

**ADR needed:** yes — design choices around k, how to order recalled episodes, whether to include recency weighting.

---

### R2 — Multi-turn conversation

**What:** Maintain a sliding context window of recent (query, response) pairs across turns within a wake session. Prepend prior turns to each new generation call.

**Why this is the gap:** Each query is currently independent. The LLM has no memory of what was said one turn ago.

**Mechanism:**
1. A ring buffer of `(query_tokens, response_tokens)` pairs, bounded by token budget (e.g. 512 tokens of conversation history).
2. On each generation call: `[recall_tokens (R1) | conversation_history | current_query_tokens]`.
3. Buffer clears on sleep entry (wake-local by design, matching working memory).
4. Context budget split: ~256 tokens conversation history, ~128 tokens recall context, query gets the rest up to Qwen's context limit.

**Dependencies:** Simpler than R1; no new training required. Can be implemented before R1 if desired — just requires the conversation ring and token budget tracking.

**ADR needed:** yes — token budget allocation, truncation policy (oldest-first vs. summarise).

---

### R3 — Introspection: explain what it learned

**What:** Allow the system to answer questions about its own memory contents — what it has seen, what concepts it has consolidated, what it knows.

**Why this is the gap:** The system builds rich warm and cold memory but has no path to *describe* that memory in language. The semantic store contains episode vectors (sparse activations) with no associated labels that can be surfaced to the LLM.

**Mechanism (two-phase):**

*Phase 1 — recent history narration (simpler, implement first):*
- Maintain a text log of ASR transcripts and text-bypass lines per session (already available as `text_target` fields on EpisodeRecord).
- On an introspective query ("what did you just hear?", "what have we been talking about?"), retrieve the N most recent `text_target` strings from hot/warm, concatenate, and include as conversation context alongside the query.
- This gives the system access to a textual record of what it recently perceived.

*Phase 2 — semantic memory narration (harder, later):*
- During REM or on demand, sample diverse episode vectors from warm, decode each through the output heads to get reconstructed encoder inputs, then use the action decoder to generate a short description of each cluster.
- Requires output heads that reconstruct interpretable encoder inputs — currently they reconstruct raw embeddings (not text), so this needs the recollection-to-generation bridge plus good reconstruction quality.

**Dependencies:** Phase 1 depends on R2 (conversation context) and on `text_target` text being threaded into the generation prompt. Phase 2 depends on R1 and significantly better reconstruction quality.

**ADR needed:** yes — at Phase 1 implementation.

---

### M — MLP projection ✅ DONE (2026-06-21)

**ADR 0032.** Commit f9e5faa. proj_loss plateau at ~0.0007 across sessions 46–50 triggered M. 2-layer MLP (640→1024→5×896, ReLU) cold-started (architecture incompatible with old linear weights). Evaluate via `mean_proj_loss` over next 3–5 sessions. Persistent hallucination unchanged — that's L's job.

---

### L — LoRA fine-tuning of Qwen2.5 (after M stable)

**Source:** ADR 0012 deferred item. Fine-tune Qwen2.5-0.5B with LoRA so the LM adapts to the episode-vector prefix distribution.

**Trigger:** Generation quality is coherent but clearly limited by Qwen's pretrained prior not matching the system's learned episode representation. Concrete signal: perplexity of generated text is high for domain-specific content the system has ingested.

**ADR needed:** yes — LoRA rank, which layers, ORT compatibility or switch to a different inference path.

---

## Open items

### Tier 1 — trigger-gated (not yet met)

| # | Source | Item | Trigger |
|---|--------|------|---------|
| A | ADR 0006 | Cold archive O(1) promotion index | `promote()` called AND cold > 500k records |
| C | ADR 0006 | Per-region sparse activation storage | Replay analysis shows episode vectors insufficient |

### Tier 2 — conditional / quality-gated

| # | Source | Item | Trigger |
|---|--------|------|---------|
| E | ADR 0009 | Separate optimizer for output heads | Conflicting gradient directions observed |
| F | ADR 0010 | Post-k-WTA prior injection | Pre-k-WTA injection shows poor `prior_cosine` uptake |
| G | ADR 0010 | Learned per-region scalar gate | **Closed** — gist −0.061 (FAIL); S4 never fired in control (comparison confounded); ADR 0027 |
| H | ADR 0018 | Activation-frequency pruning (+ base magnitude pruning) | **Closed** — magnitude pruning failed warm A/B; activation-freq pruning has same root problem (mid-run encoder change); grown size (2.7×) not a bottleneck |
| I | ADR 0018 | Knowledge distillation compression | Grown regions exceed 2× baseline |

### Tier 3 — model swaps / Phase 7+

| # | Source | Item |
|---|--------|------|
| J | ADR 0003 | Whisper-tiny → Whisper-base / WavLM-base |
| K | ADR 0003 | DINOv2-S → SigLIP-B |
| L | ADR 0012 | LoRA fine-tuning of Qwen2.5 |
| M | ADR 0012 | MLP projection (640→5×896) |
| N | ADR 0014 | Visual/audio region grounding |

### Closed items (2026-06-17)

| # | Source | Item | Outcome |
|---|--------|------|---------|
| B | ADR 0019/0014 | Language last-token pooling | Done — LM hidden states 4× better; hardened |
| D | ADR 0005 | Full differentiable SAE | Closed — L1 coding term at λ=0.1 destabilised MSE by 27%; item answered |
| H | ADR 0018 | Magnitude/frequency pruning | **FAIL (warm A/B)** — env cosine −0.065 persistent; cascade (no cooldown); activation-freq pruning same root problem. --prune-fraction 0.0 stays. ADR 0026. |
| G | ADR 0010 | Per-region prior beta | **FAIL (cold A/B)** — gist −0.061; S4 never fired so comparison was 0.01 vs 0.05 (confounded); --prior-beta-regions flag ships but not in campaign. ADR 0027. |

---

## Key facts

- `--replay-batch-size 128` hardcoded in `train_campaign.sh`
- `--max-warm-records 200000` hardcoded in `train_campaign.sh` (fixes cycle-2 HNSW starvation)
- CAPACITY default is 8192; overrideable via env var
- `--working-memory-size 8` hardcoded
- `--action-distill-only` + `--lang-use-lm` both hardcoded in `train_campaign.sh`
- `--video-stride 2`, `--recall-ef 16`, `--wake-reconstruct-interval 5` hardcoded
- `PRIOR_BETA=0.01` campaign default
- Wake schedule trigger at 602s/cycle; N_UNITS=1024, K_ACTIVE=20, D_PROJ=128, EPISODE_DIM=640
- `qwen25_lm.onnx` runs on CPU only (dynamic seq_len blocks MIGraphX)
- `--sae-l1-coeff 0.0` default (off); leave off — A/B showed harm

---

## MIGraphX constraint (hard)

Never expose ScatterElements or INT64 outputs. Two-graph split, topk/masking on CPU.
`qwen25_lm.onnx` also blocked (dynamic seq_len) — would need fixed-shape re-export + attention_mask input.

---

## Mixed campaign memory location

`memory/mixed_campaign/` (shared across s1–50); best steady state: MSE ~0.028, p_lang ~0.85–0.91.

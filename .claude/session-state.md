# Session state — somnus

> **Last updated:** 2026-06-17
> **Branch:** `main`

## Status: All ADR backlog items deferred. Awaiting new direction.

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

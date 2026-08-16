# ADR 0033 — LoRA fine-tuning of Qwen2.5-0.5B on episode prefixes (ADR 0012 item L)

**Date:** 2026-06-21
**Status:** Validated and deployed (2026-06-25) for Qwen2.5, since superseded by the Qwen3 swap
(ADR 0036) and orphaned (gaps.md §39). `scripts/train_lora.py` (referenced throughout this ADR)
was removed 2026-08-16 as dead code — `qwen25_lm.onnx` and every other Qwen2.5 model file no
longer exist anywhere in the repo or `models/`, and its logic was already ported to
`scripts/train_lora_qwen3.py`, the live script gaps.md §39 tracks. This ADR's procedure below is
kept as the historical record of what was built and validated at the time, not as a runnable
reference.

## Context

The system generates coherent tokens but fully hallucinated output. Cause: Qwen2.5-0.5B was pretrained on natural text with no exposure to somnus's episode-vector prefix distribution — the 5 region tokens prepended to each generation call are trained (via proj_loss) to encode what was seen, but Qwen treats them as noise, never having learned to condition on them. R1/R2/R3 testing (2026-06-19) confirmed complete hallucination across all queries. The ADR 0032 MLP projection improves how faithfully episodes map to token space but can't fix this prior mismatch — Qwen must *learn* to read the prefix tokens.

## Decision

Fine-tune Qwen2.5-0.5B with LoRA adapters on (episode prefix, text target) pairs from the warm memory store. Merge and re-export as `qwen25_lm.onnx` — no runtime changes, the ONNX interface is unchanged.

**Training data:** export warm records with non-empty `text_target` as JSONL (`{"episode":[...640 f32...],"text":"..."}`). After 50 mixed_campaign sessions, warm holds ~100k-200k records (cap 200k); ~20-40% are text-bearing (observed ASR+text-bypass rate). New `--export-training-data <path>` reads `memory/warm/records.bin`, writes filtered JSONL (avoids a Python bincode parser).

**LoRA config:** base = Qwen2.5-0.5B HF checkpoint (must match the exported ONNX weights exactly); target modules `q_proj`/`v_proj` in all attention layers (standard minimal target); rank r=8 (~2M added params, sufficient for the adapter distribution shift); alpha=16 (effective scale=alpha/r=2, standard start); dropout=0.05 (light, small noisy dataset); library `peft`+`transformers`.

**Training procedure** (`scripts/train_lora.py`): load projection weights (`decoder_weights.bin` via Rust export or pre-projected JSONL); per (episode,text) pair — project episode→5 region tokens (Python reimplementation of the 2-layer MLP forward), tokenize text, build `[region_tokens_as_embeds | BOS | input_ids[:-1]]` with labels `[-100×(5+1) | input_ids]` (ignore prefix positions); forward through Qwen+LoRA, cross-entropy on label positions only. AdamW lr=2e-4, 3 epochs over filtered warm records. After: `model.merge_and_unload()` → export merged model to ONNX, replacing `models/qwen25_lm.onnx`.

**Projection weights export:** new `--export-proj-weights <path>` writes `proj_weights.npz` (`w1`[640,1024], `w2`[1024,4480]) so the Python script can replicate the prefix computation.

**What updates:** `qwen25_lm.onnx` (yes — generation path, the target). `qwen25_text_hidden.onnx` (no, first iteration — distillation-target drift self-corrects over sessions). `qwen25_gain.onnx` (no — gain modulation stability matters more than perfect alignment, re-evaluate after L validates). `qwen25_embed_table.safetensors` (no — tied to `lm_head`, Qwen2.5-0.5B ties weights so LoRA on attention doesn't change embeddings). Leaving text_hidden/gain frozen leaves proj_loss targets slightly stale post-LoRA, recalibrating over 1-2 sessions — acceptable at this stage.

**ONNX re-export:** same script/settings as the original `qwen25_lm.onnx` export, interface unchanged, no Rust changes needed.

**Validation:** A/B, 3 sessions/arm, fresh memory, same mixed_campaign clips. Control: pretrained `qwen25_lm.onnx`. Treatment: LoRA-merged. Primary metric: `mean_proj_loss` must not regress (LoRA changes what the LM attends to, but the projection target is unchanged). Qualitative: stdin responses should start reflecting ingested content rather than free-generating.

## Alternatives considered

- **Runtime LoRA adapters in Rust** (base frozen in ONNX, adapter matmuls on ORT outputs). Brittle — needs intercepting intermediate ORT tensors, layer indexing, careful ordering. Rejected — merge-and-re-export is simpler with zero runtime overhead.
- **Fine-tune all linear layers, no LoRA.** More expressive but ~134M params vs. ~2M — overfits a small noisy dataset, slow. Rejected.
- **Higher rank (r=16/32).** More capacity — start at r=8, increase if validation is inconclusive. Deferred.
- **Update text_hidden and gain.onnx simultaneously.** More principled but 3 GPU re-exports + calibration, added risk. Rejected for the first iteration — follow-up if gain drift appears post-LoRA.
- **Train on cold archive instead of warm.** More diverse/larger, but cold records lack record-level `text_target` linkage (stored separately); warm records are self-contained. Revisit if warm proves too small.

## Outcome (2026-06-25)

Training: 1544 pairs, 3 epochs on CPU (~20min), cross-entropy 6.37→5.62. Adapters saved to `memory/lora_output/`, merged and exported as `models/qwen25_lm.onnx` (1977MB); prior model backed up as `qwen25_lm_pretrained.onnx`.

A/B (3 sessions × 2 cycles, fresh memory, same clips):

| arm | avg mean_proj_loss |
|---|---|
| control (pretrained) | 0.002086 |
| treatment (LoRA) | 0.001966 |
| delta | **−5.8% (PASS)** |

4/6 cycles had valid paired data, all showed −5.5% to −6.2%. **PASS** (threshold <−5%). Item L closed — `qwen25_lm.onnx` is now the LoRA-fine-tuned model.

**ONNX export fix:** PyTorch 2.12's `torch.onnx.export` needs `dynamo=False` for the legacy TorchScript path, and must use `Qwen25LMWrapper` (wraps `model.model`+`model.lm_head`, returns both `last_hidden_state` and `logits`) rather than `AutoModelForCausalLM` directly — now fixed in `scripts/train_lora.py`.

## Consequences

New Python deps `peft`/`transformers`/`torch` (training-only, not a Rust runtime dep — already available in the ONNX-export venv). One-time ~20-60min training run. `qwen25_lm.onnx` replaced, prior backed up. `--export-training-data`/`--export-proj-weights` are offline utilities, not part of the training campaign. If LoRA validation fails (proj_loss regression or no qualitative improvement), restore `qwen25_lm_pretrained.onnx` and close L as deferred pending better projection quality.

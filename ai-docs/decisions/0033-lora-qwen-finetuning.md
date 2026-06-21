# ADR 0033 — LoRA fine-tuning of Qwen2.5-0.5B on episode prefixes (ADR 0012 item L)

**Date:** 2026-06-21
**Status:** Accepted

## Context

The system generates coherent tokens but the output is fully hallucinated. The cause is clear: Qwen2.5-0.5B was pretrained on natural text with no exposure to our episode-vector prefix distribution. The 5 region tokens prepended to each generation call are trained (by proj_loss) to encode what was seen, but Qwen treats them as noise because it has never learned to condition on them. R1/R2/R3 testing (session 2026-06-19) confirmed complete hallucination across all queries.

The projection (ADR 0032 MLP, now validated) improves how faithfully episodes map to the LM's token space, but cannot fix the fundamental prior mismatch: Qwen must *learn* to read those prefix tokens.

## Decision

Fine-tune Qwen2.5-0.5B with LoRA adapters on (episode prefix, text target) pairs sourced from the warm memory store. The fine-tuned weights are merged and re-exported as a replacement `qwen25_lm.onnx`. No runtime changes required: the ONNX interface is unchanged.

### Training data

Export warm store episode records to a JSONL file:
```
{"episode": [...640 f32...], "text": "the text target string"}
```

Source: all warm records where `text_target` is non-empty. After 50 mixed_campaign sessions the warm store holds ~100k–200k records (cap: 200k). Filter to text-bearing records (~20–40% of total based on observed ASR + text-bypass rates).

Add `--export-training-data <path>` to the somnus binary: reads `memory/warm/records.bin`, writes filtered records as JSONL. This avoids a Python bincode parser.

### LoRA configuration

| Parameter | Value | Rationale |
|---|---|---|
| Base model | Qwen2.5-0.5B (HF checkpoint, same as ONNX source) | Must match exported weights exactly |
| Target modules | `q_proj`, `v_proj` in all attention layers | Standard minimal target; keys for attention re-weighting |
| Rank r | 8 | ~2M additional params; sufficient for adapter distribution shift |
| Alpha | 16 | Effective scale = alpha/r = 2; standard starting point |
| Dropout | 0.05 | Light regularization; dataset is small and noisy |
| Library | `peft` + `transformers` | Mature, well-tested LoRA implementation |

### Training procedure (Python script `scripts/train_lora.py`)

1. Load projection weights from `memory/snapshot/decoder_weights.bin` via Rust export (see below) or read directly from the JSONL if pre-projected.
2. For each (episode, text) pair:
   - Project episode → 5 region tokens via current `proj_weights_flat()` (Python reimplementation of the 2-layer MLP forward, same as `gain_from_episode`).
   - Tokenize `text` → `input_ids`.
   - Build input sequence: `[region_tokens_as_embeds | BOS | input_ids[:-1]]`.
   - Labels: `[-100 × (5 + 1) | input_ids]` (ignore prefix positions in loss).
   - Forward through Qwen with LoRA; compute cross-entropy on label positions only.
3. Optimizer: AdamW, lr=2e-4, 3 epochs over the filtered warm records.
4. After training: `model.merge_and_unload()` → export merged model to ONNX replacing `models/qwen25_lm.onnx`.

### Projection weights export

The Python training script needs the current MLP weights (W1, W2) to replicate the prefix computation. Add `--export-proj-weights <path>` to the somnus binary: writes `proj_weights.npz` with keys `w1` (shape [640, 1024]) and `w2` (shape [1024, 4480]).

### What is and is not updated

| Model file | Updated | Reason |
|---|---|---|
| `qwen25_lm.onnx` | Yes | Generation path — this is the target |
| `qwen25_text_hidden.onnx` | No (first iteration) | Distillation targets drift is acceptable; self-corrects over training sessions |
| `qwen25_gain.onnx` | No | Gain modulation stability more important than perfect alignment; re-evaluate after L validates |
| `qwen25_embed_table.safetensors` | No | Embedding table is tied to `lm_head`; Qwen2.5-0.5B ties weights, so LoRA on attention layers doesn't change embeddings |

Leaving text_hidden and gain frozen means proj_loss training targets are slightly stale post-LoRA, but the projection will recalibrate over 1–2 sessions. This is acceptable at this stage.

### ONNX re-export

After merging adapters, re-export with `torch.onnx.export` using the same script/settings that produced the original `qwen25_lm.onnx`. The interface (`inputs_embeds`, `last_hidden_state`/logits) is unchanged; the Rust code requires no modifications.

### Validation

A/B: run 3 sessions each arm from fresh memory using the same mixed_campaign clips.
- Control: existing `qwen25_lm.onnx` (pretrained only).
- Treatment: LoRA-merged `qwen25_lm.onnx`.
- Primary metric: `mean_proj_loss` must not regress (LoRA changes what the LM pays attention to, but the projection training target is unchanged).
- Qualitative: stdin query responses should begin to reflect ingested content rather than free-generate.

## Alternatives considered

**Runtime LoRA adapters in Rust:** Keep base model frozen in ONNX, apply adapter matrices as extra matmuls on ORT outputs in Rust. Brittle: requires intercepting intermediate tensors from ORT, layer indexing, and careful ordering. Rejected: merge-and-re-export is simpler and has zero runtime overhead.

**Fine-tune all linear layers (no LoRA):** More expressive. Rejected: ~134M params to update vs ~2M with LoRA; overfits badly on a small noisy dataset; slow to train.

**Higher rank (r=16 or r=32):** More capacity. Start at r=8; if validation is inconclusive, increase. Deferred.

**Update text_hidden and gain.onnx simultaneously:** Principled consistency. Rejected for first iteration: three GPU re-exports plus calibration; added risk surface. Do it as a follow-up if gain drift is observed after LoRA.

**Train on cold archive records instead of warm:** More diverse data, larger set. Rejected for now: cold records lack `text_target` linkage at the record level (stored separately); warm records are self-contained. Revisit if warm set proves too small.

## Consequences

- New Python dependencies: `peft`, `transformers`, `torch` (for training only; not a Rust runtime dep). Already available in the Python environment used for ONNX export.
- One-time training run of ~20–60 min depending on dataset size and GPU.
- `qwen25_lm.onnx` is replaced; prior file should be backed up as `qwen25_lm_pretrained.onnx`.
- `--export-training-data` and `--export-proj-weights` flags added to somnus binary; these are offline utilities, not part of the training campaign.
- If LoRA validation fails (proj_loss regression or no qualitative improvement), restore `qwen25_lm_pretrained.onnx` and close L as deferred pending better projection quality.

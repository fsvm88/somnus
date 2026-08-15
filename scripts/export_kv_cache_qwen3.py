"""
Export KV-cache ONNX models for Qwen3-0.6B (ADR 0034).

  qwen3_lm_prefill_kv.onnx  — dynamic seq_len, outputs logits + 28×2 KV tensors
  qwen3_lm_decode_kv.onnx   — static KV_SIZE=2048, inputs include past KV, outputs
                               logits + 28×2 new single-step KV tensors

Loads base Qwen3-0.6B (or LoRA-merged from memory/lora_output_qwen3/).

Usage:
  .venv/bin/python scripts/export_kv_cache_qwen3.py [--kv-size 2048] [--models-dir models/]
"""
import argparse, math, sys, time
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv
from pathlib import Path

N_LAYERS   = 28
N_KV_HEADS = 8
N_HEADS    = 16
# gaps.md §38: Qwen3's head_dim is decoupled from hidden_size/num_attention_heads (would be
# 1024/16=64) — the actual config value is 128 (confirmed via AutoConfig.from_pretrained).
# q_proj/k_proj/v_proj project to N_HEADS*HEAD_DIM / N_KV_HEADS*HEAD_DIM, not hidden_size.
# This was the real "torch.export shape inference" blocker: q_proj(hidden).view(..., 64) fails
# because q_proj's actual output is sized for head_dim=128, not a torch.export quirk per se.
HEAD_DIM   = 128
LM_HIDDEN  = 1024


# ── Wrappers ─────────────────────────────────────────────────────────────────

class PrefillKVWrapper(nn.Module):
    """Full prefill: inputs_embeds [1,n,1024] → logits [1,n,vocab] + 28 key + 28 val tensors."""
    def __init__(self, m):
        super().__init__()
        self.base    = m.model
        self.lm_head = m.lm_head

    def forward(self, inputs_embeds):
        out    = self.base(inputs_embeds=inputs_embeds, use_cache=True)
        logits = self.lm_head(out.last_hidden_state)
        cache  = out.past_key_values
        keys   = [layer.keys   for layer in cache.layers]
        vals   = [layer.values for layer in cache.layers]
        return (logits, *keys, *vals)


class DecodeKVWrapper(nn.Module):
    """Single-token decode with static KV buffer. Returns logits + 28 new_key + 28 new_val."""
    def __init__(self, m):
        super().__init__()
        self.layers     = m.model.layers
        self.norm       = m.model.norm
        self.lm_head    = m.lm_head
        self.rotary_emb = m.model.rotary_emb

    def forward(self, inputs_embeds, position_ids, attention_mask, *past_kvs):
        # inputs_embeds:  [1, 1, LM_HIDDEN]
        # position_ids:   [1, 1]
        # attention_mask: [1, 1, 1, KV_SIZE+1]
        # past_kvs:       key_0, val_0, ..., key_27, val_27  each [1, N_KV_HEADS, KV_SIZE, HEAD_DIM]
        hidden = inputs_embeds
        bsz    = 1
        new_keys, new_vals = [], []

        for i, layer in enumerate(self.layers):
            residual = hidden
            hidden   = layer.input_layernorm(hidden)
            attn     = layer.self_attn

            # Qwen3-specific: per-head QK-RMSNorm (q_norm/k_norm), applied on the head_dim
            # axis *before* the transpose to [bsz, heads, seq, head_dim] — Qwen2.5 has no
            # equivalent (transformers.models.qwen3.modeling_qwen3.Qwen3Attention.forward).
            # Omitting this wouldn't just crash, it would export a numerically wrong model.
            q     = attn.q_norm(attn.q_proj(hidden).view(bsz, 1, N_HEADS,    HEAD_DIM)).transpose(1, 2)
            k_new = attn.k_norm(attn.k_proj(hidden).view(bsz, 1, N_KV_HEADS, HEAD_DIM)).transpose(1, 2)
            v_new = attn.v_proj(hidden).view(bsz, 1, N_KV_HEADS, HEAD_DIM).transpose(1, 2)

            cos, sin = self.rotary_emb(q, position_ids)
            q, k_new = apply_rotary_pos_emb(q, k_new, cos, sin)

            new_keys.append(k_new)
            new_vals.append(v_new)

            k_full = repeat_kv(torch.cat([past_kvs[2*i],   k_new], dim=2), N_HEADS // N_KV_HEADS)
            v_full = repeat_kv(torch.cat([past_kvs[2*i+1], v_new], dim=2), N_HEADS // N_KV_HEADS)

            # Manual matmul+softmax instead of F.scaled_dot_product_attention (gaps.md §38):
            # SDPA is the one structurally new element vs. the Qwen2.5 decode wrapper that
            # exports cleanly (export_kv_cache.py) — its fused-op export path has historically
            # had static-shape/mask-broadcasting quirks. Matches export_kv_cache.py exactly.
            attn_w   = torch.matmul(q, k_full.transpose(2, 3)) * attn.scaling + attention_mask
            attn_w   = F.softmax(attn_w, dim=-1, dtype=torch.float32).to(q.dtype)
            # o_proj's input dim is N_HEADS*HEAD_DIM (2048), not LM_HIDDEN (1024) — they only
            # coincide for Qwen2.5, where head_dim == hidden_size/num_heads.
            attn_out = torch.matmul(attn_w, v_full).transpose(1, 2).reshape(bsz, 1, N_HEADS * HEAD_DIM)
            hidden   = residual + attn.o_proj(attn_out)

            ff_residual = hidden
            hidden      = layer.post_attention_layernorm(hidden)
            # Delegate to the real MLP module instead of reimplementing SwiGLU by hand
            # (export_kv_cache.py does the same for Qwen2.5) — a hand-rolled version here
            # previously had the gate/up order backwards: silu(gate_proj(x)) * up_proj(x)
            # is correct, not gate_proj(x) * silu(up_proj(x)) (Qwen3MLP.forward).
            hidden      = layer.mlp(hidden)
            hidden      = ff_residual + hidden

        hidden = self.norm(hidden)
        logits = self.lm_head(hidden)
        return (logits, *new_keys, *new_vals)


# ── Numerical equivalence check (ported from export_kv_cache.py — gaps.md §38's stub only
#    loaded the prefill ONNX and checked ORT could load it, no reference comparison) ─────────

def verify_equivalence(merged, kv_size, n_verify=20, prefix_len=320):
    print(f"\n=== Equivalence check (prefix={prefix_len}, steps={n_verify}, kv_size={kv_size}) ===")
    torch.manual_seed(42)
    prefix_embeds = torch.randn(1, prefix_len, LM_HIDDEN)

    with torch.no_grad():
        all_e = prefix_embeds.clone()
        ref_first_logits = merged(inputs_embeds=all_e).logits[0, -1].clone()
        ref_tokens = []
        for _ in range(n_verify):
            out = merged(inputs_embeds=all_e)
            tok = out.logits[0, -1].argmax().item()
            ref_tokens.append(tok)
            next_e = merged.model.embed_tokens(torch.tensor([[tok]]))
            all_e  = torch.cat([all_e, next_e], dim=1)

    pfill = PrefillKVWrapper(merged).eval()
    dec   = DecodeKVWrapper(merged).eval()
    kv_bufs = [torch.zeros(1, N_KV_HEADS, kv_size, HEAD_DIM) for _ in range(N_LAYERS * 2)]
    mask    = torch.full((1, 1, 1, kv_size + 1), -1e4)
    pos     = prefix_len

    with torch.no_grad():
        pouts = pfill(prefix_embeds)
        for i in range(N_LAYERS):
            kv_bufs[2*i  ][:, :, :prefix_len, :] = pouts[1+i]
            kv_bufs[2*i+1][:, :, :prefix_len, :] = pouts[1+N_LAYERS+i]
        mask[:, :, :, :prefix_len] = 0.0
        mask[:, :, :, kv_size]     = 0.0

        diff = (pouts[0][0, -1] - ref_first_logits).abs().max().item()
        print(f"Prefill logit max_abs_diff: {diff:.2e}  argmax_match: {int(pouts[0][0,-1].argmax()==ref_first_logits.argmax())}")

        kv_toks, next_id = [], ref_tokens[0]
        for _ in range(n_verify):
            emb = merged.model.embed_tokens(torch.tensor([[next_id]]))
            pid = torch.tensor([[pos]], dtype=torch.long)
            douts = dec(emb, pid, mask, *kv_bufs)
            for i in range(N_LAYERS):
                kv_bufs[2*i  ][:, :, pos, :] = douts[1+i][:, :, 0, :]
                kv_bufs[2*i+1][:, :, pos, :] = douts[1+N_LAYERS+i][:, :, 0, :]
            mask[:, :, :, pos] = 0.0
            pos += 1
            next_id = douts[0][0, 0].argmax().item()
            kv_toks.append(next_id)

    # gaps.md §38 finding: kv_toks[i] is the prediction for position (prefix_len+1+i), but
    # ref_tokens[i] is the prediction for position (prefix_len+i) — ref_tokens[0] is what
    # seeds the first decode call, so its counterpart in the KV chain doesn't exist yet.
    # Comparing zip(ref_tokens, kv_toks) index-for-index (as this was originally ported)
    # is off by one; the correct alignment is kv_toks[i] vs. ref_tokens[i+1].
    n_pairs = len(kv_toks) - 1
    match = sum(a == b for a, b in zip(ref_tokens[1:], kv_toks[:-1]))
    print(
        f"Token agreement: {match}/{n_pairs}  "
        f"(ref[1:6]={ref_tokens[1:6]}  kv[:5]={kv_toks[:5]})"
    )
    if match < round(n_pairs * 0.9):
        raise RuntimeError(f"Equivalence check FAILED: only {match}/{n_pairs} tokens match")
    print("PASS")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kv-size', type=int, default=2048, help='KV buffer size')
    parser.add_argument('--models-dir', type=Path, default=Path('models'), help='Output models directory')
    parser.add_argument('--skip-verify', action='store_true', help='Skip numerical verification')
    parser.add_argument('--skip-export', action='store_true', help='Skip export (verify only)')
    parser.add_argument('--bench-only', action='store_true', help='Benchmark only')
    args = parser.parse_args()

    models_dir = args.models_dir
    models_dir.mkdir(exist_ok=True)
    kv_size = args.kv_size

    print(f"Qwen3-0.6B KV-cache export (KV_SIZE={kv_size})")
    print(f"  Prefill:  qwen3_lm_prefill_kv.onnx")
    print(f"  Decode:   qwen3_lm_decode_kv.onnx")
    print()

    model = None
    if args.bench_only:
        print("Benchmarking only (no export)")
    else:
        # Load base or LoRA-merged model
        lora_dir = Path("memory/lora_output_qwen3")
        if lora_dir.exists():
            print(f"  Loading LoRA-merged from {lora_dir}")
            from peft import PeftModel
            base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", torch_dtype=torch.float32)
            model = PeftModel.from_pretrained(base, str(lora_dir))
            model = model.merge_and_unload()
        else:
            print(f"  Loading base Qwen3-0.6B")
            model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", torch_dtype=torch.float32)

        model.eval()

        # ── Prefill export ────────────────────────────────────────────────────
        prefill_out = models_dir / "qwen3_lm_prefill_kv.onnx"
        if not prefill_out.exists():
            print(f"Exporting prefill to {prefill_out.name}...")
            prefill_wrapper = PrefillKVWrapper(model).eval()
            dummy_prefill = torch.zeros(1, 4, LM_HIDDEN)
            t0 = time.time()
            torch.onnx.export(
                prefill_wrapper,
                dummy_prefill,
                str(prefill_out),
                input_names=["inputs_embeds"],
                output_names=["logits"] + [f"k_{i}" for i in range(N_LAYERS)] + [f"v_{i}" for i in range(N_LAYERS)],
                dynamic_axes={"inputs_embeds": {0: "batch", 1: "seq_len"}},
                opset_version=17,
                dynamo=False,
            )
            elapsed = time.time() - t0
            size_mb = prefill_out.stat().st_size / 1e6
            print(f"  {size_mb:.1f} MB, {elapsed:.1f}s")

        # ── Decode export ─────────────────────────────────────────────────────
        decode_out = models_dir / "qwen3_lm_decode_kv.onnx"
        if not decode_out.exists():
            print(f"Exporting decode to {decode_out.name}...")
            decode_wrapper = DecodeKVWrapper(model).eval()
            dummy_emb  = torch.zeros(1, 1, LM_HIDDEN)
            dummy_pos  = torch.zeros(1, 1, dtype=torch.long)
            dummy_mask = torch.zeros(1, 1, 1, kv_size + 1)
            dummy_kvs  = [torch.zeros(1, N_KV_HEADS, kv_size, HEAD_DIM) for _ in range(2 * N_LAYERS)]

            t0 = time.time()
            torch.onnx.export(
                decode_wrapper,
                (dummy_emb, dummy_pos, dummy_mask, *dummy_kvs),
                str(decode_out),
                input_names=["inputs_embeds", "position_ids", "attention_mask"] + [f"kv_{i}" for i in range(2*N_LAYERS)],
                output_names=["logits"] + [f"k_{i}" for i in range(N_LAYERS)] + [f"v_{i}" for i in range(N_LAYERS)],
                opset_version=17,
                dynamo=False,  # Avoid torch.export shape inference issues with static KV shapes
            )
            elapsed = time.time() - t0
            print(f"  exported in {elapsed:.1f}s, consolidating external data...")

            # The legacy TorchScript exporter scatters external weight data across one
            # loose file per initializer (onnx__MatMul_NNNN) directly in models_dir once
            # the graph exceeds its inline-protobuf threshold, instead of the single
            # `<name>.onnx.data` companion file prefill_kv.onnx already has — reload and
            # re-save consolidated so decode_kv matches that convention and models_dir
            # doesn't end up with ~200 stray same-looking filenames across regenerations.
            import onnx as _onnx
            proto = _onnx.load(str(decode_out), load_external_data=True)
            stray = [p for p in models_dir.glob("onnx__MatMul_*") if p.is_file()]
            _onnx.save_model(
                proto,
                str(decode_out),
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location=f"{decode_out.name}.data",
                size_threshold=1024,
            )
            for p in stray:
                p.unlink(missing_ok=True)

            size_mb = decode_out.stat().st_size / 1e6
            data_mb = (decode_out.parent / f"{decode_out.name}.data").stat().st_size / 1e6
            print(f"  {size_mb:.1f} MB + {data_mb:.1f} MB external data")

        print("\nExport complete.")

    # ── Verification ──────────────────────────────────────────────────────────
    if not args.skip_verify:
        if model is None:
            print("\n--skip-verify not set but no model loaded (--bench-only) — skipping")
        else:
            verify_equivalence(model, kv_size)


if __name__ == "__main__":
    main()

"""
Export two ONNX models for KV-cache generation (ADR 0034):

  qwen25_lm_prefill_kv.onnx  — dynamic seq_len, outputs logits + 24×2 KV tensors
  qwen25_lm_decode_kv.onnx   — static KV_SIZE=2048, inputs include past KV, outputs
                                logits + 24×2 new single-step KV tensors

Loads the LoRA-merged model from memory/lora_output/ on top of models/qwen25_hf/.

Usage:
  .venv/bin/python scripts/export_kv_cache.py [--kv-size 2048] [--models-dir models/]
"""
import argparse, math, sys, time
import torch, torch.nn as nn, torch.nn.functional as F
import onnxruntime as ort
import numpy as np
from peft import PeftModel
from transformers import AutoModelForCausalLM
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv

N_LAYERS   = 24
N_KV_HEADS = 2
N_HEADS    = 14
HEAD_DIM   = 64
LM_HIDDEN  = 896


# ── Wrappers ─────────────────────────────────────────────────────────────────

class PrefillKVWrapper(nn.Module):
    """Full prefill: inputs_embeds [1,n,896] → logits [1,n,vocab] + 24 key + 24 val tensors."""
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
    """Single-token decode with static KV buffer. Returns logits + 24 new_key + 24 new_val."""
    def __init__(self, m):
        super().__init__()
        self.layers     = m.model.layers
        self.norm       = m.model.norm
        self.lm_head    = m.lm_head
        self.rotary_emb = m.model.rotary_emb

    def forward(self, inputs_embeds, position_ids, attention_mask, *past_kvs):
        # inputs_embeds:  [1, 1, LM_HIDDEN]
        # position_ids:   [1, 1]
        # attention_mask: [1, 1, 1, KV_SIZE+1]  (−inf = masked, 0 = attend)
        # past_kvs:       key_0, val_0, ..., key_23, val_23  each [1, N_KV_HEADS, KV_SIZE, HEAD_DIM]
        hidden = inputs_embeds
        bsz    = 1
        new_keys, new_vals = [], []

        for i, layer in enumerate(self.layers):
            residual = hidden
            hidden   = layer.input_layernorm(hidden)
            attn     = layer.self_attn

            q     = attn.q_proj(hidden).view(bsz, 1, N_HEADS,    HEAD_DIM).transpose(1, 2)
            k_new = attn.k_proj(hidden).view(bsz, 1, N_KV_HEADS, HEAD_DIM).transpose(1, 2)
            v_new = attn.v_proj(hidden).view(bsz, 1, N_KV_HEADS, HEAD_DIM).transpose(1, 2)

            cos, sin = self.rotary_emb(q, position_ids)
            q, k_new = apply_rotary_pos_emb(q, k_new, cos, sin)

            new_keys.append(k_new)
            new_vals.append(v_new)

            k_full = repeat_kv(torch.cat([past_kvs[2*i],   k_new], dim=2), N_HEADS // N_KV_HEADS)
            v_full = repeat_kv(torch.cat([past_kvs[2*i+1], v_new], dim=2), N_HEADS // N_KV_HEADS)

            attn_w = torch.matmul(q, k_full.transpose(2, 3)) * attn.scaling + attention_mask
            attn_w = F.softmax(attn_w, dim=-1, dtype=torch.float32).to(q.dtype)
            attn_out = torch.matmul(attn_w, v_full).transpose(1, 2).reshape(bsz, 1, N_HEADS * HEAD_DIM)
            hidden   = attn.o_proj(attn_out) + residual

            residual = hidden
            hidden   = layer.post_attention_layernorm(hidden)
            hidden   = layer.mlp(hidden) + residual

        logits = self.lm_head(self.norm(hidden))
        return (logits, *new_keys, *new_vals)


# ── Numerical equivalence check ───────────────────────────────────────────────

def verify_equivalence(merged, kv_size, n_verify=20, prefix_len=320):
    print(f"\n=== Equivalence check (prefix={prefix_len}, steps={n_verify}, kv_size={kv_size}) ===")
    torch.manual_seed(42)
    prefix_embeds = torch.randn(1, prefix_len, LM_HIDDEN)

    # Reference: growing sequence, no KV cache
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

    # KV-cache path
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

    match = sum(a == b for a, b in zip(ref_tokens, kv_toks))
    print(f"Token agreement: {match}/{n_verify}  (ref[:5]={ref_tokens[:5]}  kv[:5]={kv_toks[:5]})")
    if match < 18:
        raise RuntimeError(f"Equivalence check FAILED: only {match}/{n_verify} tokens match")
    print("PASS")


# ── ONNX export ───────────────────────────────────────────────────────────────

def export_prefill(merged, out_path):
    print(f"\n=== Exporting prefill model → {out_path} ===")
    wrapper = PrefillKVWrapper(merged).eval()
    dummy   = torch.zeros(1, 4, LM_HIDDEN)

    key_names = [f"past_key_{i}"   for i in range(N_LAYERS)]
    val_names = [f"past_value_{i}" for i in range(N_LAYERS)]

    with torch.no_grad():
        torch.onnx.export(
            wrapper, dummy, out_path,
            input_names=["inputs_embeds"],
            output_names=["logits"] + key_names + val_names,
            dynamic_axes={
                "inputs_embeds": {0: "batch", 1: "seq_len"},
                "logits":        {0: "batch", 1: "seq_len"},
                **{n: {0: "batch", 2: "seq_len"} for n in key_names + val_names},
            },
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
    print(f"  Written ({out_path})")


def export_decode(merged, kv_size, out_path):
    print(f"\n=== Exporting decode model (kv_size={kv_size}) → {out_path} ===")
    wrapper = DecodeKVWrapper(merged).eval()

    kv_bufs = [torch.zeros(1, N_KV_HEADS, kv_size, HEAD_DIM) for _ in range(N_LAYERS * 2)]
    mask    = torch.full((1, 1, 1, kv_size + 1), -1e4)
    emb     = torch.zeros(1, 1, LM_HIDDEN)
    pos_ids = torch.tensor([[0]], dtype=torch.long)

    key_in_names  = [f"past_key_{i}"   for i in range(N_LAYERS)]
    val_in_names  = [f"past_value_{i}" for i in range(N_LAYERS)]
    key_out_names = [f"new_key_{i}"    for i in range(N_LAYERS)]
    val_out_names = [f"new_value_{i}"  for i in range(N_LAYERS)]

    inputs = (emb, pos_ids, mask, *kv_bufs)

    with torch.no_grad():
        torch.onnx.export(
            wrapper, inputs, out_path,
            input_names=["inputs_embeds", "position_ids", "attention_mask"]
                        + key_in_names + val_in_names,
            output_names=["logits"] + key_out_names + val_out_names,
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
    print(f"  Written ({out_path})")


# ── ORT performance check ─────────────────────────────────────────────────────

def bench_ort_decode(decode_path, kv_size, n_steps=50, warmup=5):
    print(f"\n=== ORT decode latency (kv_size={kv_size}, CPU) ===")
    sess = ort.InferenceSession(decode_path, providers=["CPUExecutionProvider"])

    kv_bufs = {f"past_key_{i}":   np.zeros((1, N_KV_HEADS, kv_size, HEAD_DIM), np.float32)
               for i in range(N_LAYERS)}
    kv_bufs.update({f"past_value_{i}": np.zeros((1, N_KV_HEADS, kv_size, HEAD_DIM), np.float32)
                    for i in range(N_LAYERS)})
    mask    = np.full((1, 1, 1, kv_size + 1), -1e4, dtype=np.float32)
    mask[:, :, :, :320] = 0.0
    mask[:, :, :, kv_size] = 0.0
    emb     = np.zeros((1, 1, LM_HIDDEN), np.float32)
    pos_ids = np.array([[320]], dtype=np.int64)

    feed = {"inputs_embeds": emb, "position_ids": pos_ids, "attention_mask": mask, **kv_bufs}

    # Warmup
    for _ in range(warmup):
        sess.run(None, feed)

    t0 = time.perf_counter()
    for _ in range(n_steps):
        sess.run(None, feed)
    lat = (time.perf_counter() - t0) / n_steps
    print(f"  {lat*1000:.1f} ms/step  → 128 tokens = {lat*128:.1f}s")
    return lat


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kv-size",    type=int, default=2048)
    ap.add_argument("--models-dir", type=str, default="models")
    ap.add_argument("--skip-verify",  action="store_true")
    ap.add_argument("--skip-export",  action="store_true")
    ap.add_argument("--bench-only",   action="store_true")
    args = ap.parse_args()

    import os
    prefill_path = os.path.join(args.models_dir, "qwen25_lm_prefill_kv.onnx")
    decode_path  = os.path.join(args.models_dir, f"qwen25_lm_decode_kv.onnx")

    if args.bench_only:
        bench_ort_decode(decode_path, args.kv_size)
        return

    print("=== Loading LoRA-merged model ===")
    base   = AutoModelForCausalLM.from_pretrained("models/qwen25_hf", dtype=torch.float32,
                                                   attn_implementation="eager", trust_remote_code=True)
    merged = PeftModel.from_pretrained(base, "memory/lora_output").merge_and_unload()
    merged.eval()
    print("LoRA merged.")

    if not args.skip_verify:
        verify_equivalence(merged, args.kv_size)

    if not args.skip_export:
        export_prefill(merged, prefill_path)
        export_decode(merged, args.kv_size, decode_path)

    bench_ort_decode(decode_path, args.kv_size)
    print("\nDone.")


if __name__ == "__main__":
    main()

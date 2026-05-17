#!/usr/bin/env python3
"""Export Whisper-tiny encoder, CLAP audio encoder, and DINOv2-S/14 to ONNX.

Run from the repo root:
    scripts/.venv/bin/python scripts/export_models.py

Exports to models/ (gitignored). Idempotent: re-running skips existing files.
Pinned HF revisions from ADR 0003.

Input/output shapes (batch=1, FP32):
  whisper_tiny_encoder.onnx
    in:  input_features   (1,  80, 3000) — mel spectrogram, 30 s @ 16 kHz, 80 bins
    out: last_hidden_state (1, 1500, 384) — Rust mean-pools to (1, 384)

  clap_audio_encoder.onnx
    in:  input_features  (1, 1, 1001, 64) — 10 s mel spectrogram @ 48 kHz, 64 bins
    out: audio_embeds    (1, 512)         — L2-normalised projection embedding

  dinov2_small.onnx
    in:  pixel_values    (1, 3, 224, 224) — ImageNet-normalised RGB
    out: last_hidden_state (1, 257, 384)  — Rust mean-pools patch tokens [1:] to (1, 384)
"""

import hashlib
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"

REVISIONS = {
    "whisper": ("openai/whisper-tiny", "169d4a4341b33bc18d8881c4b69c2e104e1cc0af"),
    "clap":    ("laion/clap-htsat-unfused", "8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a"),
    "dinov2":  ("facebook/dinov2-small",    "ed25f3a31f01632728cabb09d1542f84ab7b0056"),
    # Qwen2.5-0.5B: revision should be pinned after first export; None = current main.
    "qwen25":  ("Qwen/Qwen2.5-0.5B", "060db6499f32faf8b98477b0a26969ef7d8b9987"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def report(out: Path) -> None:
    """Print size and sha256 for an ONNX export (graph + optional external data)."""
    data = Path(str(out) + ".data")
    if data.exists():
        total = out.stat().st_size + data.stat().st_size
        print(f"  graph {out.stat().st_size / 1e6:.1f} MB  data {data.stat().st_size / 1e6:.0f} MB"
              f"  total {total / 1e6:.0f} MB")
        print(f"  sha256(graph)={sha256(out)}")
        print(f"  sha256(data) ={sha256(data)}")
    else:
        print(f"  {out.stat().st_size / 1e6:.1f} MB  sha256={sha256(out)}")


# ── wrappers (force tuple output so torch.onnx.export works cleanly) ──────────

class WhisperEncoderWrapper(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.enc = encoder

    def forward(self, input_features):
        return self.enc(input_features, return_dict=False)[0]


class ClapAudioWrapper(torch.nn.Module):
    """HTSAT encoder + ClapProjectionLayer → 512-dim L2-normalised embedding."""
    def __init__(self, audio_model, audio_projection):
        super().__init__()
        self.audio_model = audio_model
        self.audio_projection = audio_projection

    def forward(self, input_features):
        pooler = self.audio_model(input_features=input_features, return_dict=False)[1]
        return self.audio_projection(pooler)


class Dinov2Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        return self.model(pixel_values, return_dict=False)[0]


# ── per-model exports ──────────────────────────────────────────────────────────

def export_whisper_encoder() -> None:
    out = MODELS_DIR / "whisper_tiny_encoder.onnx"
    if out.exists():
        print(f"  skip: {out.name} already present")
        return

    from transformers import WhisperModel
    repo, rev = REVISIONS["whisper"]
    print(f"  loading {repo} @ {rev[:12]}…")
    model = WhisperModel.from_pretrained(repo, revision=rev)
    wrapper = WhisperEncoderWrapper(model.encoder).eval()

    dummy = torch.zeros(1, 80, 3000)
    print(f"  exporting…")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy,
            str(out),
            input_names=["input_features"],
            output_names=["last_hidden_state"],
            dynamic_axes={"input_features": {0: "batch"}, "last_hidden_state": {0: "batch"}},
            opset_version=17,
        )
    report(out)


def export_clap_encoder() -> None:
    out = MODELS_DIR / "clap_audio_encoder.onnx"
    if out.exists():
        print(f"  skip: {out.name} already present")
        return

    from transformers import ClapModel
    repo, rev = REVISIONS["clap"]
    print(f"  loading {repo} @ {rev[:12]}…")
    model = ClapModel.from_pretrained(repo, revision=rev)
    wrapper = ClapAudioWrapper(model.audio_model, model.audio_projection).eval()

    # (batch, 1, 1001, 64): mono channel, 1001 time frames, 64 mel bins
    # Preprocessed by ClapFeatureExtractor: 10 s @ 48 kHz, hop_length=480
    dummy = torch.zeros(1, 1, 1001, 64)
    print(f"  exporting…")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy,
            str(out),
            input_names=["input_features"],
            output_names=["audio_embeds"],
            dynamic_axes={"input_features": {0: "batch"}, "audio_embeds": {0: "batch"}},
            opset_version=17,
        )
    report(out)


def export_dinov2() -> None:
    out = MODELS_DIR / "dinov2_small.onnx"
    if out.exists():
        print(f"  skip: {out.name} already present")
        return

    from transformers import Dinov2Model
    repo, rev = REVISIONS["dinov2"]
    print(f"  loading {repo} @ {rev[:12]}…")
    model = Dinov2Model.from_pretrained(repo, revision=rev)
    wrapper = Dinov2Wrapper(model).eval()

    dummy = torch.zeros(1, 3, 224, 224)
    print(f"  exporting…")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy,
            str(out),
            input_names=["pixel_values"],
            output_names=["last_hidden_state"],
            dynamic_axes={"pixel_values": {0: "batch"}, "last_hidden_state": {0: "batch"}},
            opset_version=17,
        )
    report(out)


# ── Whisper-tiny decoder wrapper (ADR 0013) ────────────────────────────────────

class WhisperDecoderWrapper(torch.nn.Module):
    """Whisper-tiny decoder + LM head; only projects the last token position.

    Input:
      encoder_hidden_states  (1, 1500, 384) f32  — from whisper_tiny_encoder.onnx
      decoder_input_ids      (1, seq_len)   int64 — prefix + generated so far
    Output:
      logits                 (1, 1, 51865) f32   — logits for position [-1] only
    """
    def __init__(self, model_instance):
        super().__init__()
        self.decoder = model_instance.model.decoder
        self.proj_out = model_instance.proj_out   # Whisper uses proj_out, not lm_head

    def forward(self, encoder_hidden_states, decoder_input_ids):
        out = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
        )
        last = out.last_hidden_state[:, -1:, :]   # (1, 1, 384)
        return self.proj_out(last)                 # (1, 1, 51865)


def export_whisper_decoder() -> None:
    lm_out = MODELS_DIR / "whisper_tiny_decoder.onnx"
    tok_out = MODELS_DIR / "whisper_tokenizer.json"

    if lm_out.exists() and tok_out.exists():
        print(f"  skip: whisper decoder artifacts already present")
        return

    from transformers import WhisperForConditionalGeneration, AutoTokenizer

    repo, rev = REVISIONS["whisper"]
    print(f"  loading {repo} @ {rev[:12]}…")
    model = WhisperForConditionalGeneration.from_pretrained(
        repo, revision=rev, torch_dtype=torch.float32
    ).eval()

    if not lm_out.exists():
        wrapper = WhisperDecoderWrapper(model).eval()
        dummy_enc = torch.zeros(1, 1500, 384)
        dummy_dec = torch.zeros(1, 4, dtype=torch.long)   # 4-token prefix
        print("  exporting whisper_tiny_decoder.onnx…")
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                (dummy_enc, dummy_dec),
                str(lm_out),
                input_names=["encoder_hidden_states", "decoder_input_ids"],
                output_names=["logits"],
                dynamic_axes={
                    "encoder_hidden_states": {0: "batch"},
                    "decoder_input_ids":     {0: "batch", 1: "seq_len"},
                    "logits":                {0: "batch"},
                },
                opset_version=17,
            )
        report(lm_out)

    if not tok_out.exists():
        print("  saving whisper_tokenizer.json…")
        tokenizer = AutoTokenizer.from_pretrained(repo, revision=rev, use_fast=True)
        tokenizer.backend_tokenizer.save(str(tok_out))
        print(f"  {tok_out.stat().st_size / 1e6:.2f} MB")


# ── Qwen2.5-0.5B wrappers (ADR 0012) ──────────────────────────────────────────

class Qwen25LMWrapper(torch.nn.Module):
    """Qwen2Model (base transformer) + LM head.

    Input:  inputs_embeds (1, n, hidden_size)
    Output: (last_hidden_state (1, n, hidden_size), logits (1, n, vocab_size))

    Used for both per-episode gain computation (4-token forward) and
    autoregressive generation (growing sequence).
    """
    def __init__(self, model_instance):
        super().__init__()
        self.model = model_instance.model   # Qwen2Model
        self.lm_head = model_instance.lm_head

    def forward(self, inputs_embeds):
        out = self.model(inputs_embeds=inputs_embeds)
        last_hidden = out.last_hidden_state          # (1, n, hidden_size)
        logits = self.lm_head(last_hidden)           # (1, n, vocab_size)
        return last_hidden, logits


def export_qwen25() -> None:
    lm_out = MODELS_DIR / "qwen25_lm.onnx"
    embed_out = MODELS_DIR / "qwen25_embed_table.safetensors"
    tok_out = MODELS_DIR / "qwen25_tokenizer.json"

    if lm_out.exists() and embed_out.exists() and tok_out.exists():
        print(f"  skip: qwen25 artifacts already present")
        return

    from transformers import Qwen2ForCausalLM, AutoTokenizer
    import safetensors.torch as st

    repo, rev = REVISIONS["qwen25"]
    print(f"  loading {repo} @ {rev or 'main'}…")
    model = Qwen2ForCausalLM.from_pretrained(repo, revision=rev, torch_dtype=torch.float32).eval()
    tokenizer = AutoTokenizer.from_pretrained(repo, revision=rev)

    hidden_size = model.config.hidden_size  # 896 for Qwen2.5-0.5B

    if not lm_out.exists():
        wrapper = Qwen25LMWrapper(model).eval()
        # Export with dynamic sequence length (batch fixed at 1).
        dummy = torch.zeros(1, 4, hidden_size)
        print(f"  exporting qwen25_lm.onnx (hidden_size={hidden_size})…")
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                dummy,
                str(lm_out),
                input_names=["inputs_embeds"],
                output_names=["last_hidden_state", "logits"],
                dynamic_axes={
                    "inputs_embeds":      {0: "batch", 1: "seq_len"},
                    "last_hidden_state":  {0: "batch", 1: "seq_len"},
                    "logits":             {0: "batch", 1: "seq_len"},
                },
                opset_version=17,
            )
        report(lm_out)

    if not embed_out.exists():
        print(f"  saving qwen25_embed_table.safetensors…")
        embed_weight = model.model.embed_tokens.weight.detach().float()
        st.save_file({"embed_tokens.weight": embed_weight}, str(embed_out))
        print(f"  {embed_out.stat().st_size / 1e6:.1f} MB  "
              f"shape={list(embed_weight.shape)}")

    if not tok_out.exists():
        print(f"  saving qwen25_tokenizer.json…")
        tokenizer.backend_tokenizer.save(str(tok_out))
        print(f"  {tok_out.stat().st_size / 1e6:.2f} MB")


# ── main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    MODELS_DIR.mkdir(exist_ok=True)

    print("=== Whisper-tiny encoder ===")
    export_whisper_encoder()

    print("=== CLAP-HTSAT-unfused audio encoder ===")
    export_clap_encoder()

    print("=== DINOv2-ViT-S/14 ===")
    export_dinov2()

    print("=== Whisper-tiny decoder (ASR, ADR 0013) ===")
    export_whisper_decoder()

    print("=== Qwen2.5-0.5B (action decoder) ===")
    export_qwen25()

    print("\nAll exports complete:")
    for f in sorted(MODELS_DIR.glob("*.onnx")):
        print(f"\n  {f.name}")
        report(f)

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
    "dinov3":  ("facebook/dinov3-vits16-pretrain-lvd1689m", None),  # ViT-S, 384-dim (drop-in replacement for DINOv2-S)
    # Qwen2.5-0.5B: revision should be pinned after first export; None = current main.
    "qwen25":  ("Qwen/Qwen2.5-0.5B", "060db6499f32faf8b98477b0a26969ef7d8b9987"),
    # Qwen3-0.6B: Item 5 upgrade (hidden_size=1024, vs Qwen2.5's 896)
    "qwen3":   ("Qwen/Qwen3-0.6B", None),
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


def export_dinov3() -> None:
    out = MODELS_DIR / "dinov3_small.onnx"
    if out.exists():
        print(f"  skip: {out.name} already present")
        return

    from transformers import AutoModel
    repo, rev = REVISIONS["dinov3"]
    print(f"  loading {repo} @ {rev or 'main'}…")
    model = AutoModel.from_pretrained(repo, revision=rev)
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

    # torch.onnx.export writes this one as external-data (unlike dinov2, despite
    # similar size); the MIGraphX EP resolves that external-data path relative to
    # the process CWD rather than the model file's directory, so it fails to load
    # unless the process happens to be running from models/. Re-pack as a single
    # self-contained file to remove that fragility.
    import onnx

    model_proto = onnx.load(str(out), load_external_data=True)
    onnx.save(model_proto, str(out), save_as_external_data=False)
    data_file = out.with_name(out.name + ".data")
    if data_file.exists():
        data_file.unlink()

    report(out)


# ── Whisper-tiny decoder wrapper (ADR 0013) ────────────────────────────────────

# Static sequence length for the decoder: 4-token prefix + 200 max generated tokens.
# MIGraphX requires static shapes; dynamic seq_len causes "invalid input for conversion
# to integer" at runtime. The Rust decode loop pads decoder_input_ids to this length and
# reads logits from position (real_len - 1). Causal attention ensures padding zeros after
# the current position have no effect on earlier positions' outputs.
WHISPER_DECODER_MAX_SEQ = 204  # PREFIX_LEN(4) + max_tokens(200)


class WhisperDecoderWrapper(torch.nn.Module):
    """Whisper-tiny decoder + full LM head with static input shape.

    Input:
      encoder_hidden_states  (1, 1500, 384)              f32   — static
      decoder_input_ids      (1, WHISPER_DECODER_MAX_SEQ) int64 — padded with 0s past real_len
    Output:
      logits  (1, WHISPER_DECODER_MAX_SEQ, 51865) f32 — Rust slices position (real_len-1)
    """
    def __init__(self, model_instance):
        super().__init__()
        self.decoder = model_instance.model.decoder
        self.proj_out = model_instance.proj_out

    def forward(self, encoder_hidden_states, decoder_input_ids):
        out = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
        )
        return self.proj_out(out.last_hidden_state)  # (1, MAX_SEQ, 51865)


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
        dummy_dec = torch.zeros(1, WHISPER_DECODER_MAX_SEQ, dtype=torch.long)
        print(f"  exporting whisper_tiny_decoder.onnx (static seq={WHISPER_DECODER_MAX_SEQ})…")
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                (dummy_enc, dummy_dec),
                str(lm_out),
                input_names=["encoder_hidden_states", "decoder_input_ids"],
                output_names=["logits"],
                # No dynamic_axes: both inputs are fully static for MIGraphX.
                opset_version=17,
            )
        report(lm_out)

    if not tok_out.exists():
        print("  saving whisper_tokenizer.json…")
        tokenizer = AutoTokenizer.from_pretrained(repo, revision=rev, use_fast=True)
        tokenizer.backend_tokenizer.save(str(tok_out))
        print(f"  {tok_out.stat().st_size / 1e6:.2f} MB")


# ── Qwen2.5-0.5B wrappers (ADR 0012) ──────────────────────────────────────────

# Fixed sequence length for qwen25_text_hidden.onnx (GPU distillation-target path).
# Covers p97+ of observed text inputs (p95 ≈ 205 chars ≈ 52–68 tokens); longer sequences
# are truncated to this length before embedding, which is acceptable for distillation.
# Must match MAX_TEXT_SEQ in crates/somnus/src/action.rs.
MAX_TEXT_SEQ = 128
# Fixed sequence length for qwen25_lm_fixed.onnx (GPU generation path).
# Must cover: recalled (3×5) + current (5) + history (256) + query (~50) + generated (200) ≈ 526.
# Must match MAX_GEN_SEQ in crates/somnus/src/action.rs.
MAX_GEN_SEQ = 1024

class Qwen25GainWrapper(torch.nn.Module):
    """Qwen2Model (base transformer only, no LM head).

    Used exclusively for per-episode action gain computation.
    Static input shape [1, N_REGIONS=5, hidden_size=896] — no dynamic axes,
    so ORT can plan static memory allocation and skip the LM head entirely.

    Input:  inputs_embeds    (1, 5, 896)
    Output: last_hidden_state (1, 5, 896)
    """
    def __init__(self, model_instance):
        super().__init__()
        self.model = model_instance.model  # Qwen2Model, no lm_head

    def forward(self, inputs_embeds):
        out = self.model(inputs_embeds=inputs_embeds)
        return out.last_hidden_state


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
    gain_out = MODELS_DIR / "qwen25_gain.onnx"
    lm_out = MODELS_DIR / "qwen25_lm.onnx"
    lm_fixed_out = MODELS_DIR / "qwen25_lm_fixed.onnx"
    text_hidden_out = MODELS_DIR / "qwen25_text_hidden.onnx"
    embed_out = MODELS_DIR / "qwen25_embed_table.safetensors"
    tok_out = MODELS_DIR / "qwen25_tokenizer.json"

    if (gain_out.exists() and lm_out.exists() and lm_fixed_out.exists()
            and text_hidden_out.exists() and embed_out.exists() and tok_out.exists()):
        print(f"  skip: qwen25 artifacts already present")
        return

    from transformers import Qwen2ForCausalLM, AutoTokenizer
    import safetensors.torch as st

    repo, rev = REVISIONS["qwen25"]
    print(f"  loading {repo} @ {rev or 'main'}…")
    model = Qwen2ForCausalLM.from_pretrained(repo, revision=rev, torch_dtype=torch.float32).eval()
    tokenizer = AutoTokenizer.from_pretrained(repo, revision=rev)

    assert model.config.tie_word_embeddings, (
        "qwen25_lm_fixed GPU path requires tied embeddings (lm_head == embed_tokens) "
        "so Rust can compute logits using the already-loaded embed_table."
    )

    hidden_size = model.config.hidden_size  # 896 for Qwen2.5-0.5B

    n_regions = 5  # must match N_REGIONS in crates/somnus/src/action.rs
    if not gain_out.exists():
        gain_wrapper = Qwen25GainWrapper(model).eval()
        dummy_gain = torch.zeros(1, n_regions, hidden_size)
        print(f"  exporting qwen25_gain.onnx (static [1,{n_regions},{hidden_size}], no LM head)…")
        with torch.no_grad():
            torch.onnx.export(
                gain_wrapper,
                dummy_gain,
                str(gain_out),
                input_names=["inputs_embeds"],
                output_names=["last_hidden_state"],
                # No dynamic_axes: static shape lets ORT plan memory statically
                opset_version=17,
            )
        report(gain_out)

    if not text_hidden_out.exists():
        # Static [1, MAX_TEXT_SEQ, hidden_size] — identical structure to gain but for text sequences.
        # Causal attention ensures end-padding zeros don't affect earlier positions, so Rust can
        # mean-pool only the first real_n positions without passing an attention_mask.
        text_hidden_wrapper = Qwen25GainWrapper(model).eval()
        dummy_text = torch.zeros(1, MAX_TEXT_SEQ, hidden_size)
        print(f"  exporting qwen25_text_hidden.onnx (static [1,{MAX_TEXT_SEQ},{hidden_size}], no LM head)…")
        with torch.no_grad():
            torch.onnx.export(
                text_hidden_wrapper,
                dummy_text,
                str(text_hidden_out),
                input_names=["inputs_embeds"],
                output_names=["last_hidden_state"],
                opset_version=17,
            )
        report(text_hidden_out)

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

    if not lm_fixed_out.exists():
        # Static [1, MAX_GEN_SEQ, hidden_size] — base transformer only (no LM head).
        # GPU-eligible via MIGraphX (fixed shape, no ScatterElements/INT64 outputs).
        # Causal attention ensures position i only attends to 0..i, so zero-padding after
        # seq_len cannot corrupt hidden[seq_len-1]. Rust reads hidden[seq_len-1] and computes
        # logits as dot(hidden, embed_table^T) using the already-loaded embed_table.
        # This avoids shipping vocab_size*hidden_size logits off GPU per decode step.
        lm_fixed_wrapper = Qwen25GainWrapper(model).eval()
        dummy_lm_fixed = torch.zeros(1, MAX_GEN_SEQ, hidden_size)
        print(f"  exporting qwen25_lm_fixed.onnx (static [1,{MAX_GEN_SEQ},{hidden_size}], no LM head)…")
        with torch.no_grad():
            torch.onnx.export(
                lm_fixed_wrapper,
                dummy_lm_fixed,
                str(lm_fixed_out),
                input_names=["inputs_embeds"],
                output_names=["last_hidden_state"],
                opset_version=17,
            )
        report(lm_fixed_out)

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


def export_qwen3() -> None:
    """Export Qwen3-0.6B with the same GPU-path completeness as export_qwen25():
    qwen3_gain.onnx, qwen3_text_hidden.onnx, qwen3_lm.onnx (dynamic, CPU generation
    path), qwen3_lm_fixed.onnx, embed table, tokenizer.

    Uses Qwen3ForCausalLM (not Qwen2ForCausalLM — Qwen3 has QK-norm layers Qwen2
    doesn't, so loading its weights into a Qwen2 architecture silently drops them).
    """
    gain_out = MODELS_DIR / "qwen3_gain.onnx"
    lm_out = MODELS_DIR / "qwen3_lm.onnx"
    lm_fixed_out = MODELS_DIR / "qwen3_lm_fixed.onnx"
    text_hidden_out = MODELS_DIR / "qwen3_text_hidden.onnx"
    embed_out = MODELS_DIR / "qwen3_embed_table.safetensors"
    tok_out = MODELS_DIR / "qwen3_tokenizer.json"

    if (gain_out.exists() and lm_out.exists() and lm_fixed_out.exists()
            and text_hidden_out.exists() and embed_out.exists() and tok_out.exists()):
        print(f"  skip: qwen3 artifacts already present")
        return

    from transformers import Qwen3ForCausalLM, AutoTokenizer
    import safetensors.torch as st

    repo, rev = REVISIONS["qwen3"]
    print(f"  loading {repo} @ {rev or 'main'}…")
    model = Qwen3ForCausalLM.from_pretrained(repo, revision=rev, torch_dtype=torch.float32).eval()
    tokenizer = AutoTokenizer.from_pretrained(repo, revision=rev)

    assert model.config.tie_word_embeddings, (
        "qwen3_lm_fixed GPU path requires tied embeddings (lm_head == embed_tokens) "
        "so Rust can compute logits using the already-loaded embed_table."
    )

    hidden_size = model.config.hidden_size  # 1024 for Qwen3-0.6B

    n_regions = 5  # must match N_REGIONS in crates/somnus/src/action.rs
    if not gain_out.exists():
        gain_wrapper = Qwen25GainWrapper(model).eval()
        dummy_gain = torch.zeros(1, n_regions, hidden_size)
        print(f"  exporting qwen3_gain.onnx (static [1,{n_regions},{hidden_size}], no LM head)…")
        with torch.no_grad():
            torch.onnx.export(
                gain_wrapper,
                dummy_gain,
                str(gain_out),
                input_names=["inputs_embeds"],
                output_names=["last_hidden_state"],
                opset_version=17,
            )
        report(gain_out)

    if not text_hidden_out.exists():
        text_hidden_wrapper = Qwen25GainWrapper(model).eval()
        dummy_text = torch.zeros(1, MAX_TEXT_SEQ, hidden_size)
        print(f"  exporting qwen3_text_hidden.onnx (static [1,{MAX_TEXT_SEQ},{hidden_size}], no LM head)…")
        with torch.no_grad():
            torch.onnx.export(
                text_hidden_wrapper,
                dummy_text,
                str(text_hidden_out),
                input_names=["inputs_embeds"],
                output_names=["last_hidden_state"],
                opset_version=17,
            )
        report(text_hidden_out)

    if not lm_out.exists():
        wrapper = Qwen25LMWrapper(model).eval()
        dummy = torch.zeros(1, 4, hidden_size)
        print(f"  exporting qwen3_lm.onnx (hidden_size={hidden_size})…")
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

    if not lm_fixed_out.exists():
        lm_fixed_wrapper = Qwen25GainWrapper(model).eval()
        dummy_lm_fixed = torch.zeros(1, MAX_GEN_SEQ, hidden_size)
        print(f"  exporting qwen3_lm_fixed.onnx (static [1,{MAX_GEN_SEQ},{hidden_size}], no LM head)…")
        with torch.no_grad():
            torch.onnx.export(
                lm_fixed_wrapper,
                dummy_lm_fixed,
                str(lm_fixed_out),
                input_names=["inputs_embeds"],
                output_names=["last_hidden_state"],
                opset_version=17,
            )
        report(lm_fixed_out)

    if not embed_out.exists():
        print(f"  saving qwen3_embed_table.safetensors…")
        embed_weight = model.model.embed_tokens.weight.detach().float()
        st.save_file({"embed_tokens.weight": embed_weight}, str(embed_out))
        print(f"  {embed_out.stat().st_size / 1e6:.1f} MB  "
              f"shape={list(embed_weight.shape)}")

    if not tok_out.exists():
        print(f"  saving qwen3_tokenizer.json…")
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

    print("=== DINOv3-ViT-S (Item 6 test) ===")
    export_dinov3()

    print("=== Whisper-tiny decoder (ASR, ADR 0013) ===")
    export_whisper_decoder()

    print("=== Qwen2.5-0.5B (action decoder) ===")
    export_qwen25()

    print("=== Qwen3-0.6B (action decoder, Item 5) ===")
    export_qwen3()

    print("\nAll exports complete:")
    for f in sorted(MODELS_DIR.glob("*.onnx")):
        print(f"\n  {f.name}")
        report(f)

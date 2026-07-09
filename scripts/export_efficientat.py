#!/usr/bin/env python3
"""Export EfficientAT mn10 (MobileNetV3, AudioSet-pretrained) backbone to ONNX.

Requires an isolated venv — EfficientAT needs torchvision, whose available wheels
don't pair with scripts/.venv's pinned torch 2.11.0. Run instead with:

    scripts/.venv-efficientat/bin/python scripts/export_efficientat.py

(create it once with: python3 -m venv scripts/.venv-efficientat &&
 scripts/.venv-efficientat/bin/pip install torch torchvision --index-url \
 https://download.pytorch.org/whl/cpu && scripts/.venv-efficientat/bin/pip install \
 onnx onnxconverter_common)

Exports ONLY the MobileNetV3 conv backbone (input: precomputed mel spectrogram,
output: 960-dim pooled feature) — deliberately excludes AugmentMelSTFT's mel
frontend, which contains an STFT op. Baking that in was tested and works
numerically (verified against the PyTorch reference), but reintroduces the
exotic-op / MIGraphX-compat risk that mn10 was chosen to avoid (gaps.md item 9).
Mel extraction instead happens in Rust — see
crates/somnus-encoding/src/preprocess/mel.rs::Mn10Mel, parity-tested against
AugmentMelSTFT in crates/somnus-encoding/tests/mel_parity.rs.

Output:
  efficientat_mn10_encoder.onnx
    in:  mel_spectrogram (1, 1, 128, 1000) — 10 s @ 32 kHz, 128 mel bins, Mn10Mel layout
    out: env_embedding   (1, 960)          — pooled conv feature (pre-classifier)
"""

import subprocess
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
VENDOR_DIR = REPO_ROOT / "scripts" / ".vendor-efficientat"

# Pinned commit (fschmid56/EfficientAT), consistent with REVISIONS pinning in export_models.py.
EFFICIENTAT_COMMIT = "a425fdce92572e602a1d5634799bd9f1f2efa806"


def ensure_efficientat_source() -> Path:
    """Shallow-clone fschmid56/EfficientAT at a pinned commit (gitignored, code-only;
    weights are fetched separately by get_model() via torch.hub at export time)."""
    if VENDOR_DIR.exists():
        return VENDOR_DIR
    print(f"  cloning fschmid56/EfficientAT @ {EFFICIENTAT_COMMIT[:12]}…")
    subprocess.run(
        ["git", "clone", "https://github.com/fschmid56/EfficientAT.git", str(VENDOR_DIR)],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(VENDOR_DIR), "checkout", EFFICIENTAT_COMMIT],
        check=True,
    )
    return VENDOR_DIR


class Mn10BackboneWrapper(nn.Module):
    """Backbone only — takes a precomputed mel spectrogram, skips AugmentMelSTFT
    (mel extraction lives in Rust, see module docstring) and skips the classifier
    head (only the pooled penultimate feature is used downstream)."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        _logits, features = self.backbone(mel_spectrogram)
        return features


def report(path: Path) -> None:
    mb = path.stat().st_size / 1024 / 1024
    print(f"  -> {path.name} ({mb:.1f} MB)")


def export_efficientat_mn10() -> None:
    out = MODELS_DIR / "efficientat_mn10_encoder.onnx"
    if out.exists():
        print(f"  skip: {out.name} already present")
        return

    src_dir = ensure_efficientat_source()
    sys.path.insert(0, str(src_dir))
    # get_model() reads metadata/class_labels_indices.csv via a relative path.
    import os
    old_cwd = os.getcwd()
    os.chdir(src_dir)
    try:
        from models.mn.model import get_model
        print("  loading mn10_as (AudioSet-pretrained) weights…")
        backbone = get_model(width_mult=1.0, pretrained_name="mn10_as", head_type="mlp")
    finally:
        os.chdir(old_cwd)

    wrapper = Mn10BackboneWrapper(backbone).eval()

    # (batch, 1, 128, 1000): mono channel, 128 mel bins, 1000 time frames — matches
    # Mn10Mel::compute()'s output layout (mel-bin major, no transpose).
    dummy = torch.zeros(1, 1, 128, 1000)
    print("  exporting…")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy,
            str(out),
            input_names=["mel_spectrogram"],
            output_names=["env_embedding"],
            dynamic_axes={"mel_spectrogram": {0: "batch"}, "env_embedding": {0: "batch"}},
            opset_version=17,
            dynamo=False,
        )

    # Guard against accidentally re-introducing an STFT/DFT op (the whole point of
    # excluding AugmentMelSTFT from the graph — see module docstring).
    import onnx
    graph = onnx.load(str(out))
    ops = {n.op_type for n in graph.graph.node}
    forbidden = ops & {"STFT", "DFT"}
    if forbidden:
        raise RuntimeError(f"unexpected exotic ops in exported graph: {forbidden}")
    print(f"  op types: {sorted(ops)}")

    report(out)


if __name__ == "__main__":
    MODELS_DIR.mkdir(exist_ok=True)
    print("=== EfficientAT mn10 ===")
    export_efficientat_mn10()
    print("\nDone.")

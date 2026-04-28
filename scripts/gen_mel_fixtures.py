"""Generate mel-spectrogram parity fixtures for somnus-encoding tests.

Outputs raw f32 (little-endian) + shape metadata to
crates/somnus-encoding/tests/fixtures/.

Three inputs per encoder:
  zeros   — silence
  sine440 — 440 Hz pure tone
  (real clip not generated here; add manually if needed)
"""

import struct
import os
import numpy as np
from pathlib import Path
from transformers import WhisperFeatureExtractor, ClapFeatureExtractor
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).parent.parent
OUT_DIR = REPO_ROOT / "crates" / "somnus-encoding" / "tests" / "fixtures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

whisper_fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")
clap_fe    = ClapFeatureExtractor.from_pretrained("laion/clap-htsat-unfused")

WHISPER_SR   = 16_000
WHISPER_SAMP = 480_000   # 30 s
CLAP_SR      = 48_000
CLAP_SAMP    = 480_000   # 10 s


def save_f32(path: Path, arr: np.ndarray):
    arr = arr.astype(np.float32)
    meta_path = path.with_suffix(".shape")
    meta_path.write_text(" ".join(str(d) for d in arr.shape))
    path.write_bytes(arr.tobytes())
    print(f"  wrote {path.name}  shape={arr.shape}  bytes={path.stat().st_size}")


# ── Whisper fixtures ──────────────────────────────────────────────────────────
print("=== Whisper ===")
for name, pcm in [
    ("zeros",   np.zeros(WHISPER_SAMP, dtype=np.float32)),
    ("sine440", np.sin(2 * np.pi * 440 * np.arange(WHISPER_SAMP) / WHISPER_SR).astype(np.float32)),
]:
    # WhisperFeatureExtractor.__call__ expects a list of 1-D arrays
    out = whisper_fe(pcm, sampling_rate=WHISPER_SR, return_tensors="np")
    mel = out["input_features"][0]   # (80, 3000)
    assert mel.shape == (80, 3000), mel.shape
    save_f32(OUT_DIR / f"whisper_{name}.f32", mel)

# ── CLAP fixtures ─────────────────────────────────────────────────────────────
print("=== CLAP ===")
for name, pcm in [
    ("zeros",   np.zeros(CLAP_SAMP, dtype=np.float32)),
    ("sine440", np.sin(2 * np.pi * 440 * np.arange(CLAP_SAMP) / CLAP_SR).astype(np.float32)),
]:
    out = clap_fe(pcm, sampling_rate=CLAP_SR, return_tensors="np")
    # ClapFeatureExtractor returns shape (1, 1, 1001, 64) — strip batch+channel
    mel = out["input_features"][0, 0]   # (1001, 64)
    assert mel.shape == (1001, 64), mel.shape
    save_f32(OUT_DIR / f"clap_{name}.f32", mel)

# ── DINOv2 image fixtures ─────────────────────────────────────────────────────
print("=== DINOv2 image ===")
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess_image(img_arr: np.ndarray) -> np.ndarray:
    # (H, W, 3) uint8 → (1, 3, H, W) float32 in [0, 1]
    t = torch.from_numpy(img_arr).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    # bilinear resize to 224×224
    t = F.interpolate(t, size=(224, 224), mode="bilinear", align_corners=False)
    arr = t.squeeze(0).numpy()  # (3, 224, 224)
    mean = IMAGENET_MEAN[:, None, None]
    std  = IMAGENET_STD[:, None, None]
    return (arr - mean) / std

for name, img_arr in [
    ("black",    np.zeros((240, 320, 3), dtype=np.uint8)),
    ("gray128",  np.full((240, 320, 3), 128, dtype=np.uint8)),
    ("checkerboard", np.tile(
        np.array([[0, 255], [255, 0]], dtype=np.uint8)[:, :, None],
        (120, 160, 3)
    )),
]:
    tensor = preprocess_image(img_arr)
    assert tensor.shape == (3, 224, 224), tensor.shape
    save_f32(OUT_DIR / f"dino_{name}.f32", tensor)

print("\nAll fixtures written to", OUT_DIR)

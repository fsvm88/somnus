# models/

ONNX model artifacts for Phase 2 frozen encoders. **Not committed** — populate by running:

```
scripts/.venv/bin/python scripts/export_models.py
```

The new `torch.onnx` exporter (torch ≥ 2.x) writes weights to an external data file alongside the
graph file. Both files must be present in the same directory for OnnxRuntime to load the model.

## Canonical artifacts

Exported from HF revisions pinned in ADR 0003 using `scripts/export_models.py`.
All models are FP32, opset 18.

### whisper_tiny_encoder

- **Source:** `openai/whisper-tiny` @ `169d4a4341b33bc18d8881c4b69c2e104e1cc0af`
- **Input:** `input_features` — `(batch, 80, 3000)` float32 — mel spectrogram, 30 s @ 16 kHz, 80 bins
- **Output:** `last_hidden_state` — `(batch, 1500, 384)` float32 — Rust mean-pools to `(batch, 384)`

| File | Bytes | SHA-256 |
|------|-------|---------|
| `whisper_tiny_encoder.onnx` | 33 138 | `ea1e44119df36522c35d593c7f0eb6cf768cfce2584124e22910eab710d58ab1` |
| `whisper_tiny_encoder.onnx.data` | 32 964 608 | `3547dcf211db5aaaae4703c14a38d4ee4d74f640847e9d24d77da926521edaa9` |

### clap_audio_encoder

- **Source:** `laion/clap-htsat-unfused` @ `8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a`
- **Input:** `input_features` — `(batch, 1, 1001, 64)` float32 — mono mel spectrogram, 10 s @ 48 kHz, 64 bins
- **Output:** `audio_embeds` — `(batch, 512)` float32 — L2-normalised projection embedding

| File | Bytes | SHA-256 |
|------|-------|---------|
| `clap_audio_encoder.onnx` | 2 316 949 | `7f38ee0248388e1bc8f8ddd6150a97d59018be5dc3f725a5cc685c6006e7ea08` |
| `clap_audio_encoder.onnx.data` | 117 178 368 | `5d636ed289665b51165132f5242c39be8e6acae38d070bf0a7c912621b473af5` |

### dinov2_small

- **Source:** `facebook/dinov2-small` @ `ed25f3a31f01632728cabb09d1542f84ab7b0056`
- **Input:** `pixel_values` — `(batch, 3, 224, 224)` float32 — ImageNet-normalised RGB
- **Output:** `last_hidden_state` — `(batch, 257, 384)` float32 — token 0 is CLS; Rust mean-pools patch tokens `[1:]` to `(batch, 384)`

| File | Bytes | SHA-256 |
|------|-------|---------|
| `dinov2_small.onnx` | 1 532 061 | `0162d995523dab09f72f980f738fbeaf17ecc660ee5e24cb4755edda379db4e9` |
| `dinov2_small.onnx.data` | 88 342 528 | `be262910b67687076ec54b9d21c08d4dd3a96a62478eeecc6c68082ce07611cc` |

## Verification

```bash
sha256sum models/*.onnx models/*.onnx.data
```

Expected output (order may differ):

```
ea1e44119df36522c35d593c7f0eb6cf768cfce2584124e22910eab710d58ab1  models/whisper_tiny_encoder.onnx
3547dcf211db5aaaae4703c14a38d4ee4d74f640847e9d24d77da926521edaa9  models/whisper_tiny_encoder.onnx.data
7f38ee0248388e1bc8f8ddd6150a97d59018be5dc3f725a5cc685c6006e7ea08  models/clap_audio_encoder.onnx
5d636ed289665b51165132f5242c39be8e6acae38d070bf0a7c912621b473af5  models/clap_audio_encoder.onnx.data
0162d995523dab09f72f980f738fbeaf17ecc660ee5e24cb4755edda379db4e9  models/dinov2_small.onnx
be262910b67687076ec54b9d21c08d4dd3a96a62478eeecc6c68082ce07611cc  models/dinov2_small.onnx.data
```

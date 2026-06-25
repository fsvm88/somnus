"""
INT8 dynamic quantization of KV-cache ONNX models (Q-B).

Produces:
  qwen25_lm_decode_kv_int8.onnx
  qwen25_lm_prefill_kv_int8.onnx  (optional, --prefill)

Then benchmarks FP32 vs INT8 for the decode step (the per-token hot path).
Without VNNI (AVX2 only, Ryzen 9 5950X), speedup comes from bandwidth reduction
(700 MB → 175 MB weights/step); quant/dequant overhead may partially offset this.
The benchmark tells us whether to wire up the Rust fast-path.

Usage:
  .venv/bin/python scripts/quantize_kv.py [--models-dir models/] [--prefill] [--bench-runs 50]
"""
import argparse, time, sys
from pathlib import Path
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType


N_KV_LAYERS = 24
N_KV_HEADS  = 2
HEAD_DIM    = 64
LM_HIDDEN   = 896
KV_SIZE     = 2048
VOCAB_SIZE  = 151936


def quantize(src: Path, dst: Path, reduce_range: bool) -> None:
    print(f"  quantizing {src.name} → {dst.name}  (reduce_range={reduce_range})")
    t0 = time.time()
    quantize_dynamic(
        str(src), str(dst),
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=reduce_range,
    )
    print(f"  done in {time.time()-t0:.1f}s  ({dst.stat().st_size/1e9:.2f} GB)")


def make_decode_inputs(seq_pos: int) -> dict:
    """Synthetic single decode-step inputs matching qwen25_lm_decode_kv.onnx signature.

    Inputs (from Rust generate_kv):
      inputs_embeds  [1, 1, LM_HIDDEN]     float32
      position_ids   [1, 1]                int64
      attention_mask [1, 1, 1, KV_SIZE+1]  float32 (0 = attend, -inf = mask)
      past_key_{i}   [1, N_KV_HEADS, KV_SIZE, HEAD_DIM]  float32
      past_value_{i} same
    """
    rng = np.random.default_rng(42)

    # attention_mask: shape (1, 1, 1, KV_SIZE+1); -inf everywhere except past tokens
    mask = np.full((1, 1, 1, KV_SIZE + 1), float("-inf"), dtype=np.float32)
    mask[0, 0, 0, :seq_pos] = 0.0  # past tokens are attended to

    inputs = {
        "inputs_embeds":  rng.standard_normal((1, 1, LM_HIDDEN)).astype(np.float32),
        "attention_mask": mask,
        "position_ids":   np.array([[seq_pos]], dtype=np.int64),
    }
    for i in range(N_KV_LAYERS):
        k = rng.standard_normal((1, N_KV_HEADS, KV_SIZE, HEAD_DIM)).astype(np.float32)
        v = rng.standard_normal((1, N_KV_HEADS, KV_SIZE, HEAD_DIM)).astype(np.float32)
        inputs[f"past_key_{i}"]   = k
        inputs[f"past_value_{i}"] = v
    return inputs


def bench_session(sess: ort.InferenceSession, inputs: dict, n_runs: int) -> float:
    """Returns median ms per run."""
    # warmup
    for _ in range(3):
        sess.run(None, inputs)
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sess.run(None, inputs)
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


def bench(fp32_path: Path, int8_path: Path, n_runs: int) -> None:
    print(f"\n── Benchmarking decode step ({n_runs} runs) ──")
    import os
    total = os.cpu_count() or 4
    intra = max(total // 2, 1)
    inter = max(total // 4, 1)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = intra  # mirrors Rust cpu_session / onnx_thread_budget()
    opts.inter_op_num_threads = inter

    inputs = make_decode_inputs(seq_pos=64)

    print("  loading FP32 session…")
    fp32_sess = ort.InferenceSession(str(fp32_path), sess_options=opts,
                                     providers=["CPUExecutionProvider"])
    print("  loading INT8 session…")
    int8_sess = ort.InferenceSession(str(int8_path), sess_options=opts,
                                     providers=["CPUExecutionProvider"])

    fp32_ms = bench_session(fp32_sess, inputs, n_runs)
    int8_ms = bench_session(int8_sess, inputs, n_runs)

    speedup = fp32_ms / int8_ms if int8_ms > 0 else 0
    print(f"\n  FP32 median: {fp32_ms:.1f} ms/step")
    print(f"  INT8 median: {int8_ms:.1f} ms/step")
    print(f"  Speedup:     {speedup:.2f}×")

    if speedup >= 1.3:
        print("\n  VERDICT: INT8 is faster — wire up Rust fast-path (Q-B step 3)")
    elif speedup >= 1.0:
        print("\n  VERDICT: marginal gain — consider wiring up anyway (< 1.3×)")
    else:
        print("\n  VERDICT: INT8 is SLOWER — do not deploy; no VNNI, overhead dominates")

    # Token-match check: same logit argmax?
    fp32_out = fp32_sess.run(["logits"], inputs)[0]  # [1, 1, vocab]
    int8_out = int8_sess.run(["logits"], inputs)[0]
    match = int(np.argmax(fp32_out[0, 0])) == int(np.argmax(int8_out[0, 0]))
    print(f"  Token match (argmax):  {'PASS' if match else 'FAIL'}")

    fp32_top = set(np.argsort(fp32_out[0, 0])[-10:].tolist())
    int8_top = set(np.argsort(int8_out[0, 0])[-10:].tolist())
    overlap = len(fp32_top & int8_top)
    print(f"  Top-10 overlap:        {overlap}/10")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default="models/")
    ap.add_argument("--prefill", action="store_true", help="also quantize prefill model")
    ap.add_argument("--bench-runs", type=int, default=30)
    ap.add_argument("--skip-quantize", action="store_true", help="skip quantization, bench only")
    ap.add_argument("--reduce-range", action="store_true", default=True,
                    help="use reduce_range=True (needed on non-VNNI CPUs like Ryzen 5950X)")
    args = ap.parse_args()

    models = Path(args.models_dir)
    decode_fp32  = models / "qwen25_lm_decode_kv.onnx"
    decode_int8  = models / "qwen25_lm_decode_kv_int8.onnx"
    prefill_fp32 = models / "qwen25_lm_prefill_kv.onnx"
    prefill_int8 = models / "qwen25_lm_prefill_kv_int8.onnx"

    if not args.skip_quantize:
        print("── Quantizing decode model ──")
        quantize(decode_fp32, decode_int8, reduce_range=args.reduce_range)

        if args.prefill:
            print("\n── Quantizing prefill model ──")
            quantize(prefill_fp32, prefill_int8, reduce_range=args.reduce_range)

    if not decode_int8.exists():
        print(f"ERROR: {decode_int8} not found; run without --skip-quantize first")
        sys.exit(1)

    bench(decode_fp32, decode_int8, args.bench_runs)


if __name__ == "__main__":
    main()

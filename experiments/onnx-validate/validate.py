#!/usr/bin/env python3
"""
Phase 0 gate: burn ↔ ONNX numeric validation.

Loads the data prepared by `onnx_validate_prepare` (Rust binary), builds the
two ONNX perception graphs from the exported burn weights, and compares outputs
against what burn produced.

Checks:
  1. Encoder z vectors (tests Linear weight convention / transpose)
  2. Binding episode 1, call with prev_ep=zeros (tests attention + projection)
  3. Binding episode 2, call with prev_ep=episode_1 (tests GRU gate order + weights)
  4. GRU hidden state after call 2 (direct GRU weight check)

Usage:
  python3 experiments/onnx-validate/validate.py --data-dir /tmp/somnus_onnx_validate
"""

import argparse
import json
import os
import sys
import numpy as np
import onnxruntime as ort

# Add scripts dir so we can import the graph builder
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'scripts'))
from export_perception_graph import (
    build_encoder_graph, build_binding_graph, load_weights_from_dir,
    N_UNITS, K_ACTIVE, EPISODE_DIM, GRU_HIDDEN,
)
BOOST_STRENGTH = 1.0
TARGET_DC      = K_ACTIVE / N_UNITS
import onnx


def load_f32(path: str) -> np.ndarray:
    return np.fromfile(path, dtype=np.float32)


def check(label: str, burn_val: np.ndarray, onnx_val: np.ndarray, atol: float = 1e-4):
    diff = float(np.max(np.abs(burn_val - onnx_val)))
    status = "PASS" if diff <= atol else "FAIL"
    print(f"  [{status}] {label}: max_abs_diff = {diff:.2e}  (atol={atol:.0e})")
    return diff <= atol


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True,
                    help="Directory written by onnx_validate_prepare")
    ap.add_argument("--migraphx", action="store_true",
                    help="Also run on MIGraphX EP")
    ap.add_argument("--atol", type=float, default=1e-4,
                    help="Absolute tolerance for episode comparison")
    args = ap.parse_args()

    d = args.data_dir
    weights_dir = os.path.join(d, "weights")

    with open(os.path.join(d, "meta.json")) as f:
        meta = json.load(f)
    print(f"Meta: {meta}")
    print()

    # ── Load burn reference data ───────────────────────────────────────────────
    speech_enc   = load_f32(os.path.join(d, "speech_enc.f32")).reshape(1, -1)
    env_enc      = load_f32(os.path.join(d, "env_enc.f32")).reshape(1, -1)
    visual_enc   = load_f32(os.path.join(d, "visual_enc.f32")).reshape(1, -1)
    language_enc = load_f32(os.path.join(d, "language_enc.f32")).reshape(1, -1)

    dcs = [load_f32(os.path.join(d, f"dc_{i}.f32")).reshape(1, -1) for i in range(5)]
    z_burn   = [load_f32(os.path.join(d, f"z_{i}.f32")) for i in range(5)]
    spz_burn = [load_f32(os.path.join(d, f"sparse_z_{i}.f32")).reshape(1, -1) for i in range(5)]

    ep1_burn  = load_f32(os.path.join(d, "episode_1_burn.f32"))
    ep2_burn  = load_f32(os.path.join(d, "episode_2_burn.f32"))
    h1_burn   = load_f32(os.path.join(d, "gru_h_1_burn.f32"))
    h2_burn   = load_f32(os.path.join(d, "gru_h_2_burn.f32"))
    gru_h_init = load_f32(os.path.join(d, "gru_hidden_init.f32")).reshape(1, 1, -1)

    print("Burn reference loaded.")
    print(f"  episode_1[0:4] = {ep1_burn[:4]}")
    print(f"  episode_2[0:4] = {ep2_burn[:4]}")
    print(f"  gru_h_2[0:4]   = {h2_burn[:4]}")
    print()

    # ── Build ONNX graphs ─────────────────────────────────────────────────────
    print("Loading weights and building ONNX graphs...")
    weights = load_weights_from_dir(weights_dir)

    enc_model  = build_encoder_graph(weights)
    bind_model = build_binding_graph(weights)
    onnx.checker.check_model(enc_model)
    onnx.checker.check_model(bind_model)

    enc_path  = os.path.join(d, "enc_test.onnx")
    bind_path = os.path.join(d, "bind_test.onnx")
    onnx.save(enc_model, enc_path)
    onnx.save(bind_model, bind_path)
    print("  Graphs built.")
    print()

    def run_validation(ep_label: str, providers: list):
        print(f"=== Validation: {ep_label} ===")
        sess_enc  = ort.InferenceSession(enc_path,  providers=providers)
        sess_bind = ort.InferenceSession(bind_path, providers=providers)

        # ── Check 1: Encoder z vectors ─────────────────────────────────────────
        enc_feeds = {
            "speech_enc":   speech_enc,
            "env_enc":      env_enc,
            "visual_enc":   visual_enc,
            "language_enc": language_enc,
        }
        z_onnx = sess_enc.run(None, enc_feeds)

        print("Encoder z vectors:")
        all_ok = True
        for i in range(5):
            all_ok &= check(f"z_{i}", z_burn[i], z_onnx[i][0], atol=1e-5)

        # ── Check 2: CPU topk (must match burn's sparse_z) ────────────────────
        # dc = zeros → boosted = z + BOOST*(target_dc - 0) = z + 0.0195
        # Constant offset, so topk(boosted) == topk(z)
        target_dc = K_ACTIVE / N_UNITS
        spz_onnx = []
        for i in range(5):
            dc_i = dcs[i][0]
            z_i  = z_onnx[i][0]
            boosted = z_i + BOOST_STRENGTH * (target_dc - dc_i)
            topk_idx = np.argpartition(boosted, -K_ACTIVE)[-K_ACTIVE:]
            mask = np.zeros(N_UNITS, dtype=np.float32)
            mask[topk_idx] = 1.0
            spz_i = (z_i * mask).reshape(1, -1)
            spz_onnx.append(spz_i)

        print("CPU topk sparse_z sets (burn vs ONNX):")
        for i in range(5):
            burn_nz = set(np.nonzero(spz_burn[i][0])[0].tolist())
            onnx_nz = set(np.nonzero(spz_onnx[i][0])[0].tolist())
            same_set = burn_nz == onnx_nz
            val_diff = float(np.max(np.abs(spz_burn[i] - spz_onnx[i])))
            ok = same_set and val_diff < 1e-5
            all_ok &= ok
            print(f"  [{'PASS' if ok else 'FAIL'}] sparse_z_{i}: same_set={same_set} val_diff={val_diff:.2e}")

        # ── Check 3: Binding episode 1 (prev_ep=zeros, GRU output=zeros) ──────
        bind_feeds_1 = {f"sp_{i}": spz_onnx[i] for i in range(5)}
        bind_feeds_1["prev_episode"] = np.zeros((1, EPISODE_DIM), np.float32)
        bind_feeds_1["gru_hidden"]   = gru_h_init.copy()
        out1 = sess_bind.run(None, bind_feeds_1)
        ep1_onnx = out1[0][0]
        h1_onnx  = out1[1][0, 0, :]

        print("Binding call 1 (prev_ep=zeros):")
        all_ok &= check("episode_1", ep1_burn, ep1_onnx, atol=args.atol)
        all_ok &= check("gru_h_1",   h1_burn,  h1_onnx,  atol=1e-5)

        # ── Check 4: Binding episode 2 (prev_ep=episode_1, GRU non-trivial) ───
        bind_feeds_2 = {f"sp_{i}": spz_onnx[i] for i in range(5)}
        bind_feeds_2["prev_episode"] = ep1_onnx.reshape(1, -1)
        bind_feeds_2["gru_hidden"]   = h1_onnx.reshape(1, 1, -1)
        out2 = sess_bind.run(None, bind_feeds_2)
        ep2_onnx = out2[0][0]
        h2_onnx  = out2[1][0, 0, :]

        print("Binding call 2 (prev_ep=episode_1, GRU exercised):")
        all_ok &= check("episode_2", ep2_burn, ep2_onnx, atol=args.atol)
        all_ok &= check("gru_h_2",   h2_burn,  h2_onnx,  atol=args.atol)

        print()
        print(f"{'ALL PASS' if all_ok else 'SOME CHECKS FAILED'} on {ep_label}")
        print()
        return all_ok

    ok_cpu = run_validation("CPU EP", ["CPUExecutionProvider"])

    ok_mgx = True
    if args.migraphx:
        ok_mgx = run_validation("MIGraphX EP", ["MIGraphXExecutionProvider", "CPUExecutionProvider"])

    sys.exit(0 if (ok_cpu and ok_mgx) else 1)


if __name__ == "__main__":
    main()

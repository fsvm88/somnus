#!/usr/bin/env python3
"""
Build the somnus perception inference ONNX graphs (two-graph split).

Produces two ONNX files:
  perception_encoder.onnx  — five linear forwards + cross-modal projection
  perception_binding.onnx  — binding area (projections + GRU + attention)

The caller (Rust PerceptionProcessor) runs them with one CPU step in between:
  1. encoder.onnx  →  5 z vectors [1, N_UNITS]
  2. CPU: topk + mask + duty-cycle update (uses non-zero z values; fast)
  3. binding.onnx  →  episode [1, 640]  +  gru_hidden_out [1, 1, 256]

Architecture and MIGraphX bug notes: ai-docs/research/gpu-unification-design.md.

MIGraphX bug (RDNA4/gfx1201): ScatterElements feeding into graph outputs triggers
"invalid input for conversion to integer" when >= 4 such outputs are present.
The split design avoids this entirely — neither graph has ScatterElements at all.

Usage:
  # Validate with random weights (numpy reference):
  python3 scripts/export_perception_graph.py --validate-numpy

  # Build from burn weights exported by Rust (PerceptionProcessor::export_weights_for_onnx):
  python3 scripts/export_perception_graph.py --weights-dir <dir> [--out-dir <dir>]
"""

import argparse
import json
import os
import numpy as np
import onnx
import onnx.helper as h
import onnx.numpy_helper as nph
import onnxruntime as ort
from onnx import TensorProto

# ── Dimensions (must match crates/somnus-perception/src/*.rs) ────────────────────
N_UNITS         = 1024
K_ACTIVE        = 20
D_PROJ          = 128
N_HEADS         = 4
D_HEAD          = D_PROJ // N_HEADS     # 32
EPISODE_DIM     = 640                   # 5 * D_PROJ
GRU_INPUT       = EPISODE_DIM
GRU_HIDDEN      = 256
CROSS_MODAL_DIM = 384 + 512 + 384 + 896  # 2176
DUTY_CYCLE_ALPHA = 1.0 / 1000.0

# Region encoder input dimensions [speech, env, visual, symbolic, language]
REGION_DIMS = [384, 512, 384, 384, 896]


# ── Graph 1: encoder ──────────────────────────────────────────────────────────

def build_encoder_graph(weights: dict) -> onnx.ModelProto:
    """
    Inputs:
      speech_enc   [1, 384]
      env_enc      [1, 512]
      visual_enc   [1, 384]
      language_enc [1, 896]

    Outputs (all [1, N_UNITS]):
      z_0 .. z_4   — raw pre-topk activations per region (symbolic derived internally)
    """
    w = weights
    nodes = [
        # Cross-modal concat → symbolic encoder input (derived, not a graph input)
        h.make_node("Concat",
                    ["speech_enc", "env_enc", "visual_enc", "language_enc"],
                    ["cross_cat"], axis=-1),
        h.make_node("MatMul", ["cross_cat", "W_cross"], ["symbolic_enc"]),
        # Linear forwards for each region
        h.make_node("MatMul", ["speech_enc",   "W_enc_0"], ["z_0"]),
        h.make_node("MatMul", ["env_enc",       "W_enc_1"], ["z_1"]),
        h.make_node("MatMul", ["visual_enc",    "W_enc_2"], ["z_2"]),
        h.make_node("MatMul", ["symbolic_enc",  "W_enc_3"], ["z_3"]),
        h.make_node("MatMul", ["language_enc",  "W_enc_4"], ["z_4"]),
    ]
    initializers = [
        nph.from_array(w["W_cross"].astype(np.float32),  name="W_cross"),
        nph.from_array(w["W_enc_0"].astype(np.float32),  name="W_enc_0"),
        nph.from_array(w["W_enc_1"].astype(np.float32),  name="W_enc_1"),
        nph.from_array(w["W_enc_2"].astype(np.float32),  name="W_enc_2"),
        nph.from_array(w["W_enc_3"].astype(np.float32),  name="W_enc_3"),
        nph.from_array(w["W_enc_4"].astype(np.float32),  name="W_enc_4"),
    ]
    graph_inputs = [
        h.make_tensor_value_info("speech_enc",   TensorProto.FLOAT, [1, 384]),
        h.make_tensor_value_info("env_enc",       TensorProto.FLOAT, [1, 512]),
        h.make_tensor_value_info("visual_enc",    TensorProto.FLOAT, [1, 384]),
        h.make_tensor_value_info("language_enc",  TensorProto.FLOAT, [1, 896]),
    ]
    graph_outputs = [
        h.make_tensor_value_info(f"z_{i}", TensorProto.FLOAT, [1, N_UNITS])
        for i in range(5)
    ]
    graph = h.make_graph(nodes, "somnus_encoder", graph_inputs, graph_outputs, initializers)
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 14)])
    model.ir_version = 8
    return model


# ── Graph 2: binding ──────────────────────────────────────────────────────────

def build_binding_graph(weights: dict) -> onnx.ModelProto:
    """
    Inputs:
      sp_0 .. sp_4   [1, N_UNITS]  — sparse activations (after CPU topk/mask)
      prev_episode   [1, EPISODE_DIM]
      gru_hidden     [1, 1, GRU_HIDDEN]

    Outputs:
      episode        [1, EPISODE_DIM]
      gru_hidden_out [1, 1, GRU_HIDDEN]
    """
    w = weights

    # GRU weights: burn stores [in, out]; ONNX needs transposed + stacked per gate.
    # Gate order: update(z), reset(r), new(n) — same in burn and ONNX.
    # ONNX W: [1, 3*H, I]; R: [1, 3*H, H]
    gru_W = np.concatenate(
        [w[f"gru_{g}_input_W"].T for g in ["update", "reset", "new"]], axis=0
    )[None].astype(np.float32)  # [1, 3H, I]
    gru_R = np.concatenate(
        [w[f"gru_{g}_hidden_W"].T for g in ["update", "reset", "new"]], axis=0
    )[None].astype(np.float32)  # [1, 3H, H]

    nodes = [
        # Project each sparse region to D_PROJ
    ] + [
        h.make_node("MatMul", [f"sp_{i}", f"W_region_{i}"], [f"proj_{i}"])
        for i in range(5)
    ] + [
        # x = concat([proj_0..4], axis=0) → [5, D_PROJ]
        h.make_node("Concat", [f"proj_{i}" for i in range(5)], ["x"], axis=0),
        # GRU step: prev_episode [1, EP] → reshape [1, 1, EP] (seq-first ONNX GRU)
        h.make_node("Reshape", ["prev_episode", "gru_input_shape"], ["gru_in"]),
        h.make_node("GRU", ["gru_in", "gru_W", "gru_R", "", "", "gru_hidden"],
                    ["gru_Y", "gru_hidden_out"],
                    hidden_size=GRU_HIDDEN,
                    linear_before_reset=1,   # burn reset_after=True (default)
                    direction="forward"),
        # Extract hidden: Y_h [1, 1, GRU_HIDDEN] → [1, GRU_HIDDEN] for matmul
        h.make_node("Reshape", ["gru_hidden_out", "h_2d_shape"], ["h_2d"]),
        # ctx = h @ W_temporal_proj → [1, EPISODE_DIM]
        h.make_node("MatMul", ["h_2d", "W_temporal_proj"], ["ctx"]),
        # ctx_tokens = reshape(ctx, [5, D_PROJ])
        h.make_node("Reshape", ["ctx", "ctx_shape"], ["ctx_tokens"]),
        # x_ext = concat([x, ctx_tokens], axis=0) → [10, D_PROJ]
        h.make_node("Concat", ["x", "ctx_tokens"], ["x_ext"], axis=0),
        # QKV projections
        h.make_node("MatMul", ["x",     "W_q"], ["Q"]),
        h.make_node("MatMul", ["x_ext", "W_k"], ["K"]),
        h.make_node("MatMul", ["x_ext", "W_v"], ["V"]),
        # Multi-head: [seq, D_PROJ] → [seq, N_HEADS, D_HEAD] → swap → [N_HEADS, seq, D_HEAD]
        h.make_node("Reshape",   ["Q", "q_shape"],  ["Qr"]),    # [5, NH, DH]
        h.make_node("Reshape",   ["K", "kv_shape"], ["Kr"]),    # [10, NH, DH]
        h.make_node("Reshape",   ["V", "kv_shape"], ["Vr"]),
        h.make_node("Transpose", ["Qr"], ["Qt"], perm=[1, 0, 2]),   # [NH, 5, DH]
        h.make_node("Transpose", ["Kr"], ["Kt"], perm=[1, 0, 2]),   # [NH, 10, DH]
        h.make_node("Transpose", ["Vr"], ["Vt"], perm=[1, 0, 2]),
        h.make_node("Transpose", ["Kt"], ["Ktt"], perm=[0, 2, 1]),  # [NH, DH, 10]
        h.make_node("MatMul",    ["Qt", "Ktt"], ["scores_raw"]),    # [NH, 5, 10]
        h.make_node("Mul",       ["scores_raw", "attn_scale"], ["scores"]),
        h.make_node("Softmax",   ["scores"], ["attn"], axis=-1),
        h.make_node("MatMul",    ["attn", "Vt"], ["out_heads"]),    # [NH, 5, DH]
        h.make_node("Transpose", ["out_heads"], ["out_heads_t"], perm=[1, 0, 2]),  # [5, NH, DH]
        h.make_node("Reshape",   ["out_heads_t", "out_shape"], ["out_2d"]),        # [5, D_PROJ]
        h.make_node("Reshape",   ["out_2d", "episode_shape"], ["episode"]),        # [1, EP]
    ]

    initializers = [
        # Region projections
    ] + [
        nph.from_array(w[f"W_region_{i}"].astype(np.float32), name=f"W_region_{i}")
        for i in range(5)
    ] + [
        # GRU
        nph.from_array(gru_W,                    name="gru_W"),
        nph.from_array(gru_R,                    name="gru_R"),
        nph.from_array(np.array([1, 1, GRU_INPUT],      dtype=np.int64), name="gru_input_shape"),
        nph.from_array(np.array([1, GRU_HIDDEN],        dtype=np.int64), name="h_2d_shape"),
        nph.from_array(w["W_temporal_proj"].astype(np.float32),          name="W_temporal_proj"),
        nph.from_array(np.array([5, D_PROJ],            dtype=np.int64), name="ctx_shape"),
        # Attention
        nph.from_array(w["W_q"].astype(np.float32),     name="W_q"),
        nph.from_array(w["W_k"].astype(np.float32),     name="W_k"),
        nph.from_array(w["W_v"].astype(np.float32),     name="W_v"),
        nph.from_array(np.array([1.0 / np.sqrt(D_HEAD)], dtype=np.float32), name="attn_scale"),
        nph.from_array(np.array([5,  N_HEADS, D_HEAD],  dtype=np.int64), name="q_shape"),
        nph.from_array(np.array([10, N_HEADS, D_HEAD],  dtype=np.int64), name="kv_shape"),
        nph.from_array(np.array([5,  D_PROJ],           dtype=np.int64), name="out_shape"),
        nph.from_array(np.array([1,  EPISODE_DIM],      dtype=np.int64), name="episode_shape"),
    ]

    graph_inputs = [
        h.make_tensor_value_info(f"sp_{i}", TensorProto.FLOAT, [1, N_UNITS])
        for i in range(5)
    ] + [
        h.make_tensor_value_info("prev_episode",  TensorProto.FLOAT, [1, EPISODE_DIM]),
        h.make_tensor_value_info("gru_hidden",    TensorProto.FLOAT, [1, 1, GRU_HIDDEN]),
    ]
    graph_outputs = [
        h.make_tensor_value_info("episode",        TensorProto.FLOAT, [1, EPISODE_DIM]),
        h.make_tensor_value_info("gru_hidden_out", TensorProto.FLOAT, [1, 1, GRU_HIDDEN]),
    ]
    graph = h.make_graph(nodes, "somnus_binding", graph_inputs, graph_outputs, initializers)
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 14)])
    model.ir_version = 8
    return model


# ── Numpy reference (for validation) ─────────────────────────────────────────

def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

def numpy_encoder(feeds: dict, weights: dict):
    """Encoder forward: raw embeddings → 5 z vectors. Mirrors ONNX encoder graph."""
    w = weights
    cross = np.concatenate([feeds["speech_enc"], feeds["env_enc"],
                            feeds["visual_enc"],  feeds["language_enc"]], axis=-1)
    symbolic_enc = cross @ w["W_cross"]
    encs = [feeds["speech_enc"], feeds["env_enc"], feeds["visual_enc"],
            symbolic_enc, feeds["language_enc"]]
    return [enc @ w[f"W_enc_{i}"] for i, enc in enumerate(encs)]

def numpy_topk_mask(z: np.ndarray, dc: np.ndarray):
    """CPU step: boosted topk selection + sparse mask. Must match burn's select_top_k."""
    boost = 1.0
    target_dc = K_ACTIVE / N_UNITS
    boosted = z + boost * (target_dc - dc)
    topk_idx = np.argpartition(boosted[0], -K_ACTIVE)[-K_ACTIVE:]
    mask = np.zeros_like(z)
    mask[0, topk_idx] = 1.0
    sparse_z = z * mask
    # Duty-cycle update
    new_dc = dc * (1.0 - DUTY_CYCLE_ALPHA)
    new_dc[0, topk_idx] += DUTY_CYCLE_ALPHA
    return sparse_z, new_dc

def numpy_binding(sparse_zs, prev_ep, gru_hidden, weights):
    """Binding area forward: sparse activations → episode. Mirrors ONNX binding graph."""
    w = weights

    projs = [sparse_zs[i] @ w[f"W_region_{i}"] for i in range(5)]  # each [1, D_PROJ]
    x = np.concatenate(projs, axis=0)  # [5, D_PROJ]

    # GRU step: burn does x @ W_input + h @ W_hidden (weights stored [in, out])
    h_prev = gru_hidden[0, 0, :]  # [GRU_HIDDEN]
    x_t    = prev_ep[0]           # [EPISODE_DIM]

    def gru_gate_val(x_in, h_in, gate):
        return x_in @ w[f"gru_{gate}_input_W"] + h_in @ w[f"gru_{gate}_hidden_W"]

    z_gate = sigmoid(gru_gate_val(x_t, h_prev, "update"))
    r_gate = sigmoid(gru_gate_val(x_t, h_prev, "reset"))
    # reset_after=True: n_gate = tanh(x@W_n_in + r * (h@W_n_hid))
    n_gate = np.tanh(
        x_t @ w["gru_new_input_W"] + r_gate * (h_prev @ w["gru_new_hidden_W"])
    )
    h_new = (1.0 - z_gate) * n_gate + z_gate * h_prev

    ctx        = h_new @ w["W_temporal_proj"]       # [EPISODE_DIM]
    ctx_tokens = ctx.reshape(5, D_PROJ)              # [5, D_PROJ]
    x_ext      = np.concatenate([x, ctx_tokens], axis=0)  # [10, D_PROJ]

    Q = x @ w["W_q"]        # [5, D_PROJ]
    K = x_ext @ w["W_k"]    # [10, D_PROJ]
    V = x_ext @ w["W_v"]

    Q3 = Q.reshape(5,  N_HEADS, D_HEAD).transpose(1, 0, 2)   # [NH, 5,  DH]
    K3 = K.reshape(10, N_HEADS, D_HEAD).transpose(1, 0, 2)   # [NH, 10, DH]
    V3 = V.reshape(10, N_HEADS, D_HEAD).transpose(1, 0, 2)

    scores = Q3 @ K3.transpose(0, 2, 1) / np.sqrt(D_HEAD)    # [NH, 5, 10]
    attn   = softmax(scores, axis=-1)
    out3   = attn @ V3                                        # [NH, 5, DH]
    out    = out3.transpose(1, 0, 2).reshape(5, D_PROJ)       # [5, D_PROJ]
    episode = out.reshape(1, -1)                               # [1, EPISODE_DIM]

    return episode, h_new.reshape(1, 1, -1)


# ── Weight loading ────────────────────────────────────────────────────────────

def load_weights_from_dir(weights_dir: str) -> dict:
    """Load weight bundle written by PerceptionProcessor::export_weights_for_onnx()."""
    with open(os.path.join(weights_dir, "weights_manifest.json")) as f:
        manifest = json.load(f)
    weights = {}
    for entry in manifest:
        data = np.fromfile(os.path.join(weights_dir, entry["file"]), dtype=np.float32)
        weights[entry["name"]] = data.reshape(entry["shape"])
    return weights

def make_random_weights(seed: int = 42) -> dict:
    """Random weights in correct shapes for validation."""
    rng = np.random.default_rng(seed)
    def randn(*shape): return rng.normal(0, 0.02, shape).astype(np.float32)
    w = {}
    for i, d in enumerate(REGION_DIMS):
        w[f"W_enc_{i}"] = randn(d, N_UNITS)
    w["W_cross"] = randn(CROSS_MODAL_DIM, 384)
    for i in range(5):
        w[f"W_region_{i}"] = randn(N_UNITS, D_PROJ)
    w["W_q"] = randn(D_PROJ, D_PROJ)
    w["W_k"] = randn(D_PROJ, D_PROJ)
    w["W_v"] = randn(D_PROJ, D_PROJ)
    w["W_temporal_proj"] = randn(GRU_HIDDEN, EPISODE_DIM)
    for gate in ["update", "reset", "new"]:
        w[f"gru_{gate}_input_W"]  = randn(GRU_INPUT, GRU_HIDDEN)
        w[f"gru_{gate}_hidden_W"] = randn(GRU_HIDDEN, GRU_HIDDEN)
    return w

def make_random_feeds(rng=None) -> dict:
    if rng is None: rng = np.random.default_rng(7)
    def r(*s): return rng.normal(0, 1, s).astype(np.float32)
    target = K_ACTIVE / N_UNITS
    return {
        "speech_enc":   r(1, 384),
        "env_enc":      r(1, 512),
        "visual_enc":   r(1, 384),
        "language_enc": r(1, 896),
        "prev_episode": np.zeros((1, EPISODE_DIM), np.float32),
        "gru_hidden":   np.zeros((1, 1, GRU_HIDDEN), np.float32),
        # Non-uniform dc: exercises the boost term (uniform dc → boost=0, hides errors)
        "dc": [rng.uniform(0.005, 0.05, (1, N_UNITS)).astype(np.float32)
               for _ in range(5)],
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights-dir", help="Directory with weight bundle from Rust export")
    ap.add_argument("--out-dir",     default=".", help="Output directory for ONNX files")
    ap.add_argument("--validate-numpy", action="store_true",
                    help="Validate ONNX graphs vs numpy reference using random weights")
    ap.add_argument("--validate-migraphx", action="store_true",
                    help="Also run validation on MIGraphX EP (requires AMD GPU)")
    args = ap.parse_args()

    if args.weights_dir:
        print(f"Loading weights from {args.weights_dir}")
        weights = load_weights_from_dir(args.weights_dir)
    elif args.validate_numpy:
        print("Using random weights for numpy validation")
        weights = make_random_weights()
    else:
        ap.error("Provide --weights-dir or --validate-numpy")

    print("Building encoder graph...")
    enc_model = build_encoder_graph(weights)
    onnx.checker.check_model(enc_model)
    enc_path = os.path.join(args.out_dir, "perception_encoder.onnx")
    onnx.save(enc_model, enc_path)
    print(f"  Saved: {enc_path}")

    print("Building binding graph...")
    bind_model = build_binding_graph(weights)
    onnx.checker.check_model(bind_model)
    bind_path = os.path.join(args.out_dir, "perception_binding.onnx")
    onnx.save(bind_model, bind_path)
    print(f"  Saved: {bind_path}")

    if not args.validate_numpy:
        print("Done.")
        return

    print("\nValidating ONNX vs numpy reference (non-uniform dc, exercises boost term)...")
    feeds = make_random_feeds()
    enc_feeds = {k: feeds[k] for k in ["speech_enc","env_enc","visual_enc","language_enc"]}

    # Numpy reference
    z_np  = numpy_encoder(enc_feeds, weights)
    sps_np, dcs_np = [], []
    for i, z in enumerate(z_np):
        sp, dc = numpy_topk_mask(z, feeds["dc"][i])
        sps_np.append(sp); dcs_np.append(dc)
    ep_np, h_np = numpy_binding(sps_np, feeds["prev_episode"], feeds["gru_hidden"], weights)

    def validate_ep(name, sess_enc, sess_bind, atol=1e-3):
        z_out = sess_enc.run(None, enc_feeds)
        # CPU topk (matching numpy path for this comparison)
        sps = []
        for i, z in enumerate(z_out):
            sp, _ = numpy_topk_mask(z, feeds["dc"][i])
            sps.append(sp)
        bind_feeds = {f"sp_{i}": sps[i] for i in range(5)}
        bind_feeds["prev_episode"] = feeds["prev_episode"]
        bind_feeds["gru_hidden"]   = feeds["gru_hidden"]
        ep_ort, h_ort = sess_bind.run(None, bind_feeds)
        diff_ep = float(np.max(np.abs(ep_np - ep_ort)))
        diff_h  = float(np.max(np.abs(h_np  - h_ort)))
        ok = diff_ep < atol and diff_h < atol
        print(f"  numpy vs {name}: ep={diff_ep:.2e} gru_h={diff_h:.2e} {'PASS' if ok else 'FAIL'}")
        return ok

    cpu_ep   = [ort.InferenceSession(p, providers=["CPUExecutionProvider"])
                for p in [enc_path, bind_path]]
    validate_ep("CPU EP", *cpu_ep)

    if args.validate_migraphx:
        mgx_ep = [ort.InferenceSession(p, providers=["MIGraphXExecutionProvider","CPUExecutionProvider"])
                  for p in [enc_path, bind_path]]
        validate_ep("MIGraphX EP", *mgx_ep, atol=1e-3)

    print("Done.")


if __name__ == "__main__":
    main()

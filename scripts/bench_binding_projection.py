#!/usr/bin/env python3
"""
Isolate the cost of the binding graph's 5 dense region-projection matmuls
(`sp_i [1, n_units] @ W_region_i [n_units, D_PROJ]`) from everything else in
`perception_binding.onnx` (GRU + attention), at real campaign scale.

Why: a proposed optimization (replace the dense projection with a
Gather-on-active-indices + small matmul, mirroring gaps.md §25's already-
measured 11.4x win on the burn/CPU path) needs to know how much of the
binding graph's wall-clock the projection actually is *on this GPU/MIGraphX
path* before the rewrite is worth doing — §25's number doesn't transfer
(that was a CPU measurement; GPU matmul throughput can absorb a lot of
"wasted" FLOPs a CPU implementation could not, per ADR 0038's own invalidation
clause).

Method: build three ONNX graphs from identical random weights —
  "full"      — scripts/export_perception_graph.py's real (pre-rewrite shape)
                build_binding_graph, dense sp_i @ W_region_i projection
  "gather"    — the real (post-rewrite) build_binding_graph: Gather-on-active-
                indices + small matmul, at the campaign's actual k_active
  "projected" — same graph, but the 5 projection matmuls are removed entirely
                and proj_0..4 [1, D_PROJ] are taken as direct graph inputs
                instead — GRU + attention + marshalling only, a lower bound
All three graphs are otherwise byte-for-byte identical (GRU + attention nodes
unchanged). Time all three with the same execution provider: full vs.
projected isolates the *dense* projection's cost (what the rewrite replaces);
gather vs. projected isolates the *actual* rewrite's residual cost; full vs.
gather is the headline win number.

Usage:
  python3 scripts/bench_binding_projection.py [--n-units 9660] [--reps 2000]
                                               [--provider migraphx|cpu]
"""

import argparse
import time

import numpy as np
import onnx
import onnx.helper as h
import onnx.numpy_helper as nph
import onnxruntime as ort
from onnx import TensorProto

import export_perception_graph as epg

D_PROJ = epg.D_PROJ
N_HEADS = epg.N_HEADS
D_HEAD = epg.D_HEAD
EPISODE_DIM = epg.EPISODE_DIM
GRU_INPUT = epg.GRU_INPUT
GRU_HIDDEN = epg.GRU_HIDDEN


def make_random_binding_weights(n_units: int, seed: int = 42) -> dict:
    """Binding-graph-relevant weights only, at a chosen n_units (campaign scale)."""
    rng = np.random.default_rng(seed)

    def randn(*shape):
        return rng.normal(0, 0.02, shape).astype(np.float32)

    w = {}
    for i in range(5):
        w[f"W_region_{i}"] = randn(n_units, D_PROJ)
    w["W_q"] = randn(D_PROJ, D_PROJ)
    w["W_k"] = randn(D_PROJ, D_PROJ)
    w["W_v"] = randn(D_PROJ, D_PROJ)
    w["W_temporal_proj"] = randn(GRU_HIDDEN, EPISODE_DIM)
    for gate in ["update", "reset", "new"]:
        w[f"gru_{gate}_input_W"] = randn(GRU_INPUT, GRU_HIDDEN)
        w[f"gru_{gate}_hidden_W"] = randn(GRU_HIDDEN, GRU_HIDDEN)
    return w


def build_binding_graph_dense(weights: dict) -> onnx.ModelProto:
    """The pre-rewrite dense-projection binding graph (sp_i [1, n_units] @
    W_region_i [n_units, D_PROJ]) — kept here as the "full" baseline arm now that
    export_perception_graph.build_binding_graph has been rewritten to the
    sparse-gather form. Otherwise identical to the current production graph.
    """
    w = weights
    _region_dims, n_units, _cross_modal_dim = epg.infer_dims(weights)

    gru_W = np.concatenate(
        [w[f"gru_{g}_input_W"].T for g in ["update", "reset", "new"]], axis=0
    )[None].astype(np.float32)
    gru_R = np.concatenate(
        [w[f"gru_{g}_hidden_W"].T for g in ["update", "reset", "new"]], axis=0
    )[None].astype(np.float32)

    nodes = [
        h.make_node("MatMul", [f"sp_{i}", f"W_region_{i}"], [f"proj_{i}"]) for i in range(5)
    ] + [
        h.make_node("Concat", [f"proj_{i}" for i in range(5)], ["x"], axis=0),
        h.make_node("Reshape", ["prev_episode", "gru_input_shape"], ["gru_in"]),
        h.make_node(
            "GRU",
            ["gru_in", "gru_W", "gru_R", "", "", "gru_hidden"],
            ["gru_Y", "gru_hidden_out"],
            hidden_size=GRU_HIDDEN,
            linear_before_reset=1,
            direction="forward",
        ),
        h.make_node("Reshape", ["gru_hidden_out", "h_2d_shape"], ["h_2d"]),
        h.make_node("MatMul", ["h_2d", "W_temporal_proj"], ["ctx"]),
        h.make_node("Reshape", ["ctx", "ctx_shape"], ["ctx_tokens"]),
        h.make_node("Concat", ["x", "ctx_tokens"], ["x_ext"], axis=0),
        h.make_node("MatMul", ["x", "W_q"], ["Q"]),
        h.make_node("MatMul", ["x_ext", "W_k"], ["K"]),
        h.make_node("MatMul", ["x_ext", "W_v"], ["V"]),
        h.make_node("Reshape", ["Q", "q_shape"], ["Qr"]),
        h.make_node("Reshape", ["K", "kv_shape"], ["Kr"]),
        h.make_node("Reshape", ["V", "kv_shape"], ["Vr"]),
        h.make_node("Transpose", ["Qr"], ["Qt"], perm=[1, 0, 2]),
        h.make_node("Transpose", ["Kr"], ["Kt"], perm=[1, 0, 2]),
        h.make_node("Transpose", ["Vr"], ["Vt"], perm=[1, 0, 2]),
        h.make_node("Transpose", ["Kt"], ["Ktt"], perm=[0, 2, 1]),
        h.make_node("MatMul", ["Qt", "Ktt"], ["scores_raw"]),
        h.make_node("Mul", ["scores_raw", "attn_scale"], ["scores"]),
        h.make_node("Softmax", ["scores"], ["attn"], axis=-1),
        h.make_node("MatMul", ["attn", "Vt"], ["out_heads"]),
        h.make_node("Transpose", ["out_heads"], ["out_heads_t"], perm=[1, 0, 2]),
        h.make_node("Reshape", ["out_heads_t", "out_shape"], ["out_2d"]),
        h.make_node("Reshape", ["out_2d", "episode_shape"], ["episode"]),
    ]

    initializers = [
        nph.from_array(w[f"W_region_{i}"].astype(np.float32), name=f"W_region_{i}")
        for i in range(5)
    ] + [
        nph.from_array(gru_W, name="gru_W"),
        nph.from_array(gru_R, name="gru_R"),
        nph.from_array(np.array([1, 1, GRU_INPUT], dtype=np.int64), name="gru_input_shape"),
        nph.from_array(np.array([1, GRU_HIDDEN], dtype=np.int64), name="h_2d_shape"),
        nph.from_array(w["W_temporal_proj"].astype(np.float32), name="W_temporal_proj"),
        nph.from_array(np.array([5, D_PROJ], dtype=np.int64), name="ctx_shape"),
        nph.from_array(w["W_q"].astype(np.float32), name="W_q"),
        nph.from_array(w["W_k"].astype(np.float32), name="W_k"),
        nph.from_array(w["W_v"].astype(np.float32), name="W_v"),
        nph.from_array(np.array([1.0 / np.sqrt(D_HEAD)], dtype=np.float32), name="attn_scale"),
        nph.from_array(np.array([5, N_HEADS, D_HEAD], dtype=np.int64), name="q_shape"),
        nph.from_array(np.array([10, N_HEADS, D_HEAD], dtype=np.int64), name="kv_shape"),
        nph.from_array(np.array([5, D_PROJ], dtype=np.int64), name="out_shape"),
        nph.from_array(np.array([1, EPISODE_DIM], dtype=np.int64), name="episode_shape"),
    ]

    graph_inputs = [
        h.make_tensor_value_info(f"sp_{i}", TensorProto.FLOAT, [1, n_units[i]]) for i in range(5)
    ] + [
        h.make_tensor_value_info("prev_episode", TensorProto.FLOAT, [1, EPISODE_DIM]),
        h.make_tensor_value_info("gru_hidden", TensorProto.FLOAT, [1, 1, GRU_HIDDEN]),
    ]
    graph_outputs = [
        h.make_tensor_value_info("episode", TensorProto.FLOAT, [1, EPISODE_DIM]),
        h.make_tensor_value_info("gru_hidden_out", TensorProto.FLOAT, [1, 1, GRU_HIDDEN]),
    ]
    graph = h.make_graph(nodes, "somnus_binding_dense", graph_inputs, graph_outputs, initializers)
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 14)])
    model.ir_version = 8
    return model


def build_binding_graph_projected(weights: dict) -> onnx.ModelProto:
    """Same as export_perception_graph.build_binding_graph, minus the 5 region
    projection matmuls — takes proj_0..4 [1, D_PROJ] directly as graph inputs.
    Isolates GRU + attention + marshalling cost from the projection matmul.
    """
    w = weights
    gru_W = np.concatenate(
        [w[f"gru_{g}_input_W"].T for g in ["update", "reset", "new"]], axis=0
    )[None].astype(np.float32)
    gru_R = np.concatenate(
        [w[f"gru_{g}_hidden_W"].T for g in ["update", "reset", "new"]], axis=0
    )[None].astype(np.float32)

    nodes = [
        h.make_node("Concat", [f"proj_{i}" for i in range(5)], ["x"], axis=0),
        h.make_node("Reshape", ["prev_episode", "gru_input_shape"], ["gru_in"]),
        h.make_node(
            "GRU",
            ["gru_in", "gru_W", "gru_R", "", "", "gru_hidden"],
            ["gru_Y", "gru_hidden_out"],
            hidden_size=GRU_HIDDEN,
            linear_before_reset=1,
            direction="forward",
        ),
        h.make_node("Reshape", ["gru_hidden_out", "h_2d_shape"], ["h_2d"]),
        h.make_node("MatMul", ["h_2d", "W_temporal_proj"], ["ctx"]),
        h.make_node("Reshape", ["ctx", "ctx_shape"], ["ctx_tokens"]),
        h.make_node("Concat", ["x", "ctx_tokens"], ["x_ext"], axis=0),
        h.make_node("MatMul", ["x", "W_q"], ["Q"]),
        h.make_node("MatMul", ["x_ext", "W_k"], ["K"]),
        h.make_node("MatMul", ["x_ext", "W_v"], ["V"]),
        h.make_node("Reshape", ["Q", "q_shape"], ["Qr"]),
        h.make_node("Reshape", ["K", "kv_shape"], ["Kr"]),
        h.make_node("Reshape", ["V", "kv_shape"], ["Vr"]),
        h.make_node("Transpose", ["Qr"], ["Qt"], perm=[1, 0, 2]),
        h.make_node("Transpose", ["Kr"], ["Kt"], perm=[1, 0, 2]),
        h.make_node("Transpose", ["Vr"], ["Vt"], perm=[1, 0, 2]),
        h.make_node("Transpose", ["Kt"], ["Ktt"], perm=[0, 2, 1]),
        h.make_node("MatMul", ["Qt", "Ktt"], ["scores_raw"]),
        h.make_node("Mul", ["scores_raw", "attn_scale"], ["scores"]),
        h.make_node("Softmax", ["scores"], ["attn"], axis=-1),
        h.make_node("MatMul", ["attn", "Vt"], ["out_heads"]),
        h.make_node("Transpose", ["out_heads"], ["out_heads_t"], perm=[1, 0, 2]),
        h.make_node("Reshape", ["out_heads_t", "out_shape"], ["out_2d"]),
        h.make_node("Reshape", ["out_2d", "episode_shape"], ["episode"]),
    ]

    initializers = [
        nph.from_array(gru_W, name="gru_W"),
        nph.from_array(gru_R, name="gru_R"),
        nph.from_array(np.array([1, 1, GRU_INPUT], dtype=np.int64), name="gru_input_shape"),
        nph.from_array(np.array([1, GRU_HIDDEN], dtype=np.int64), name="h_2d_shape"),
        nph.from_array(w["W_temporal_proj"].astype(np.float32), name="W_temporal_proj"),
        nph.from_array(np.array([5, D_PROJ], dtype=np.int64), name="ctx_shape"),
        nph.from_array(w["W_q"].astype(np.float32), name="W_q"),
        nph.from_array(w["W_k"].astype(np.float32), name="W_k"),
        nph.from_array(w["W_v"].astype(np.float32), name="W_v"),
        nph.from_array(np.array([1.0 / np.sqrt(D_HEAD)], dtype=np.float32), name="attn_scale"),
        nph.from_array(np.array([5, N_HEADS, D_HEAD], dtype=np.int64), name="q_shape"),
        nph.from_array(np.array([10, N_HEADS, D_HEAD], dtype=np.int64), name="kv_shape"),
        nph.from_array(np.array([5, D_PROJ], dtype=np.int64), name="out_shape"),
        nph.from_array(np.array([1, EPISODE_DIM], dtype=np.int64), name="episode_shape"),
    ]

    graph_inputs = [
        h.make_tensor_value_info(f"proj_{i}", TensorProto.FLOAT, [1, D_PROJ]) for i in range(5)
    ] + [
        h.make_tensor_value_info("prev_episode", TensorProto.FLOAT, [1, EPISODE_DIM]),
        h.make_tensor_value_info("gru_hidden", TensorProto.FLOAT, [1, 1, GRU_HIDDEN]),
    ]
    graph_outputs = [
        h.make_tensor_value_info("episode", TensorProto.FLOAT, [1, EPISODE_DIM]),
        h.make_tensor_value_info("gru_hidden_out", TensorProto.FLOAT, [1, 1, GRU_HIDDEN]),
    ]
    graph = h.make_graph(nodes, "somnus_binding_projected", graph_inputs, graph_outputs, initializers)
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 14)])
    model.ir_version = 8
    return model


def bench_session(sess, feeds, reps, warmup=50):
    for _ in range(warmup):
        sess.run(None, feeds)
    t0 = time.perf_counter()
    for _ in range(reps):
        sess.run(None, feeds)
    total = time.perf_counter() - t0
    return total / reps * 1000.0  # ms/call


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-units", type=int, default=9660, help="Region size (campaign scale)")
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--provider", choices=["migraphx", "cpu"], default="migraphx")
    args = ap.parse_args()

    providers = (
        ["MIGraphXExecutionProvider", "CPUExecutionProvider"]
        if args.provider == "migraphx"
        else ["CPUExecutionProvider"]
    )

    # scaled_k mirrors crates/somnus-perception/src/region.rs::scaled_k exactly (preserves the
    # baseline K_ACTIVE/N_UNITS ratio) — this is the only place in this script that computes
    # k rather than reading it from a manifest, which is fine here since it's a synthetic
    # benchmark harness, not a code path that feeds a real ONNX graph's shape from Rust.
    k_active_val = max(1, round(args.n_units * epg.BASELINE_K_ACTIVE / epg.BASELINE_N_UNITS))
    k_active = [k_active_val] * 5

    print(
        f"n_units={args.n_units}  k_active={k_active_val}  "
        f"provider={args.provider}  reps={args.reps}"
    )
    w = make_random_binding_weights(args.n_units)

    # build_binding_graph calls infer_dims(weights), which needs W_enc_i/W_cross purely to
    # derive n_units/region_dims — dummy shapes consistent with infer_dims' own consistency
    # check (cross_modal_dim == region_dims[0]+[1]+[2]+[4]) are enough; their values are unused.
    region_dims = epg.BASELINE_REGION_DIMS
    cross_modal_dim = region_dims[0] + region_dims[1] + region_dims[2] + region_dims[4]
    dummy = {
        **w,
        **{f"W_enc_{i}": np.zeros((region_dims[i], args.n_units), np.float32) for i in range(5)},
        "W_cross": np.zeros((cross_modal_dim, 384), np.float32),
    }
    full_model = build_binding_graph_dense(dummy)
    onnx.checker.check_model(full_model)

    gather_model = epg.build_binding_graph(dummy, k_active)
    onnx.checker.check_model(gather_model)

    proj_model = build_binding_graph_projected(w)
    onnx.checker.check_model(proj_model)

    rng = np.random.default_rng(7)
    sp_feeds = {f"sp_{i}": rng.normal(0, 1, (1, args.n_units)).astype(np.float32) for i in range(5)}
    gather_feeds = {}
    for i in range(5):
        idx, val = epg.dense_to_active(sp_feeds[f"sp_{i}"], k_active_val)
        gather_feeds[f"active_idx_{i}"] = idx
        gather_feeds[f"active_val_{i}"] = val
    proj_feeds = {f"proj_{i}": rng.normal(0, 1, (1, D_PROJ)).astype(np.float32) for i in range(5)}
    common_feeds = {
        "prev_episode": np.zeros((1, EPISODE_DIM), np.float32),
        "gru_hidden": np.zeros((1, 1, GRU_HIDDEN), np.float32),
    }

    full_sess = ort.InferenceSession(full_model.SerializeToString(), providers=providers)
    gather_sess = ort.InferenceSession(gather_model.SerializeToString(), providers=providers)
    proj_sess = ort.InferenceSession(proj_model.SerializeToString(), providers=providers)
    print(f"full graph EPs in use: {full_sess.get_providers()}")

    full_ms = bench_session(full_sess, {**sp_feeds, **common_feeds}, args.reps)
    gather_ms = bench_session(gather_sess, {**gather_feeds, **common_feeds}, args.reps)
    proj_ms = bench_session(proj_sess, {**proj_feeds, **common_feeds}, args.reps)

    delta_dense = full_ms - proj_ms
    delta_gather = gather_ms - proj_ms
    pct_dense = 100.0 * delta_dense / full_ms if full_ms > 0 else float("nan")
    win = full_ms - gather_ms
    win_pct = 100.0 * win / full_ms if full_ms > 0 else float("nan")
    print(f"full (dense projection + GRU + attention):   {full_ms:.4f} ms/call")
    print(f"gather (sparse-gather projection + GRU/attn): {gather_ms:.4f} ms/call")
    print(f"projected (GRU + attention only, lower bound): {proj_ms:.4f} ms/call")
    print(f"dense projection's share of full:             {delta_dense:.4f} ms/call ({pct_dense:.1f}% of full)")
    print(f"gather rewrite's residual over the lower bound: {delta_gather:.4f} ms/call")
    print(f"HEADLINE: full - gather (measured rewrite win): {win:.4f} ms/call ({win_pct:.1f}% of full)")


if __name__ == "__main__":
    main()

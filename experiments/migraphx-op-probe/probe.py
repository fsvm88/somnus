"""
MIGraphX op-support probe for the GPU unification plan.

Tests GRU, TopK, Softmax+MatMul (cross-attention), and ScatterElements
on the MIGraphX EP. Each probe:
  1. builds a minimal ONNX graph for the op
  2. runs it on CPU EP (reference)
  3. runs it on MIGraphX EP
  4. reports max absolute diff and pass/fail

Run: python3 experiments/migraphx-op-probe/probe.py
"""

import numpy as np
import onnx
import onnx.helper as h
import onnx.numpy_helper as nph
import onnxruntime as ort
from onnx import TensorProto
import sys

# Dimensions matching somnus-perception at current scale
GRU_INPUT_SIZE  = 640   # EPISODE_DIM — GRU sequence input
GRU_HIDDEN_SIZE = 256   # GRU_HIDDEN
N_UNITS         = 1024  # k-WTA units per region
K_ACTIVE        = 20    # top-k
D_PROJ          = 128   # binding projection dim
N_HEADS         = 4
D_HEAD          = D_PROJ // N_HEADS  # 32
SEQ_LEN         = 5     # regions (Q tokens)
SEQ_K           = 10    # regions + temporal context (K/V tokens)

TOL = 1e-3              # max abs diff CPU vs MIGraphX

MIGRAPHX_EP = "MIGraphXExecutionProvider"
CPU_EP      = "CPUExecutionProvider"

results = []

def run_both(model, feeds, atol=TOL):
    """Run model on CPU and MIGraphX; return (cpu_out, mgx_out, max_diff)."""
    sess_cpu = ort.InferenceSession(
        model.SerializeToString(), providers=[CPU_EP]
    )
    sess_mgx = ort.InferenceSession(
        model.SerializeToString(), providers=[MIGRAPHX_EP, CPU_EP]
    )
    out_cpu = sess_cpu.run(None, feeds)
    out_mgx = sess_mgx.run(None, feeds)
    max_diffs = []
    for c, m in zip(out_cpu, out_mgx):
        max_diffs.append(float(np.max(np.abs(c.astype(np.float32) - m.astype(np.float32)))))
    return out_cpu, out_mgx, max_diffs


def probe(name, build_fn, atol=TOL):
    print(f"\n{'='*60}")
    print(f"PROBE: {name}")
    try:
        model, feeds = build_fn()
        onnx.checker.check_model(model)
        cpu_out, mgx_out, diffs = run_both(model, feeds, atol)
        passed = all(d <= atol for d in diffs)
        status = "PASS" if passed else "FAIL"
        print(f"  max abs diff: {[f'{d:.2e}' for d in diffs]}")
        print(f"  => {status} (atol={atol})")
        results.append((name, status, diffs))
    except Exception as e:
        print(f"  => ERROR: {e}")
        results.append((name, "ERROR", str(e)))


# ── 1. GRU (one step, matches burn GruConfig) ─────────────────────────────────
def build_gru():
    """
    ONNX GRU op, seq=1, matches burn's Gru (no bias, linear_before_reset=0).
    Input shape: [seq=1, batch=1, input_size=GRU_INPUT_SIZE]
    H_0 shape:   [num_directions=1, batch=1, hidden_size=GRU_HIDDEN_SIZE]
    """
    W_size = 3 * GRU_HIDDEN_SIZE * GRU_INPUT_SIZE   # [1, 3*H, I]
    R_size = 3 * GRU_HIDDEN_SIZE * GRU_HIDDEN_SIZE  # [1, 3*H, H]

    rng = np.random.default_rng(42)
    W = rng.normal(0, 0.01, (1, 3 * GRU_HIDDEN_SIZE, GRU_INPUT_SIZE)).astype(np.float32)
    R = rng.normal(0, 0.01, (1, 3 * GRU_HIDDEN_SIZE, GRU_HIDDEN_SIZE)).astype(np.float32)

    X   = rng.normal(0, 1, (1, 1, GRU_INPUT_SIZE)).astype(np.float32)
    H_0 = rng.normal(0, 0.1, (1, 1, GRU_HIDDEN_SIZE)).astype(np.float32)

    W_init = nph.from_array(W, name="gru_W")
    R_init = nph.from_array(R, name="gru_R")

    node = h.make_node(
        "GRU",
        inputs=["X", "gru_W", "gru_R", "", "seq_lens", "H_0"],
        outputs=["Y", "Y_h"],
        hidden_size=GRU_HIDDEN_SIZE,
        linear_before_reset=0,
        direction="forward",
    )
    graph = h.make_graph(
        [node],
        "gru_test",
        inputs=[
            h.make_tensor_value_info("X",       TensorProto.FLOAT, [1, 1, GRU_INPUT_SIZE]),
            h.make_tensor_value_info("seq_lens", TensorProto.INT32,  [1]),
            h.make_tensor_value_info("H_0",     TensorProto.FLOAT, [1, 1, GRU_HIDDEN_SIZE]),
        ],
        outputs=[
            h.make_tensor_value_info("Y",   TensorProto.FLOAT, [1, 1, 1, GRU_HIDDEN_SIZE]),
            h.make_tensor_value_info("Y_h", TensorProto.FLOAT, [1, 1, GRU_HIDDEN_SIZE]),
        ],
        initializer=[W_init, R_init],
    )
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 14)])
    feeds = {"X": X, "seq_lens": np.array([1], dtype=np.int32), "H_0": H_0}
    return model, feeds


# ── 2. TopK (largest-first, axis=-1) ─────────────────────────────────────────
def build_topk():
    """
    ONNX TopK: returns top K_ACTIVE values and indices from a [1, N_UNITS] tensor.
    """
    rng = np.random.default_rng(7)
    X = rng.normal(0, 1, (1, N_UNITS)).astype(np.float32)
    k_val = np.array([K_ACTIVE], dtype=np.int64)

    k_init = nph.from_array(k_val, name="k_const")
    node = h.make_node(
        "TopK",
        inputs=["X", "k_const"],
        outputs=["Values", "Indices"],
        axis=-1,
        largest=1,
        sorted=0,  # don't need sorted output
    )
    graph = h.make_graph(
        [node],
        "topk_test",
        inputs=[h.make_tensor_value_info("X", TensorProto.FLOAT, [1, N_UNITS])],
        outputs=[
            h.make_tensor_value_info("Values",  TensorProto.FLOAT, [1, K_ACTIVE]),
            h.make_tensor_value_info("Indices", TensorProto.INT64,  [1, K_ACTIVE]),
        ],
        initializer=[k_init],
    )
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 14)])
    feeds = {"X": X}
    return model, feeds


# ── 3. Multi-head cross-attention (matmul + reshape + softmax pattern) ────────
def build_attention():
    """
    Simplified cross-attention matching binding.rs:
      Q: [SEQ_LEN, D_PROJ] @ W_q → [SEQ_LEN, D_PROJ]
      K: [SEQ_K, D_PROJ]  @ W_k → [SEQ_K, D_PROJ]
      reshape to [N_HEADS, seq, D_HEAD], scores = Q @ K^T / sqrt(D_HEAD)
      attn = softmax(scores), out = attn @ V, flatten
    """
    rng = np.random.default_rng(13)
    X_q = rng.normal(0, 1, (SEQ_LEN, D_PROJ)).astype(np.float32)
    X_kv = rng.normal(0, 1, (SEQ_K, D_PROJ)).astype(np.float32)
    W_q = rng.normal(0, 0.01, (D_PROJ, D_PROJ)).astype(np.float32)
    W_k = rng.normal(0, 0.01, (D_PROJ, D_PROJ)).astype(np.float32)
    W_v = rng.normal(0, 0.01, (D_PROJ, D_PROJ)).astype(np.float32)
    scale = np.array(1.0 / np.sqrt(D_HEAD), dtype=np.float32)

    wq_init = nph.from_array(W_q, name="W_q")
    wk_init = nph.from_array(W_k, name="W_k")
    wv_init = nph.from_array(W_v, name="W_v")
    scale_init = nph.from_array(scale, name="scale")

    shape_q  = nph.from_array(np.array([N_HEADS, SEQ_LEN, D_HEAD], dtype=np.int64), name="shape_q")
    shape_kv = nph.from_array(np.array([N_HEADS, SEQ_K,   D_HEAD], dtype=np.int64), name="shape_kv")
    shape_out= nph.from_array(np.array([SEQ_LEN, D_PROJ], dtype=np.int64), name="shape_out")

    nodes = [
        h.make_node("MatMul", ["X_q",  "W_q"], ["Q"]),
        h.make_node("MatMul", ["X_kv", "W_k"], ["K"]),
        h.make_node("MatMul", ["X_kv", "W_v"], ["V"]),
        # reshape to multi-head: [seq, D_PROJ] → [N_HEADS, seq, D_HEAD]
        # ONNX Reshape: [seq, D_PROJ] — transpose first to [D_PROJ, seq] is wrong.
        # Per binding.rs: reshape [seq, N_HEADS, D_HEAD] then swap_dims(0,1)
        h.make_node("Reshape", ["Q",  "shape_q"],  ["Q_r"]),   # [N_HEADS, SEQ_LEN, D_HEAD]
        h.make_node("Reshape", ["K",  "shape_kv"], ["K_r"]),   # [N_HEADS, SEQ_K, D_HEAD]
        h.make_node("Reshape", ["V",  "shape_kv"], ["V_r"]),
        # scores = Q_r @ K_r^T * scale: [N_HEADS, SEQ_LEN, SEQ_K]
        h.make_node("Transpose", ["K_r"], ["K_t"], perm=[0, 2, 1]),
        h.make_node("MatMul", ["Q_r", "K_t"], ["scores_raw"]),
        h.make_node("Mul", ["scores_raw", "scale"], ["scores"]),
        h.make_node("Softmax", ["scores"], ["attn"], axis=-1),
        # out = attn @ V_r: [N_HEADS, SEQ_LEN, D_HEAD]
        h.make_node("MatMul", ["attn", "V_r"], ["out_heads"]),
        # merge heads: [N_HEADS, SEQ_LEN, D_HEAD] → [SEQ_LEN, D_PROJ]
        h.make_node("Transpose", ["out_heads"], ["out_t"], perm=[1, 0, 2]),
        h.make_node("Reshape", ["out_t", "shape_out"], ["out"]),
    ]
    graph = h.make_graph(
        nodes, "attn_test",
        inputs=[
            h.make_tensor_value_info("X_q",  TensorProto.FLOAT, [SEQ_LEN, D_PROJ]),
            h.make_tensor_value_info("X_kv", TensorProto.FLOAT, [SEQ_K,   D_PROJ]),
        ],
        outputs=[h.make_tensor_value_info("out", TensorProto.FLOAT, [SEQ_LEN, D_PROJ])],
        initializer=[wq_init, wk_init, wv_init, scale_init, shape_q, shape_kv, shape_out],
    )
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 14)])
    feeds = {"X_q": X_q, "X_kv": X_kv}
    return model, feeds


# ── 4. Duty-cycle boost + TopK + scatter mask (full k-WTA graph) ──────────────
def build_kwta_full():
    """
    Full k-WTA inference step:
      z = X @ W                              [1, N_UNITS]
      boosted = z + BOOST * (target_dc - dc) [1, N_UNITS]
      _, top_idx = TopK(boosted, K_ACTIVE)   [1, K_ACTIVE]
      mask = ScatterElements(zeros, top_idx, ones)  [1, N_UNITS]
      sparse = z * mask                      [1, N_UNITS]
    """
    rng = np.random.default_rng(99)
    n_in = 384
    W_val = rng.normal(0, 0.01, (n_in, N_UNITS)).astype(np.float32)
    dc_val = rng.uniform(0.01, 0.03, (1, N_UNITS)).astype(np.float32)
    target_dc = np.float32(K_ACTIVE / N_UNITS)
    boost_strength = np.float32(1.0)

    k_val = np.array([K_ACTIVE], dtype=np.int64)
    ones_val = np.ones((1, K_ACTIVE), dtype=np.float32)

    W_init      = nph.from_array(W_val,   name="W")
    k_init      = nph.from_array(k_val,   name="k_const")
    ones_init   = nph.from_array(ones_val, name="ones_topk")
    target_init = nph.from_array(np.array([[target_dc]], dtype=np.float32), name="target_dc")
    boost_init  = nph.from_array(np.array([[boost_strength]], dtype=np.float32), name="boost")

    nodes = [
        h.make_node("MatMul", ["X", "W"], ["z"]),
        h.make_node("Sub",    ["target_dc", "dc"], ["dc_diff"]),
        h.make_node("Mul",    ["boost", "dc_diff"], ["boost_term"]),
        h.make_node("Add",    ["z", "boost_term"], ["z_boosted"]),
        h.make_node("TopK",   ["z_boosted", "k_const"], ["topk_vals", "topk_idx"],
                    axis=-1, largest=1, sorted=0),
        # ScatterElements: scatter 1.0 into zero tensor at topk indices
        h.make_node("ScatterElements",
                    ["zeros_like_z", "topk_idx", "ones_topk"],
                    ["mask"],
                    axis=-1),
        h.make_node("Mul", ["z", "mask"], ["sparse_z"]),
    ]
    # zeros_like_z via ConstantOfShape
    shape_init = nph.from_array(np.array([1, N_UNITS], dtype=np.int64), name="z_shape")
    zero_val = nph.from_array(np.array([0.0], dtype=np.float32), name="zero_val")
    nodes_full = [
        h.make_node("MatMul", ["X", "W"], ["z"]),
        h.make_node("ConstantOfShape", ["z_shape"], ["zeros_like_z"],
                    value=onnx.helper.make_tensor("zero_fill", TensorProto.FLOAT, [1], [0.0])),
        h.make_node("Sub",    ["target_dc", "dc"], ["dc_diff"]),
        h.make_node("Mul",    ["boost", "dc_diff"], ["boost_term"]),
        h.make_node("Add",    ["z", "boost_term"], ["z_boosted"]),
        h.make_node("TopK",   ["z_boosted", "k_const"], ["topk_vals", "topk_idx"],
                    axis=-1, largest=1, sorted=0),
        h.make_node("ScatterElements",
                    ["zeros_like_z", "topk_idx", "ones_topk"],
                    ["mask"], axis=-1),
        h.make_node("Mul", ["z", "mask"], ["sparse_z"]),
    ]

    X_in = rng.normal(0, 1, (1, n_in)).astype(np.float32)
    graph = h.make_graph(
        nodes_full, "kwta_full",
        inputs=[
            h.make_tensor_value_info("X",  TensorProto.FLOAT, [1, n_in]),
            h.make_tensor_value_info("dc", TensorProto.FLOAT, [1, N_UNITS]),
        ],
        outputs=[
            h.make_tensor_value_info("sparse_z",  TensorProto.FLOAT, [1, N_UNITS]),
            h.make_tensor_value_info("topk_idx",  TensorProto.INT64,  [1, K_ACTIVE]),
        ],
        initializer=[W_init, k_init, ones_init, target_init, boost_init, shape_init],
    )
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 14)])
    feeds = {"X": X_in, "dc": dc_val}
    return model, feeds


# ── 5. Sparse-gather binding projection (Gather + small MatMul) ───────────────
def build_gather_binding():
    """
    Candidate rewrite for the ORT binding graph's dense sp_i @ W_region_i:
      gathered = Gather(W_region, active_indices, axis=0)  [K_ACTIVE_CAMPAIGN, D_PROJ]
      proj     = ReduceSum(gathered * vals[:, None], axis=0)  # weighted sum of gathered rows
    active_indices is an int64 GRAPH INPUT (CPU-computed at runtime), not an
    initializer — this is the untested part: Gather fed by a runtime index
    input, at campaign scale (n_units=9660, k=190), on MIGraphX/gfx1201.
    """
    n_units_campaign = 9660
    k_active_campaign = 190

    rng = np.random.default_rng(7)
    W_region = rng.normal(0, 0.01, (n_units_campaign, D_PROJ)).astype(np.float32)
    active_indices = np.sort(
        rng.choice(n_units_campaign, size=k_active_campaign, replace=False)
    ).astype(np.int64)
    active_vals = rng.normal(0, 1, (k_active_campaign,)).astype(np.float32)

    W_init = nph.from_array(W_region, name="W_region")

    nodes = [
        h.make_node("Gather", ["W_region", "active_indices"], ["gathered"], axis=0),
        h.make_node("Unsqueeze", ["active_vals", "unsq_axis"], ["vals_col"]),
        h.make_node("Mul", ["gathered", "vals_col"], ["weighted"]),
        h.make_node("ReduceSum", ["weighted", "reduce_axis"], ["proj"], keepdims=0),
    ]
    unsq_axis_init = nph.from_array(np.array([1], dtype=np.int64), name="unsq_axis")
    reduce_axis_init = nph.from_array(np.array([0], dtype=np.int64), name="reduce_axis")

    graph = h.make_graph(
        nodes, "gather_binding_test",
        inputs=[
            h.make_tensor_value_info("active_indices", TensorProto.INT64, [k_active_campaign]),
            h.make_tensor_value_info("active_vals", TensorProto.FLOAT, [k_active_campaign]),
        ],
        outputs=[h.make_tensor_value_info("proj", TensorProto.FLOAT, [D_PROJ])],
        initializer=[W_init, unsq_axis_init, reduce_axis_init],
    )
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 14)])
    feeds = {"active_indices": active_indices, "active_vals": active_vals}
    return model, feeds


# ── Run all probes ─────────────────────────────────────────────────────────────
probe("GRU (one step, 640→256, no bias)",            build_gru,       atol=1e-3)
probe("TopK (N=1024, K=20, largest-first)",          build_topk,      atol=0.0)  # exact
probe("Multi-head cross-attention (5×10, 4 heads)",  build_attention, atol=1e-3)
probe("Full k-WTA (MatMul+TopK+ScatterElements)",    build_kwta_full, atol=1e-4)
probe("Sparse-gather binding (Gather+Mul+ReduceSum, n=9660 k=190)", build_gather_binding, atol=1e-3)

print("\n" + "="*60)
print("SUMMARY")
for name, status, info in results:
    print(f"  [{status:5s}] {name}")
    if status == "ERROR":
        print(f"         {info}")
    elif status == "FAIL":
        print(f"         diffs: {info}")

all_passed = all(s == "PASS" for _, s, _ in results)
sys.exit(0 if all_passed else 1)

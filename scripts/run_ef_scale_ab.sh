#!/usr/bin/env bash
# At-scale A/B: compare --recall-ef 64 (default) vs --recall-ef 16
# Goal: grow warm index to 100k+ records so HNSW query latency tradeoff is visible.
# Uses two 1-hr walk clips for dense multimodal content.
# Run sequentially: A first, then B.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SESSIONS=10
CYCLES=3
CAPACITY=4096
CLIP_A="clips/long/4k_hdr_japan_Walk_in_Shibuya_Tokyo_japan_One_hour_uninterrupted_urban__[zbFASNIp0jI].mp4"
CLIP_B="clips/long/Dreamy_London_Walk_in_Summer_-_2023_London_1_Hour_Walk_4K_HDR_[4p4sIQ5aNpA].mp4"

echo "[ef-ab] ── Campaign A: recall-ef 64 (default) ──────────────────────"
SESSIONS=$SESSIONS CYCLES=$CYCLES CAPACITY=$CAPACITY \
  CLIPS="$CLIP_A $CLIP_B" \
  MEMORY_DIR=memory/ab_ef_scale_a \
  CAMPAIGN_DIR=memory/ab_ef_scale_a/campaign \
  ONNX_MODELS_DIR=memory/ab_ef_scale_a/onnx_models \
  bash scripts/train_campaign.sh

echo ""
echo "[ef-ab] ── Campaign B: recall-ef 16 ───────────────────────────────"
SESSIONS=$SESSIONS CYCLES=$CYCLES CAPACITY=$CAPACITY \
  CLIPS="$CLIP_A $CLIP_B" \
  MEMORY_DIR=memory/ab_ef_scale_b \
  CAMPAIGN_DIR=memory/ab_ef_scale_b/campaign \
  ONNX_MODELS_DIR=memory/ab_ef_scale_b/onnx_models \
  EXTRA_ARGS="--recall-ef 16" \
  bash scripts/train_campaign.sh

echo ""
echo "[ef-ab] Both campaigns complete. Compare:"
echo "  A (ef=64): memory/ab_ef_scale_a/campaign/metrics.jsonl"
echo "  B (ef=16): memory/ab_ef_scale_b/campaign/metrics.jsonl"
echo "Key fields: warm_query_us_mean, warm_recall_l2, warm_len, gist_recall_rate"

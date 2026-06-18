#!/usr/bin/env bash
# A/B: validate qwen25_text_hidden.onnx GPU path for NREM distillation targets.
#
# Control:   --models-dir models_control/  (no qwen25_text_hidden.onnx → CPU lm_session)
# Treatment: --models-dir models/          (qwen25_text_hidden.onnx present → MIGraphX)
#
# Key metrics to compare:
#   mean_proj_loss      — must not regress in treatment
#   distill_precompute_ms — should be lower in treatment (GPU speedup, cold cache only)
#   lm_hidden_computed  — same between arms (same number of unique texts)
#
# Prerequisite: models_control/ symlink dir must exist (no qwen25_text_hidden.onnx).
#   Run once: mkdir models_control && for f in models/*; do n=$(basename $f); [[ $n == qwen25_text_hidden* || $n == README.md ]] || ln -sf "../models/$n" models_control/$n; done
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SESSIONS="${SESSIONS:-5}"
CYCLES="${CYCLES:-2}"
CAPACITY="${CAPACITY:-8192}"

# Same clips as ab_lang_lm: 5 h3h3 (dense speech) + representative shorts
CLIPS="clips/h3h3/Ashton_Kutcher_Mila_Kunis_Just_Made_The_Biggest_Mistake_of_Their_Caree_[EPUKKVeTLpw].mp4 clips/h3h3/BBTV_Stole_620_000_Refuses_To_Return_It..._in_my_opinion_CEO_Shahrzad__[pHHVpvLqYCE].mp4 clips/h3h3/David_Dobrik_Sued_By_State_Farm_Over_Jeff_Wittek_Injury_Crew_Designs_N_[_uO9aEEOUa8].mp4 clips/h3h3/Debating_Fresh_Fit_Best_Moments_H3_Podcast_Clips_[F1n7HZA2pcc].mp4 clips/h3h3/Ethan_gets_banned_from_The_Metaverse_Full_episode_[FmHEAFQT5u8].mp4 clips/shorts/10_Awesome_Chemistry_Science_Projects_[Ng_obFzcgVw].mp4 clips/shorts/15_Awesome_Science_Project_Ideas_[qPJJDpBoPGU].mp4 clips/shorts/A_Brief_Introduction_on_Definition_of_Culture_[B589Ia5M2_Y].mp4 clips/shorts/Unraveling_the_Mystery_-_The_Enigmatic_Origins_of_Language_F_[NpfqIF_Ur0s].mp4 clips/shorts/Discovering_history_about_Mexican_American_Culture_[XPkhKHb3kTo].mp4"

if [ ! -d "$REPO_ROOT/models_control" ]; then
    echo "[ab-q1] ERROR: models_control/ missing. Create it first:"
    echo "  mkdir models_control && for f in models/*; do n=\$(basename \$f); [[ \$n == qwen25_text_hidden* || \$n == README.md ]] || ln -sf \"../models/\$n\" models_control/\$n; done"
    exit 1
fi

echo "[ab-q1] ── Control: CPU lm_session (no qwen25_text_hidden.onnx) ───"
SESSIONS=$SESSIONS CYCLES=$CYCLES CAPACITY=$CAPACITY \
    CLIPS="$CLIPS" \
    MEMORY_DIR=memory/ab_q1_text_hidden/control \
    CAMPAIGN_DIR=memory/ab_q1_text_hidden/control/campaign \
    EXTRA_ARGS="--models-dir models_control" \
    bash scripts/train_campaign.sh

echo ""
echo "[ab-q1] ── Treatment: GPU text_hidden_session (qwen25_text_hidden.onnx) ──"
SESSIONS=$SESSIONS CYCLES=$CYCLES CAPACITY=$CAPACITY \
    CLIPS="$CLIPS" \
    MEMORY_DIR=memory/ab_q1_text_hidden/treatment \
    CAMPAIGN_DIR=memory/ab_q1_text_hidden/treatment/campaign \
    bash scripts/train_campaign.sh

echo ""
echo "[ab-q1] Both arms complete. Compare:"
echo "  Control:   memory/ab_q1_text_hidden/control/campaign/metrics.jsonl"
echo "  Treatment: memory/ab_q1_text_hidden/treatment/campaign/metrics.jsonl"
echo ""
echo "Key fields:"
echo "  mean_proj_loss       — should match between arms (no regression)"
echo "  distill_precompute_ms — should be lower in treatment (GPU speedup)"
echo "  lm_hidden_computed   — should match (same text trace count)"
python3 - <<'PYEOF'
import json, sys

control_f = "memory/ab_q1_text_hidden/control/campaign/metrics.jsonl"
treat_f   = "memory/ab_q1_text_hidden/treatment/campaign/metrics.jsonl"

try:
    ctrl  = [json.loads(l) for l in open(control_f)]
    treat = [json.loads(l) for l in open(treat_f)]
except FileNotFoundError as e:
    print(f"metrics not found: {e}")
    sys.exit(0)

print(f"\n{'s':>3} {'c':>2}  {'ctrl_proj':>10} {'trt_proj':>10}  {'ctrl_ms':>8} {'trt_ms':>8}  {'ctrl_lm':>8} {'trt_lm':>8}")
print("-" * 70)
for a, b in zip(ctrl, treat):
    s, c = a['session'], a['cycle']
    cp = a.get('mean_proj_loss') or 0
    tp = b.get('mean_proj_loss') or 0
    cm = a.get('distill_precompute_ms', 0)
    tm = b.get('distill_precompute_ms', 0)
    cl = a.get('lm_hidden_computed', 0)
    tl = b.get('lm_hidden_computed', 0)
    print(f"{s:3d} {c:2d}  {cp:10.4f} {tp:10.4f}  {cm:8d} {tm:8d}  {cl:8d} {tl:8d}")

# Summary
ctrl_proj  = [r['mean_proj_loss'] for r in ctrl  if r.get('mean_proj_loss')]
treat_proj = [r['mean_proj_loss'] for r in treat if r.get('mean_proj_loss')]
if ctrl_proj and treat_proj:
    avg_c = sum(ctrl_proj) / len(ctrl_proj)
    avg_t = sum(treat_proj) / len(treat_proj)
    delta = (avg_t - avg_c) / avg_c * 100
    print(f"\navg proj_loss  control={avg_c:.4f}  treatment={avg_t:.4f}  delta={delta:+.1f}%")
    verdict = "PASS" if abs(delta) < 10 else "FAIL"
    print(f"verdict: {verdict} (threshold: <10% regression in mean_proj_loss)")

ctrl_ms  = [r['distill_precompute_ms'] for r in ctrl  if r.get('distill_precompute_ms',0) > 0]
treat_ms = [r['distill_precompute_ms'] for r in treat if r.get('distill_precompute_ms',0) > 0]
if ctrl_ms and treat_ms:
    speedup = sum(ctrl_ms)/len(ctrl_ms) / (sum(treat_ms)/len(treat_ms))
    print(f"avg distill_ms control={sum(ctrl_ms)/len(ctrl_ms):.0f}  treatment={sum(treat_ms)/len(treat_ms):.0f}  speedup={speedup:.1f}×")
PYEOF

#!/usr/bin/env bash
# A/B: validate ADR 0033 LoRA fine-tuning of Qwen2.5-0.5B.
#
# Control:   models_lora_ab_control/ (qwen25_lm.onnx = pretrained)
# Treatment: models/                 (qwen25_lm.onnx = LoRA fine-tuned)
#
# Key metric: mean_proj_loss — lower in treatment indicates LoRA adapts the LM
# to produce hidden states better aligned with the episode projection.
#
# Usage:
#   bash scripts/ab_lora.sh
#   SESSIONS=3 CYCLES=2 bash scripts/ab_lora.sh
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SESSIONS="${SESSIONS:-3}"
CYCLES="${CYCLES:-2}"
CAPACITY="${CAPACITY:-8192}"

CLIPS="clips/h3h3/Ashton_Kutcher_Mila_Kunis_Just_Made_The_Biggest_Mistake_of_Their_Caree_[EPUKKVeTLpw].mp4 clips/h3h3/BBTV_Stole_620_000_Refuses_To_Return_It..._in_my_opinion_CEO_Shahrzad__[pHHVpvLqYCE].mp4 clips/h3h3/David_Dobrik_Sued_By_State_Farm_Over_Jeff_Wittek_Injury_Crew_Designs_N_[_uO9aEEOUa8].mp4 clips/h3h3/Debating_Fresh_Fit_Best_Moments_H3_Podcast_Clips_[F1n7HZA2pcc].mp4 clips/h3h3/Ethan_gets_banned_from_The_Metaverse_Full_episode_[FmHEAFQT5u8].mp4 clips/shorts/10_Awesome_Chemistry_Science_Projects_[Ng_obFzcgVw].mp4 clips/shorts/15_Awesome_Science_Project_Ideas_[qPJJDpBoPGU].mp4 clips/shorts/A_Brief_Introduction_on_Definition_of_Culture_[B589Ia5M2_Y].mp4 clips/shorts/Unraveling_the_Mystery_-_The_Enigmatic_Origins_of_Language_F_[NpfqIF_Ur0s].mp4 clips/shorts/Discovering_history_about_Mexican_Culture_[XPkhKHb3kTo].mp4"

echo "[ab-lora] ── Control: pretrained Qwen2.5 (qwen25_lm_pretrained.onnx) ──────"
SESSIONS=$SESSIONS CYCLES=$CYCLES CAPACITY=$CAPACITY \
    CLIPS="$CLIPS" \
    MEMORY_DIR=memory/ab_lora/control \
    CAMPAIGN_DIR=memory/ab_lora/control/campaign \
    EXTRA_ARGS="--models-dir models_lora_ab_control" \
    bash scripts/train_campaign.sh

echo ""
echo "[ab-lora] ── Treatment: LoRA fine-tuned Qwen2.5 (qwen25_lm.onnx) ──────────"
SESSIONS=$SESSIONS CYCLES=$CYCLES CAPACITY=$CAPACITY \
    CLIPS="$CLIPS" \
    MEMORY_DIR=memory/ab_lora/treatment \
    CAMPAIGN_DIR=memory/ab_lora/treatment/campaign \
    bash scripts/train_campaign.sh

echo ""
echo "[ab-lora] Both arms complete. Results:"
echo "  Control:   memory/ab_lora/control/campaign/metrics.jsonl"
echo "  Treatment: memory/ab_lora/treatment/campaign/metrics.jsonl"

python3 - <<'PYEOF'
import json, sys

control_f = "memory/ab_lora/control/campaign/metrics.jsonl"
treat_f   = "memory/ab_lora/treatment/campaign/metrics.jsonl"

try:
    ctrl  = [json.loads(l) for l in open(control_f)]
    treat = [json.loads(l) for l in open(treat_f)]
except FileNotFoundError as e:
    print(f"metrics not found: {e}")
    sys.exit(0)

print(f"\n{'s':>3} {'c':>2}  {'ctrl_proj':>12} {'trt_proj':>12}  {'delta%':>8}  {'ctrl_mse':>10} {'trt_mse':>10}")
print("-" * 68)
for a, b in zip(ctrl, treat):
    s, c = a['session'], a['cycle']
    cp = a.get('mean_proj_loss') or 0.0
    tp = b.get('mean_proj_loss') or 0.0
    delta = (tp - cp) / max(cp, 1e-9) * 100
    cm = a.get('output_mse') or 0.0
    tm = b.get('output_mse') or 0.0
    print(f"{s:3d} {c:2d}  {cp:12.6f} {tp:12.6f}  {delta:+8.1f}%  {cm:10.6f} {tm:10.6f}")

ctrl_proj  = [r['mean_proj_loss'] for r in ctrl  if r.get('mean_proj_loss')]
treat_proj = [r['mean_proj_loss'] for r in treat if r.get('mean_proj_loss')]
if ctrl_proj and treat_proj:
    avg_c = sum(ctrl_proj) / len(ctrl_proj)
    avg_t = sum(treat_proj) / len(treat_proj)
    delta = (avg_t - avg_c) / avg_c * 100
    print(f"\navg proj_loss  control={avg_c:.6f}  treatment={avg_t:.6f}  delta={delta:+.1f}%")
    if delta < -5:
        verdict = "PASS — LoRA improves proj_loss (deploy treatment model)"
    elif delta > 5:
        verdict = "FAIL — LoRA hurts proj_loss (restore pretrained, investigate)"
    else:
        verdict = "INCONCLUSIVE (<5% difference — run more sessions or collect more training data)"
    print(f"verdict: {verdict}")
PYEOF

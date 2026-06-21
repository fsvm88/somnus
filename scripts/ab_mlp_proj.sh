#!/usr/bin/env bash
# A/B: validate ADR 0032 MLP projection vs single linear layer.
#
# Control:   --linear-proj  (EPISODE_DIM → N_REGIONS*LM_HIDDEN, single layer)
# Treatment: (default MLP)  (EPISODE_DIM → 1024 → N_REGIONS*LM_HIDDEN, ReLU)
#
# Key metric: mean_proj_loss — treatment should converge lower than control.
# Both arms start from fresh memory (no snapshot); comparison is fair from session 1.
#
# Usage:
#   bash scripts/ab_mlp_proj.sh
#   SESSIONS=5 CYCLES=2 bash scripts/ab_mlp_proj.sh
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SESSIONS="${SESSIONS:-5}"
CYCLES="${CYCLES:-2}"
CAPACITY="${CAPACITY:-8192}"

CLIPS="clips/h3h3/Ashton_Kutcher_Mila_Kunis_Just_Made_The_Biggest_Mistake_of_Their_Caree_[EPUKKVeTLpw].mp4 clips/h3h3/BBTV_Stole_620_000_Refuses_To_Return_It..._in_my_opinion_CEO_Shahrzad__[pHHVpvLqYCE].mp4 clips/h3h3/David_Dobrik_Sued_By_State_Farm_Over_Jeff_Wittek_Injury_Crew_Designs_N_[_uO9aEEOUa8].mp4 clips/h3h3/Debating_Fresh_Fit_Best_Moments_H3_Podcast_Clips_[F1n7HZA2pcc].mp4 clips/h3h3/Ethan_gets_banned_from_The_Metaverse_Full_episode_[FmHEAFQT5u8].mp4 clips/shorts/10_Awesome_Chemistry_Science_Projects_[Ng_obFzcgVw].mp4 clips/shorts/15_Awesome_Science_Project_Ideas_[qPJJDpBoPGU].mp4 clips/shorts/A_Brief_Introduction_on_Definition_of_Culture_[B589Ia5M2_Y].mp4 clips/shorts/Unraveling_the_Mystery_-_The_Enigmatic_Origins_of_Language_F_[NpfqIF_Ur0s].mp4 clips/shorts/Discovering_history_about_Mexican_American_Culture_[XPkhKHb3kTo].mp4"

echo "[ab-mlp-proj] ── Control: linear projection (--linear-proj) ──────────────"
SESSIONS=$SESSIONS CYCLES=$CYCLES CAPACITY=$CAPACITY \
    CLIPS="$CLIPS" \
    MEMORY_DIR=memory/ab_mlp_proj/control \
    CAMPAIGN_DIR=memory/ab_mlp_proj/control/campaign \
    EXTRA_ARGS="--linear-proj" \
    bash scripts/train_campaign.sh

echo ""
echo "[ab-mlp-proj] ── Treatment: MLP projection (default) ──────────────────────"
SESSIONS=$SESSIONS CYCLES=$CYCLES CAPACITY=$CAPACITY \
    CLIPS="$CLIPS" \
    MEMORY_DIR=memory/ab_mlp_proj/treatment \
    CAMPAIGN_DIR=memory/ab_mlp_proj/treatment/campaign \
    bash scripts/train_campaign.sh

echo ""
echo "[ab-mlp-proj] Both arms complete. Results:"
echo "  Control:   memory/ab_mlp_proj/control/campaign/metrics.jsonl"
echo "  Treatment: memory/ab_mlp_proj/treatment/campaign/metrics.jsonl"

python3 - <<'PYEOF'
import json, sys

control_f = "memory/ab_mlp_proj/control/campaign/metrics.jsonl"
treat_f   = "memory/ab_mlp_proj/treatment/campaign/metrics.jsonl"

try:
    ctrl  = [json.loads(l) for l in open(control_f)]
    treat = [json.loads(l) for l in open(treat_f)]
except FileNotFoundError as e:
    print(f"metrics not found: {e}")
    sys.exit(0)

print(f"\n{'s':>3} {'c':>2}  {'ctrl_proj':>12} {'trt_proj':>12}  {'delta':>8}")
print("-" * 50)
for a, b in zip(ctrl, treat):
    s, c = a['session'], a['cycle']
    cp = a.get('mean_proj_loss') or 0.0
    tp = b.get('mean_proj_loss') or 0.0
    delta = (tp - cp) / max(cp, 1e-9) * 100
    print(f"{s:3d} {c:2d}  {cp:12.6f} {tp:12.6f}  {delta:+8.1f}%")

ctrl_proj  = [r['mean_proj_loss'] for r in ctrl  if r.get('mean_proj_loss')]
treat_proj = [r['mean_proj_loss'] for r in treat if r.get('mean_proj_loss')]
if ctrl_proj and treat_proj:
    avg_c = sum(ctrl_proj) / len(ctrl_proj)
    avg_t = sum(treat_proj) / len(treat_proj)
    delta = (avg_t - avg_c) / avg_c * 100
    print(f"\navg proj_loss  control={avg_c:.6f}  treatment={avg_t:.6f}  delta={delta:+.1f}%")
    # PASS if MLP is at least 5% better (lower), FAIL if worse or negligible
    if delta < -5:
        verdict = "PASS (MLP improves projection)"
    elif delta > 5:
        verdict = "FAIL (MLP is worse — revert to linear)"
    else:
        verdict = "INCONCLUSIVE (<5% difference — run more sessions)"
    print(f"verdict: {verdict}")
PYEOF

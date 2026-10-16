#!/usr/bin/env bash
# HISTORICAL: predates the 2026-08-15 standing A/B instruction (build+run both arms with
# fine-metrics, record deviations). Results are recorded in ai-docs/; to re-run, add
# --features fine-metrics to the build first.
# A/B: validate DER++ auto-lambda weighting vs default unweighted sum (gaps.md item 3, phase 3).
#
# Control:   (default)              wake_loss weighted by implicit lambda=1
# Treatment: --der-auto-lambda      wake_loss weighted by 1/EWMA-variance, clamped [0.1, 10.0]
#
# Key metric: output_mse — treatment should improve or stay stable vs control (primary).
# Secondary: per-region cosines should remain stable.
# Both arms start from fresh memory (no snapshot); comparison is fair from session 1.
#
# Rationale (arXiv:2504.20932): weighting the stale-output MSE term by its own uncertainty
# avoids a hand-tuned lambda while still down-weighting a noisy wake_loss signal.
#
# Usage:
#   bash scripts/ab_der_auto_lambda.sh
#   SESSIONS=5 CYCLES=2 bash scripts/ab_der_auto_lambda.sh
set -euo pipefail
export SEED="${SEED:-42}"   # fixed burn RNG seed for both arms (standing reproducibility policy)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SESSIONS="${SESSIONS:-5}"
CYCLES="${CYCLES:-2}"
CAPACITY="${CAPACITY:-8192}"

CLIPS="clips/h3h3/Ashton_Kutcher_Mila_Kunis_Just_Made_The_Biggest_Mistake_of_Their_Caree_[EPUKKVeTLpw].mp4 clips/h3h3/BBTV_Stole_620_000_Refuses_To_Return_It..._in_my_opinion_CEO_Shahrzad__[pHHVpvLqYCE].mp4 clips/h3h3/David_Dobrik_Sued_By_State_Farm_Over_Jeff_Wittek_Injury_Crew_Designs_N_[_uO9aEEOUa8].mp4 clips/h3h3/Debating_Fresh_Fit_Best_Moments_H3_Podcast_Clips_[F1n7HZA2pcc].mp4 clips/h3h3/Ethan_gets_banned_from_The_Metaverse_Full_episode_[FmHEAFQT5u8].mp4 clips/shorts/10_Awesome_Chemistry_Science_Projects_[Ng_obFzcgVw].mp4 clips/shorts/15_Awesome_Science_Project_Ideas_[qPJJDpBoPGU].mp4 clips/shorts/A_Brief_Introduction_on_Definition_of_Culture_[B589Ia5M2_Y].mp4 clips/shorts/Unraveling_the_Mystery_-_The_Enigmatic_Origins_of_Language_F_[NpfqIF_Ur0s].mp4 clips/shorts/Discovering_history_about_Mexican_American_Culture_[XPkhKHb3kTo].mp4"

echo "[ab-der-auto-lambda] ── Control: unweighted (lambda=1, default) ──────────────"
SESSIONS=$SESSIONS CYCLES=$CYCLES CAPACITY=$CAPACITY \
    CLIPS="$CLIPS" \
    MEMORY_DIR=memory/ab_der_auto_lambda/control \
    CAMPAIGN_DIR=memory/ab_der_auto_lambda/control/campaign \
    bash scripts/train_campaign.sh

echo ""
echo "[ab-der-auto-lambda] ── Treatment: --der-auto-lambda ──────────────────────"
SESSIONS=$SESSIONS CYCLES=$CYCLES CAPACITY=$CAPACITY \
    CLIPS="$CLIPS" \
    MEMORY_DIR=memory/ab_der_auto_lambda/treatment \
    CAMPAIGN_DIR=memory/ab_der_auto_lambda/treatment/campaign \
    EXTRA_ARGS="--der-auto-lambda" \
    bash scripts/train_campaign.sh

echo ""
echo "[ab-der-auto-lambda] Both arms complete. Results:"
echo "  Control:   memory/ab_der_auto_lambda/control/campaign/metrics.jsonl"
echo "  Treatment: memory/ab_der_auto_lambda/treatment/campaign/metrics.jsonl"

python3 - <<'PYEOF'
import json, sys

control_f = "memory/ab_der_auto_lambda/control/campaign/metrics.jsonl"
treat_f   = "memory/ab_der_auto_lambda/treatment/campaign/metrics.jsonl"

try:
    ctrl  = [json.loads(l) for l in open(control_f)]
    treat = [json.loads(l) for l in open(treat_f)]
except FileNotFoundError as e:
    print(f"metrics not found: {e}")
    sys.exit(0)

print(f"\n{'s':>3} {'c':>2}  {'ctrl_mse':>12} {'trt_mse':>12}  {'delta':>8}")
print("-" * 50)
for a, b in zip(ctrl, treat):
    s, c = a['session'], a['cycle']
    cm = a.get('mean_output_mse') or 0.0
    tm = b.get('mean_output_mse') or 0.0
    delta = (tm - cm) / max(cm, 1e-9) * 100
    print(f"{s:3d} {c:2d}  {cm:12.6f} {tm:12.6f}  {delta:+8.1f}%")

ctrl_mse  = [r['mean_output_mse'] for r in ctrl  if r.get('mean_output_mse')]
treat_mse = [r['mean_output_mse'] for r in treat if r.get('mean_output_mse')]
if ctrl_mse and treat_mse:
    avg_c = sum(ctrl_mse) / len(ctrl_mse)
    avg_t = sum(treat_mse) / len(treat_mse)
    delta = (avg_t - avg_c) / avg_c * 100
    print(f"\navg output_mse  control={avg_c:.6f}  treatment={avg_t:.6f}  delta={delta:+.1f}%")
    # PASS if auto-lambda improves or matches (delta < +2%), FAIL if regresses significantly (+10%)
    if delta <= 2:
        verdict = "PASS (auto-lambda improves or matches consolidation)"
    elif delta > 10:
        verdict = "FAIL (auto-lambda significantly worse — do not adopt)"
    else:
        verdict = "INCONCLUSIVE (marginal regression, run more sessions)"
    print(f"verdict: {verdict}")
PYEOF

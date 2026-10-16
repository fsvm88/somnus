#!/usr/bin/env bash
# HISTORICAL: predates the 2026-08-15 standing A/B instruction (build+run both arms with
# fine-metrics, record deviations). Results are recorded in ai-docs/; to re-run, add
# --features fine-metrics to the build first.
# A/B: validate BatchTopK SAE vs per-sample top-k k-WTA.
#
# Control:   (default)        (per-sample top-k, order-dependent duty-cycle)
# Treatment: --batch-topk     (batch-quantile threshold, order-independent duty-cycle)
#
# Key metric: output_mse — treatment should not regress vs control (primary).
# Secondary: per-region cosines should remain stable.
# Both arms start from fresh memory (no snapshot); comparison is fair from session 1.
#
# Semantic change: batch selection (snapshot duty-cycle, compute threshold, then update)
# vs per-sample selection (interleaved boost/decay per sample). Same total decay count.
# Per gaps.md (arXiv:2412.06410): BatchTopK shows quality improvements on GPT-2 Small
# and Gemma 2 2B at matched sparsity; L1 penalty (ADR 0025) was detrimental, so this
# pure L2 threshold approach is the next experiment.
#
# Usage:
#   bash scripts/ab_batch_topk.sh
#   SESSIONS=5 CYCLES=2 bash scripts/ab_batch_topk.sh
set -euo pipefail
export SEED="${SEED:-42}"   # fixed burn RNG seed for both arms (standing reproducibility policy)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SESSIONS="${SESSIONS:-5}"
CYCLES="${CYCLES:-2}"
CAPACITY="${CAPACITY:-8192}"

CLIPS="clips/h3h3/Ashton_Kutcher_Mila_Kunis_Just_Made_The_Biggest_Mistake_of_Their_Caree_[EPUKKVeTLpw].mp4 clips/h3h3/BBTV_Stole_620_000_Refuses_To_Return_It..._in_my_opinion_CEO_Shahrzad__[pHHVpvLqYCE].mp4 clips/h3h3/David_Dobrik_Sued_By_State_Farm_Over_Jeff_Wittek_Injury_Crew_Designs_N_[_uO9aEEOUa8].mp4 clips/h3h3/Debating_Fresh_Fit_Best_Moments_H3_Podcast_Clips_[F1n7HZA2pcc].mp4 clips/h3h3/Ethan_gets_banned_from_The_Metaverse_Full_episode_[FmHEAFQT5u8].mp4 clips/shorts/10_Awesome_Chemistry_Science_Projects_[Ng_obFzcgVw].mp4 clips/shorts/15_Awesome_Science_Project_Ideas_[qPJJDpBoPGU].mp4 clips/shorts/A_Brief_Introduction_on_Definition_of_Culture_[B589Ia5M2_Y].mp4 clips/shorts/Unraveling_the_Mystery_-_The_Enigmatic_Origins_of_Language_F_[NpfqIF_Ur0s].mp4 clips/shorts/Discovering_history_about_Mexican_American_Culture_[XPkhKHb3kTo].mp4"

echo "[ab-batch-topk] ── Control: per-sample k-WTA (default) ──────────────────────"
SESSIONS=$SESSIONS CYCLES=$CYCLES CAPACITY=$CAPACITY \
    CLIPS="$CLIPS" \
    MEMORY_DIR=memory/ab_batch_topk/control \
    CAMPAIGN_DIR=memory/ab_batch_topk/control/campaign \
    bash scripts/train_campaign.sh

echo ""
echo "[ab-batch-topk] ── Treatment: batch-quantile k-WTA (--batch-topk) ──────────"
SESSIONS=$SESSIONS CYCLES=$CYCLES CAPACITY=$CAPACITY \
    CLIPS="$CLIPS" \
    MEMORY_DIR=memory/ab_batch_topk/treatment \
    CAMPAIGN_DIR=memory/ab_batch_topk/treatment/campaign \
    EXTRA_ARGS="--batch-topk" \
    bash scripts/train_campaign.sh

echo ""
echo "[ab-batch-topk] Both arms complete. Results:"
echo "  Control:   memory/ab_batch_topk/control/campaign/metrics.jsonl"
echo "  Treatment: memory/ab_batch_topk/treatment/campaign/metrics.jsonl"

python3 - <<'PYEOF'
import json, sys

control_f = "memory/ab_batch_topk/control/campaign/metrics.jsonl"
treat_f   = "memory/ab_batch_topk/treatment/campaign/metrics.jsonl"

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
    cm = a.get('output_mse') or 0.0
    tm = b.get('output_mse') or 0.0
    delta = (tm - cm) / max(cm, 1e-9) * 100
    print(f"{s:3d} {c:2d}  {cm:12.6f} {tm:12.6f}  {delta:+8.1f}%")

ctrl_mse  = [r['output_mse'] for r in ctrl  if r.get('output_mse')]
treat_mse = [r['output_mse'] for r in treat if r.get('output_mse')]
if ctrl_mse and treat_mse:
    avg_c = sum(ctrl_mse) / len(ctrl_mse)
    avg_t = sum(treat_mse) / len(treat_mse)
    delta = (avg_t - avg_c) / avg_c * 100
    print(f"\navg output_mse  control={avg_c:.6f}  treatment={avg_t:.6f}  delta={delta:+.1f}%")
    # PASS if BatchTopK does not regress (delta < +5%), FAIL if significantly worse (+10%)
    if delta < 5:
        verdict = "PASS (BatchTopK does not regress)"
    elif delta > 10:
        verdict = "FAIL (BatchTopK is significantly worse — revert)"
    else:
        verdict = "INCONCLUSIVE (marginal regression, run more sessions)"
    print(f"verdict: {verdict}")
PYEOF

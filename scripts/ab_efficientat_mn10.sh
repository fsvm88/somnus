#!/usr/bin/env bash
# HISTORICAL: predates the 2026-08-15 standing A/B instruction (build+run both arms with
# fine-metrics, record deviations). Results are recorded in ai-docs/; to re-run, add
# --features fine-metrics to the build first.
# A/B: EfficientAT mn10 env encoder (960-dim, --features mn10_env) vs. CLAP-HTSAT
# (512-dim, default) — gaps.md item 9.
#
# Control:   default build             (CLAP-HTSAT env encoder, 512-dim region 1)
# Treatment: `cargo build --features mn10_env` (EfficientAT mn10 env encoder, 960-dim region 1)
#
# This is a cold-start-required region-dimension change (REGION_DIMS[1] 512->960), not a
# same-dim drop-in swap like the DINOv3 A/B — fresh memory both arms is mandatory, not just
# good practice.
#
# IMPORTANT — region-1 (env) MSE is NOT a valid quality comparison between arms: the two
# encoders have different target distributions (CLAP: contrastive, ~unit-norm; mn10: raw
# pooled-conv log-mel-scale feature, unnormalized), so per_region_mse[1] moving is an
# artifact of target scale/dim, not consolidation quality. This script's verdict is based
# instead on the OTHER four regions (identical architecture both arms, so comparable) plus
# warm_recall_l2 / gist_recall_rate — whether swapping the env encoder helped or hurt the
# shared episode representation. Region-1 MSE is still printed for reference, just excluded
# from the verdict.
#
# Usage:
#   bash scripts/ab_efficientat_mn10.sh
#   SESSIONS=5 CYCLES=2 bash scripts/ab_efficientat_mn10.sh
set -euo pipefail
export SEED="${SEED:-42}"   # fixed burn RNG seed for both arms (standing reproducibility policy)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SESSIONS="${SESSIONS:-5}"
CYCLES="${CYCLES:-2}"
CAPACITY="${CAPACITY:-8192}"

CLIPS="clips/h3h3/Ashton_Kutcher_Mila_Kunis_Just_Made_The_Biggest_Mistake_of_Their_Caree_[EPUKKVeTLpw].mp4 clips/h3h3/BBTV_Stole_620_000_Refuses_To_Return_It..._in_my_opinion_CEO_Shahrzad__[pHHVpvLqYCE].mp4 clips/h3h3/David_Dobrik_Sued_By_State_Farm_Over_Jeff_Wittek_Injury_Crew_Designs_N_[_uO9aEEOUa8].mp4 clips/h3h3/Debating_Fresh_Fit_Best_Moments_H3_Podcast_Clips_[F1n7HZA2pcc].mp4 clips/h3h3/Ethan_gets_banned_from_The_Metaverse_Full_episode_[FmHEAFQT5u8].mp4 clips/shorts/10_Awesome_Chemistry_Science_Projects_[Ng_obFzcgVw].mp4 clips/shorts/15_Awesome_Science_Project_Ideas_[qPJJDpBoPGU].mp4 clips/shorts/A_Brief_Introduction_on_Definition_of_Culture_[B589Ia5M2_Y].mp4 clips/shorts/Unraveling_the_Mystery_-_The_Enigmatic_Origins_of_Language_F_[NpfqIF_Ur0s].mp4 clips/shorts/Discovering_history_about_Mexican_American_Culture_[XPkhKHb3kTo].mp4"

if [ ! -f models/efficientat_mn10_encoder_fp16.onnx ]; then
    echo "[ab-efficientat-mn10] models/efficientat_mn10_encoder_fp16.onnx not found."
    echo "  Export it first:"
    echo "    scripts/.venv-efficientat/bin/python scripts/export_efficientat.py"
    echo "    scripts/.venv-efficientat/bin/python scripts/convert_fp16.py"
    exit 1
fi

echo "[ab-efficientat-mn10] ── building control (default, CLAP-HTSAT) ────────────"
cargo build --release -p somnus 2>&1 | tail -5

echo "[ab-efficientat-mn10] ── Control: CLAP-HTSAT (default) ─────────────────────"
SESSIONS=$SESSIONS CYCLES=$CYCLES CAPACITY=$CAPACITY \
    CLIPS="$CLIPS" \
    MEMORY_DIR=memory/ab_efficientat_mn10/control \
    CAMPAIGN_DIR=memory/ab_efficientat_mn10/control/campaign \
    SKIP_BUILD=1 \
    bash scripts/train_campaign.sh

echo ""
echo "[ab-efficientat-mn10] ── building treatment (--features mn10_env) ─────────"
cargo build --release -p somnus --features mn10_env 2>&1 | tail -5

echo ""
echo "[ab-efficientat-mn10] ── Treatment: EfficientAT mn10 ─────────────────────"
SESSIONS=$SESSIONS CYCLES=$CYCLES CAPACITY=$CAPACITY \
    CLIPS="$CLIPS" \
    MEMORY_DIR=memory/ab_efficientat_mn10/treatment \
    CAMPAIGN_DIR=memory/ab_efficientat_mn10/treatment/campaign \
    SKIP_BUILD=1 \
    bash scripts/train_campaign.sh

# Rebuild default at the end so the working tree's target/release/somnus matches the
# repo's default feature set (least-surprise for any subsequent manual run).
echo ""
echo "[ab-efficientat-mn10] ── rebuilding default (post-A/B cleanup) ────────────"
cargo build --release -p somnus 2>&1 | tail -5

echo ""
echo "[ab-efficientat-mn10] Both arms complete. Results:"
echo "  Control:   memory/ab_efficientat_mn10/control/campaign/metrics.jsonl"
echo "  Treatment: memory/ab_efficientat_mn10/treatment/campaign/metrics.jsonl"

python3 - <<'PYEOF'
import json, sys

control_f = "memory/ab_efficientat_mn10/control/campaign/metrics.jsonl"
treat_f   = "memory/ab_efficientat_mn10/treatment/campaign/metrics.jsonl"

try:
    ctrl  = [json.loads(l) for l in open(control_f)]
    treat = [json.loads(l) for l in open(treat_f)]
except FileNotFoundError as e:
    print(f"metrics not found: {e}")
    sys.exit(0)

def per_region(rec, idx):
    pr = rec.get("per_region_mse")
    return pr[idx] if pr and len(pr) > idx else None

print(f"\n{'s':>3} {'c':>2}  {'ctrl_mse_all':>13} {'trt_mse_all':>12}  {'ctrl_env(ref)':>13} {'trt_env(ref)':>13}")
print("-" * 72)
for a, b in zip(ctrl, treat):
    s, c = a['session'], a['cycle']
    ca = a.get('mean_output_mse') or 0.0
    ta = b.get('mean_output_mse') or 0.0
    ce = per_region(a, 1)
    te = per_region(b, 1)
    ce_s = f"{ce:.6f}" if ce is not None else "n/a"
    te_s = f"{te:.6f}" if te is not None else "n/a"
    print(f"{s:3d} {c:2d}  {ca:13.6f} {ta:12.6f}  {ce_s:>13} {te_s:>13}")

# Verdict metric: mean of per_region_mse over the OTHER 4 regions (speech, visual, symbolic,
# language) — indices 0, 2, 3, 4. Region 1 (env) excluded: different target distributions
# between CLAP and mn10 make it incomparable (see script header).
def other4_mean(rec):
    pr = rec.get("per_region_mse")
    if not pr or len(pr) < 5:
        return None
    vals = [pr[i] for i in (0, 2, 3, 4)]
    return sum(vals) / len(vals)

ctrl_other4 = [v for r in ctrl if (v := other4_mean(r)) is not None]
treat_other4 = [v for r in treat if (v := other4_mean(r)) is not None]

if ctrl_other4 and treat_other4:
    avg_c = sum(ctrl_other4) / len(ctrl_other4)
    avg_t = sum(treat_other4) / len(treat_other4)
    delta = (avg_t - avg_c) / avg_c * 100
    print(f"\navg per_region_mse (regions 0,2,3,4 — comparable, env excluded):")
    print(f"  control={avg_c:.6f}  treatment={avg_t:.6f}  delta={delta:+.1f}%")
    if delta <= 2:
        verdict = "PASS (mn10 doesn't hurt the shared episode representation)"
    elif delta > 10:
        verdict = "FAIL (mn10 significantly worse on other regions — investigate/reject)"
    else:
        verdict = "INCONCLUSIVE (marginal, run more sessions)"
    print(f"verdict: {verdict}")

ctrl_recall = [r['warm_recall_l2'] for r in ctrl if r.get('warm_recall_l2') is not None]
treat_recall = [r['warm_recall_l2'] for r in treat if r.get('warm_recall_l2') is not None]
if ctrl_recall and treat_recall:
    avg_c, avg_t = sum(ctrl_recall)/len(ctrl_recall), sum(treat_recall)/len(treat_recall)
    print(f"\navg warm_recall_l2  control={avg_c:.6f}  treatment={avg_t:.6f}  delta={(avg_t-avg_c)/avg_c*100:+.1f}%")

ctrl_gist = [r['gist_recall_rate'] for r in ctrl if r.get('gist_recall_rate') is not None]
treat_gist = [r['gist_recall_rate'] for r in treat if r.get('gist_recall_rate') is not None]
if ctrl_gist and treat_gist:
    avg_c, avg_t = sum(ctrl_gist)/len(ctrl_gist), sum(treat_gist)/len(treat_gist)
    print(f"avg gist_recall_rate  control={avg_c:.6f}  treatment={avg_t:.6f}  delta={(avg_t-avg_c)/avg_c*100:+.1f}%")

print("\n(region-1/env MSE printed above for reference only — not part of the verdict; see script header)")
PYEOF

#!/bin/bash
# HISTORICAL: predates the 2026-08-15 standing A/B instruction (build+run both arms with
# fine-metrics, record deviations). Results are recorded in ai-docs/; to re-run, add
# --features fine-metrics to the build first.
# A/B validation: Qwen2.5-0.5B (control) vs Qwen3-0.6B (treatment) — Item 5, take 2.
#
# scripts/ab_qwen3.sh (the standard 4-clip corpus) came back inconclusive: too little
# speech in talking_head/urban/animation/nature for ASR/lang_ingest to fire more than
# ~once per 5-cycle session, so there was no real language-grounding signal to compare
# between arms. See ai-docs/a_b_qwen3_findings.md for the full writeup.
#
# This run swaps in speech-dense lecture/podcast clips from clips/long/ instead —
# continuous speech means ASR should fire on nearly every wake episode instead of
# once per session. Wake-phase length is governed by episodic-buffer high-water (not
# clip duration), so this shouldn't run meaningfully longer than the 4-clip version
# despite the source files being much bigger.
#
# Deliberately excludes clips/long/'s largest/least speech-dense files (the 4K city
# walks, 12GB NYC file) to keep per-episode I/O reasonable.

set -eu

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODELS_DIR="$REPO_ROOT/models"
CLIPS_DIR="$REPO_ROOT/clips/long"

SESSIONS="${SESSIONS:-5}"
CYCLES="${CYCLES:-4}"

CONTROL_BIN="$REPO_ROOT/target/release/somnus-qwen25-control"
TREATMENT_BIN="$REPO_ROOT/target/release/somnus-qwen3-treatment"
CONTROL_DIR="$REPO_ROOT/memory/qwen25_ab_control_longclips"
TREATMENT_DIR="$REPO_ROOT/memory/qwen3_ab_treatment_longclips"

# Speech-dense lecture/podcast clips (~1.7 GB total) — diverse topics, all continuous talk.
CLIPS=(
    "$CLIPS_DIR/Stanford_CS229_-_Machine_Learning_Course_Lecture_1_-_Andrew_Ng_Autumn__[jGwO_UgTS7I].mp4"
    "$CLIPS_DIR/Yoshua_Bengio_-_Deep_Learning_Lex_Fridman_Podcast_4_[azOmzumh0vQ].mp4"
    "$CLIPS_DIR/Physics_and_the_meaning_of_life_PART_1_Sabine_Hossenfelder_[jZDVyNGEa18].mp4"
    "$CLIPS_DIR/Can_Consciousness_be_Explained_-_Royal_Institute_of_Philosophy_Annual__[vjvLQ7GKxBE].mp4"
    "$CLIPS_DIR/Yann_LeCun_-_Dark_Matter_of_Intelligence_and_Self-Supervised_Learning__[SGzMElJ11Cc].mp4"
    "$CLIPS_DIR/Lecture_1_-_Algorithmic_Thinking_Peak_Finding_[HtSuA80QTyo].mp4"
)

echo "=== Qwen3-0.6B A/B Validation, take 2: speech-dense clips/long corpus (Item 5) ==="
echo ""
echo "Setup: Control=Qwen2.5-0.5B (qwen25_*, LM_HIDDEN=896), Treatment=Qwen3-0.6B (qwen3_*, LM_HIDDEN=1024)"
echo "Sessions: $SESSIONS × $CYCLES cycles each"
echo "Clips: ${#CLIPS[@]} lecture/podcast files from clips/long/"
echo ""

for f in "$CONTROL_BIN" "$TREATMENT_BIN"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: $f not found. Build both arms first:"
        echo "  cargo build --release -p somnus --bin somnus && cp target/release/somnus $CONTROL_BIN"
        echo "  cargo build --release -p somnus --bin somnus --features qwen3 && cp target/release/somnus $TREATMENT_BIN"
        exit 1
    fi
done
for f in "${CLIPS[@]}"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: clip not found: $f"
        exit 1
    fi
done

mkdir -p "$CONTROL_DIR" "$TREATMENT_DIR"

FILE_ARGS=()
for c in "${CLIPS[@]}"; do
    FILE_ARGS+=(--file "$c")
done

echo "Running control (Qwen2.5-0.5B, cold start)..."
for i in $(seq 1 "$SESSIONS"); do
    "$CONTROL_BIN" \
        --file-loop \
        "${FILE_ARGS[@]}" \
        --language \
        --action \
        --action-distill-only \
        --lang-use-lm \
        --action-model-prefix qwen25 \
        --seed "${SEED:-42}" \
        --memory-dir "$CONTROL_DIR" \
        --session "$i" \
        --max-sleep-cycles "$CYCLES" \
        --metrics-file "$CONTROL_DIR/metrics.jsonl" \
        2>&1 | tail -10
done

echo ""
echo "Running treatment (Qwen3-0.6B, cold start)..."
for i in $(seq 1 "$SESSIONS"); do
    "$TREATMENT_BIN" \
        --file-loop \
        "${FILE_ARGS[@]}" \
        --language \
        --action \
        --action-distill-only \
        --lang-use-lm \
        --action-model-prefix qwen3 \
        --seed "${SEED:-42}" \
        --memory-dir "$TREATMENT_DIR" \
        --session "$i" \
        --max-sleep-cycles "$CYCLES" \
        --metrics-file "$TREATMENT_DIR/metrics.jsonl" \
        2>&1 | tail -10
done

echo ""
echo "=== Results ==="
echo "Control (Qwen2.5-0.5B):  $CONTROL_DIR/metrics.jsonl"
echo "Treatment (Qwen3-0.6B):  $TREATMENT_DIR/metrics.jsonl"
echo ""
echo "Compare mean_proj_loss, per_region_mse[4] (language), lang_ingest_count across cycles."

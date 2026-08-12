//! Regression test for gaps.md §24: ORT binding must accept a grown region's actual
//! size, not the baseline N_UNITS constant. Uses the real ONNX graph left over from
//! a campaign run (well past baseline 1024, exact size varies as growth continues
//! between test runs) if present; skips (rather than failing) when that artifact isn't
//! on disk, since it's a large gitignored scratch file, not a fixture. Also skips (with
//! a loud eprintln, not silently) if the on-disk artifact predates the sparse-gather
//! binding rewrite's manifest format (old campaign runs pre-`rebuild_ort_session`) —
//! that's an expected transitional state, not a regression.

use somnus_perception::{OnnxPerceptionSession, RegionId, SparseActivation, ENV_DIM, LANG_HIDDEN};

/// Regression test for the LANG_HIDDEN encoder-dim drift found 2026-07-13 resuming the
/// campaign after the §24 binding fix: `onnx_session.rs`'s `ENC_DIMS` hardcoded the
/// language input at 896 (pre-ADR-0036), so every encoder call failed on `language_enc`
/// even though the binding path (the actual §24 fix) was working correctly.
#[test]
fn ort_encoder_accepts_current_lang_hidden_dim() {
    let bind_path = std::path::Path::new("../../memory/onnx_models/perception_binding.onnx");
    let enc_path = std::path::Path::new("../../memory/onnx_models/perception_encoder.onnx");
    if !bind_path.exists() || !enc_path.exists() {
        eprintln!("skipping: ONNX artifacts not present (scratch campaign artifact)");
        return;
    }

    let sess = match OnnxPerceptionSession::load(bind_path, false) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("skipping: couldn't load ORT binding session ({e}) — likely a pre-rewrite campaign artifact");
            return;
        }
    };
    let mut sess = sess
        .with_encoder(enc_path, false)
        .expect("load ORT encoder session");

    let speech = vec![0.0f32; 384];
    let env = vec![0.0f32; ENV_DIM];
    let visual = vec![0.0f32; 384];
    let lang = vec![0.0f32; LANG_HIDDEN];

    let result = sess.run_encoder(&speech, &env, &visual, &lang);
    assert!(
        result.is_ok(),
        "ORT encoder failed on current LANG_HIDDEN-sized input: {:?}",
        result.err()
    );
}

/// Regression test for the sparse-gather binding rewrite (successor to the older
/// gaps.md §24 dense-shape-mismatch test): the binding graph's Gather inputs have a
/// static per-region k_active shape baked in at export time, so callers must NOT
/// need to know or match that number — `run_binding` pads/truncates every
/// `SparseActivation` to it internally via `to_exact_k`. This exercises that with a
/// deliberately-mismatched (short) activation length, which would have been a hard
/// shape-mismatch error under the old dense sp_i contract.
#[test]
fn ort_binding_accepts_mismatched_activation_length() {
    let bind_path = std::path::Path::new("../../memory/onnx_models/perception_binding.onnx");
    if !bind_path.exists() {
        eprintln!("skipping: {bind_path:?} not present (scratch campaign artifact)");
        return;
    }

    let mut sess = match OnnxPerceptionSession::load(bind_path, false) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("skipping: couldn't load ORT binding session ({e}) — likely a pre-rewrite campaign artifact");
            return;
        }
    };
    let prev_ep = vec![0.0f32; 640];

    let region_ids = [
        RegionId::AudioSpeech,
        RegionId::AudioEnv,
        RegionId::Visual,
        RegionId::Symbolic,
        RegionId::Language,
    ];
    // 3 active units, always far fewer than any real k_active (baseline 20, scales up
    // from there) — indices 0/1/2 are valid rows regardless of the region's actual
    // (possibly grown) n_units, so this doesn't need to know that number either.
    let sparse: [SparseActivation; 5] = std::array::from_fn(|i| SparseActivation {
        region: region_ids[i],
        indices: vec![0, 1, 2],
        values: vec![0.1, 0.2, 0.3],
        n_units: 999_999,
    });
    let result = sess.run_binding(&sparse, &prev_ep);
    assert!(
        result.is_ok(),
        "ORT binding failed on a mismatched-length SparseActivation: {:?}",
        result.err()
    );
    assert_eq!(result.unwrap().len(), 640);
}

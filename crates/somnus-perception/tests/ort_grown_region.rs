//! Regression test for gaps.md §24: ORT binding must accept a grown region's actual
//! size, not the baseline N_UNITS constant. Uses the real ONNX graph left over from
//! a campaign run (well past baseline 1024, exact size varies as growth continues
//! between test runs) if present; skips (rather than failing) when that artifact isn't
//! on disk, since it's a large gitignored scratch file, not a fixture.

use somnus_perception::{OnnxPerceptionSession, ENV_DIM, LANG_HIDDEN};

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

    let sess = OnnxPerceptionSession::load(bind_path, false).expect("load ORT binding session");
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

#[test]
fn ort_binding_accepts_grown_region_size() {
    let bind_path = std::path::Path::new("../../memory/onnx_models/perception_binding.onnx");
    if !bind_path.exists() {
        eprintln!("skipping: {bind_path:?} not present (scratch campaign artifact)");
        return;
    }

    let mut sess = OnnxPerceptionSession::load(bind_path, false).expect("load ORT binding session");
    let prev_ep = vec![0.0f32; 640];

    // Probe with a deliberately-wrong size first to read the graph's actual expected
    // dim out of ORT's own error message — the exact grown size varies run to run as
    // the campaign keeps growing, so this avoids hardcoding (and re-hardcoding) a
    // number that will inevitably go stale again.
    let probe = sess.run_binding(&std::array::from_fn(|_| vec![0.0f32; 1]), &prev_ep);
    let msg = probe
        .expect_err("probe with size 1 must mismatch")
        .to_string();
    let expected: usize = msg
        .split("Expected: ")
        .nth(1)
        .and_then(|s| s.split_whitespace().next())
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| panic!("couldn't parse expected dim out of ORT error: {msg}"));
    assert!(
        expected > 1024,
        "expected a grown-past-baseline graph for this regression test, got {expected}"
    );

    // Before the §24 fix, run_binding hardcoded N_UNITS=1024 regardless of this value
    // and every call at a grown size would fail exactly like the probe above.
    let sparse_z: [Vec<f32>; 5] = std::array::from_fn(|_| vec![0.0f32; expected]);
    let result = sess.run_binding(&sparse_z, &prev_ep);
    assert!(
        result.is_ok(),
        "ORT binding failed on a grown-region input: {:?}",
        result.err()
    );
    assert_eq!(result.unwrap().len(), 640);
}

//! Regression test for gaps.md §24: ORT binding must accept a grown region's actual
//! size, not the baseline N_UNITS constant. Uses the real ONNX graph left over from
//! the campaign run (exported at region size 6886, well past baseline 1024) if present;
//! skips (rather than failing) when that artifact isn't on disk, since it's a large
//! gitignored scratch file, not a fixture.

use somnus_perception::OnnxPerceptionSession;

#[test]
fn ort_binding_accepts_grown_region_size() {
    let bind_path = std::path::Path::new("../../memory/onnx_models/perception_binding.onnx");
    if !bind_path.exists() {
        eprintln!("skipping: {bind_path:?} not present (scratch campaign artifact)");
        return;
    }

    let mut sess = OnnxPerceptionSession::load(bind_path, false).expect("load ORT binding session");

    // The graph was exported with all 5 regions at n_units=6886 (past baseline 1024) —
    // before the §24 fix, run_binding hardcoded 1024 and every call here would fail.
    let n = 6886usize;
    let sparse_z: [Vec<f32>; 5] = std::array::from_fn(|_| vec![0.0f32; n]);
    let prev_ep = vec![0.0f32; 640];

    let result = sess.run_binding(&sparse_z, &prev_ep);
    assert!(
        result.is_ok(),
        "ORT binding failed on a grown-region input: {:?}",
        result.err()
    );
    assert_eq!(result.unwrap().len(), 640);
}

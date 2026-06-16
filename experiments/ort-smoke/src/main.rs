use std::ffi::CString;
use std::path::PathBuf;

use ndarray::{Array3, Array4};
use ort::{AsPointer, session::Session, value::TensorRef};

fn cpu_session(model: impl AsRef<std::path::Path>) -> anyhow::Result<Session> {
    Ok(Session::builder()?.commit_from_file(model)?)
}

/// Register MIGraphX EP using the generic key-value API (same path as Python ORT).
/// The struct-based SessionOptionsAppendExecutionProvider_MIGraphX triggers MIGraphX's
/// internal problem-cache save with an empty path, which causes kernel execution to fail.
fn migraphx_session(model: impl AsRef<std::path::Path>) -> anyhow::Result<Session> {
    let mut builder = Session::builder()?;
    // Call the generic EP registration — identical to Python's providers=["MIGraphXExecutionProvider"]
    let provider_name = CString::new("MIGraphXExecutionProvider").unwrap();
    let keys: Vec<*const std::ffi::c_char> = vec![];
    let vals: Vec<*const std::ffi::c_char> = vec![];
    let status = unsafe {
        (ort::api().SessionOptionsAppendExecutionProvider)(
            builder.ptr_mut(),
            provider_name.as_ptr(),
            keys.as_ptr(),
            vals.as_ptr(),
            0,
        )
    };
    if !status.0.is_null() {
        // ORT returns a non-null OrtStatus* on error; log and fall through to CPU-only
        eprintln!("MIGraphX EP registration returned non-null status (may fall back to CPU)");
        unsafe { (ort::api().ReleaseStatus)(status.0) };
    }
    Ok(builder.commit_from_file(model)?)
}

fn main() -> anyhow::Result<()> {
    ort::init_from("/usr/lib/libonnxruntime.so")?.commit();

    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()   // experiments/
        .unwrap()
        .parent()   // repo root
        .unwrap()
        .to_owned();

    let models = repo_root.join("models");

    // ── Whisper-tiny encoder ──────────────────────────────────────────────────
    println!("=== whisper_tiny_encoder ===");
    {
        let mut session = migraphx_session(models.join("whisper_tiny_encoder.onnx"))?;
        for i in session.inputs()  { println!("  in  {:?} {:?}", i.name(), i.dtype()); }
        for o in session.outputs() { println!("  out {:?} {:?}", o.name(), o.dtype()); }

        let input: Array3<f32> = Array3::zeros((1, 80, 3000));
        let input_ref = TensorRef::from_array_view(&input)?;
        let outputs = session.run(ort::inputs!["input_features" => input_ref])?;
        let (shape, _) = outputs["last_hidden_state"].try_extract_tensor::<f32>()?;
        let dims: Vec<i64> = shape.iter().copied().collect();
        assert_eq!(dims, vec![1, 1500, 384], "whisper shape mismatch");
        println!("  output shape: {:?}  OK", dims);
    }

    // ── CLAP audio encoder ────────────────────────────────────────────────────
    println!("=== clap_audio_encoder ===");
    {
        let mut session = migraphx_session(models.join("clap_audio_encoder.onnx"))?;
        for i in session.inputs()  { println!("  in  {:?} {:?}", i.name(), i.dtype()); }
        for o in session.outputs() { println!("  out {:?} {:?}", o.name(), o.dtype()); }

        let input: Array4<f32> = Array4::zeros((1, 1, 1001, 64));
        let input_ref = TensorRef::from_array_view(&input)?;
        let outputs = session.run(ort::inputs!["input_features" => input_ref])?;
        let (shape, _) = outputs["audio_embeds"].try_extract_tensor::<f32>()?;
        let dims: Vec<i64> = shape.iter().copied().collect();
        assert_eq!(dims, vec![1, 512], "clap shape mismatch");
        println!("  output shape: {:?}  OK", dims);
    }

    // ── DINOv2-S/14 ──────────────────────────────────────────────────────────
    println!("=== dinov2_small ===");
    {
        let mut session = migraphx_session(models.join("dinov2_small.onnx"))?;
        for i in session.inputs()  { println!("  in  {:?} {:?}", i.name(), i.dtype()); }
        for o in session.outputs() { println!("  out {:?} {:?}", o.name(), o.dtype()); }

        let input: Array4<f32> = Array4::zeros((1, 3, 224, 224));
        let input_ref = TensorRef::from_array_view(&input)?;
        let outputs = session.run(ort::inputs!["pixel_values" => input_ref])?;
        let (shape, _) = outputs["last_hidden_state"].try_extract_tensor::<f32>()?;
        let dims: Vec<i64> = shape.iter().copied().collect();
        assert_eq!(dims, vec![1, 257, 384], "dinov2 shape mismatch");
        println!("  output shape: {:?}  OK", dims);
    }

    // ── Qwen2.5-0.5B LM (action-loop decoder, currently CPU-only in production) ─
    println!("=== qwen25_lm ===");
    {
        let mut session = migraphx_session(models.join("qwen25_lm.onnx"))?;
        for i in session.inputs()  { println!("  in  {:?} {:?}", i.name(), i.dtype()); }
        for o in session.outputs() { println!("  out {:?} {:?}", o.name(), o.dtype()); }

        // warm-up
        {
            let w: Array3<f32> = Array3::zeros((1, 8, 896));
            session.run(ort::inputs!["inputs_embeds" => TensorRef::from_array_view(&w)?])?;
        }
        println!("  MIGraphX (GPU):");
        for seq_len in [4usize, 8, 16, 32, 64] {
            let runs = 5;
            let t0 = std::time::Instant::now();
            for _ in 0..runs {
                let input: Array3<f32> = Array3::zeros((1, seq_len, 896));
                session.run(ort::inputs!["inputs_embeds" => TensorRef::from_array_view(&input)?])?;
            }
            println!("    seq_len={seq_len:3}  avg {:?}", t0.elapsed() / runs);
        }
    }
    println!("=== qwen25_lm (CPU) ===");
    {
        let lm_path = models.join("qwen25_lm.onnx");
        let mut session = cpu_session(&lm_path)?;
        // warm-up
        {
            let w: Array3<f32> = Array3::zeros((1, 8, 896));
            session.run(ort::inputs!["inputs_embeds" => TensorRef::from_array_view(&w)?])?;
        }
        println!("  CPU:");
        for seq_len in [4usize, 8, 16, 32, 64] {
            let runs = 5;
            let t0 = std::time::Instant::now();
            for _ in 0..runs {
                let input: Array3<f32> = Array3::zeros((1, seq_len, 896));
                session.run(ort::inputs!["inputs_embeds" => TensorRef::from_array_view(&input)?])?;
            }
            println!("    seq_len={seq_len:3}  avg {:?}", t0.elapsed() / runs);
        }
    }

    println!("\nAll encoders passed.");
    Ok(())
}

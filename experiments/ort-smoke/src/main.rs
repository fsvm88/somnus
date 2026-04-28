use std::path::PathBuf;

use ndarray::{Array3, Array4};
use ort::{session::Session, value::TensorRef};

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
        let mut session = Session::builder()?.commit_from_file(models.join("whisper_tiny_encoder.onnx"))?;
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
        let mut session = Session::builder()?.commit_from_file(models.join("clap_audio_encoder.onnx"))?;
        for i in session.inputs()  { println!("  in  {:?} {:?}", i.name(), i.dtype()); }
        for o in session.outputs() { println!("  out {:?} {:?}", o.name(), o.dtype()); }

        // (batch=1, channels=1, time=1001, mels=64)
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
        let mut session = Session::builder()?.commit_from_file(models.join("dinov2_small.onnx"))?;
        for i in session.inputs()  { println!("  in  {:?} {:?}", i.name(), i.dtype()); }
        for o in session.outputs() { println!("  out {:?} {:?}", o.name(), o.dtype()); }

        // (batch=1, channels=3, H=224, W=224)
        let input: Array4<f32> = Array4::zeros((1, 3, 224, 224));
        let input_ref = TensorRef::from_array_view(&input)?;
        let outputs = session.run(ort::inputs!["pixel_values" => input_ref])?;
        let (shape, _) = outputs["last_hidden_state"].try_extract_tensor::<f32>()?;
        let dims: Vec<i64> = shape.iter().copied().collect();
        assert_eq!(dims, vec![1, 257, 384], "dinov2 shape mismatch");
        println!("  output shape: {:?}  OK", dims);
    }

    println!("\nAll three encoders passed.");
    Ok(())
}

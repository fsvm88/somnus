use std::path::Path;

use somnus_sensory::{Timestamp, VideoFrame};
use ndarray::Array4;
use ort::{session::Session, value::TensorRef};

use crate::{
    audio::mean_pool_2d, preprocess::image::frame_to_tensor, session_with_migraphx, EncodingError,
    EncodingResult,
};

pub struct VideoEmbedding {
    pub values: Vec<f32>, // 384-dim, DINOv2 mean-pooled patch tokens
    pub timestamp: Timestamp,
    /// Wall time for this encode call (resize + normalise + ORT inference), in milliseconds.
    pub encode_ms: u64,
}

pub struct VideoEncoder {
    session: Session,
}

impl VideoEncoder {
    pub fn new(models_dir: &Path) -> EncodingResult<Self> {
        let session = session_with_migraphx(&models_dir.join("dinov2_small.onnx"))?;
        Ok(Self { session })
    }

    /// Encode one video frame → 384-dim DINOv2 embedding.
    ///
    /// Internally: converts to RGB24, resizes to 224×224, ImageNet-normalises,
    /// runs DINOv2, mean-pools the 256 patch tokens (skips CLS token 0).
    pub fn encode(&mut self, frame: &VideoFrame) -> EncodingResult<VideoEmbedding> {
        let t0 = std::time::Instant::now();
        let tensor_flat = frame_to_tensor(frame)?;
        // Shape: (1, 3, 224, 224)
        let arr = Array4::from_shape_vec((1, 3, 224, 224), tensor_flat)
            .map_err(|e| EncodingError::Image(e.to_string()))?;
        let tensor = TensorRef::from_array_view(&arr)?;

        let outputs = self.session.run(ort::inputs!["pixel_values" => tensor])?;
        let (shape, data) = outputs["last_hidden_state"].try_extract_tensor::<f32>()?;
        let dims: Vec<i64> = shape.iter().copied().collect();
        if dims != [1, 257, 384] {
            return Err(EncodingError::ShapeMismatch {
                expected: vec![1, 257, 384],
                got: dims,
            });
        }

        // Skip CLS token (index 0), mean-pool patch tokens [1..] → (256, 384)
        let flat = data.to_vec();
        let values = mean_pool_2d(&flat[384..], 256, 384); // skip CLS token (384 values)

        Ok(VideoEmbedding {
            values,
            timestamp: frame.timestamp.clone(),
            encode_ms: t0.elapsed().as_millis() as u64,
        })
    }
}

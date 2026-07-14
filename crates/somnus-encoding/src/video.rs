use std::path::Path;

use somnus_sensory::{Timestamp, VideoFrame};
use ndarray::Array4;
use ort::{session::Session, value::TensorRef};

use crate::{
    audio::mean_pool_2d, preprocess::image::frame_to_tensor, session_with_migraphx, EncodingError,
    EncodingResult,
};

pub struct VideoEmbedding {
    pub values: Vec<f32>, // 384-dim, DINOv2/DINOv3 mean-pooled patch tokens
    pub timestamp: Timestamp,
    /// Wall time for this encode call (resize + normalise + ORT inference), in milliseconds.
    pub encode_ms: u64,
}

pub struct VideoEncoder {
    session: Session,
    /// Non-patch tokens at the start of the sequence: 1 (CLS) for DINOv2,
    /// 1 + 4 register tokens for DINOv3 (order is [CLS, register x4, patch...],
    /// per `modeling_dinov3_vit.py`'s `torch.cat([cls_token, register_tokens, patch_embeddings])`).
    prefix_tokens: usize,
    total_tokens: usize,
}

impl VideoEncoder {
    pub fn new(models_dir: &Path) -> EncodingResult<Self> {
        // Prefer DINOv3 if available (drop-in replacement with same 384-dim output), else fall back to DINOv2.
        let dinov3_path = models_dir.join("dinov3_small.onnx");
        let (model_path, prefix_tokens, total_tokens) = if dinov3_path.exists() {
            (dinov3_path, 5, 201) // CLS + 4 register tokens + 196 patches (16x16 @ patch16)
        } else {
            (models_dir.join("dinov2_small.onnx"), 1, 257) // CLS + 256 patches (16x16 @ patch14)
        };
        let session = session_with_migraphx(&model_path)?;
        Ok(Self {
            session,
            prefix_tokens,
            total_tokens,
        })
    }

    /// Encode one video frame → 384-dim DINOv2/DINOv3 embedding.
    ///
    /// Internally: converts to RGB24, resizes to 224×224, ImageNet-normalises,
    /// runs the encoder, mean-pools the patch tokens (skips CLS/register prefix tokens).
    pub fn encode(&mut self, frame: &VideoFrame) -> EncodingResult<VideoEmbedding> {
        let (values, d0) = somnus_metrics::timed_core!(somnus_metrics::Metric::EncodeVideo, {
            let tensor_flat = frame_to_tensor(frame)?;
            // Shape: (1, 3, 224, 224)
            let arr = Array4::from_shape_vec((1, 3, 224, 224), tensor_flat)
                .map_err(|e| EncodingError::Image(e.to_string()))?;
            let tensor = TensorRef::from_array_view(&arr)?;

            let outputs = self.session.run(ort::inputs!["pixel_values" => tensor])?;
            let (shape, data) = outputs["last_hidden_state"].try_extract_tensor::<f32>()?;
            let dims: Vec<i64> = shape.iter().copied().collect();
            let expected = self.total_tokens as i64;
            if dims != [1, expected, 384] {
                return Err(EncodingError::ShapeMismatch {
                    expected: vec![1, expected, 384],
                    got: dims,
                });
            }

            // Skip CLS (+ register) prefix tokens, mean-pool the remaining patch tokens.
            let flat = data.to_vec();
            let num_patches = self.total_tokens - self.prefix_tokens;
            mean_pool_2d(&flat[self.prefix_tokens * 384..], num_patches, 384)
        });

        Ok(VideoEmbedding {
            values,
            timestamp: frame.timestamp.clone(),
            encode_ms: d0.as_millis() as u64,
        })
    }
}

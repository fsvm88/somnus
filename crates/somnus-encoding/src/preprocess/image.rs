use crate::{EncodingError, EncodingResult};
use somnus_sensory::{PixelFormat, VideoFrame};

const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];
const TARGET: usize = 224;

/// Convert a `VideoFrame` to a `(3, 224, 224)` f32 tensor, ImageNet-normalised,
/// channel-first. Matches the DINOv2 preprocessing in ADR 0004.
pub fn frame_to_tensor(frame: &VideoFrame) -> EncodingResult<Vec<f32>> {
    let rgb = to_rgb24(frame)?;
    let sw = frame.width as usize;
    let sh = frame.height as usize;

    // Convert u8 → f32 in [0, 1] before resizing, matching PyTorch's pipeline
    // (torch tensor / 255.0, then F.interpolate in float space).
    let rgb_f32: Vec<f32> = rgb.iter().map(|&b| b as f32 / 255.0).collect();

    // Bilinear resize in float space → (TARGET, TARGET, 3) interleaved
    let resized = bilinear_resize_f32(&rgb_f32, sw, sh, TARGET, TARGET);

    // Channel-first layout + ImageNet normalisation
    let mut out = vec![0.0f32; 3 * TARGET * TARGET];
    for c in 0..3 {
        for i in 0..TARGET * TARGET {
            out[c * TARGET * TARGET + i] =
                (resized[i * 3 + c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
        }
    }
    Ok(out)
}

fn to_rgb24(frame: &VideoFrame) -> EncodingResult<Vec<u8>> {
    match frame.format {
        PixelFormat::Rgb8 => Ok(frame.data.to_vec()),
        PixelFormat::Bgr8 => {
            let mut out = frame.data.to_vec();
            for px in out.chunks_exact_mut(3) {
                px.swap(0, 2);
            }
            Ok(out)
        }
        PixelFormat::Yuyv422 => Ok(yuyv_to_rgb(&frame.data)),
        fmt => Err(EncodingError::UnsupportedPixelFormat(fmt)),
    }
}

fn yuyv_to_rgb(yuyv: &[u8]) -> Vec<u8> {
    let n_pixels = yuyv.len() / 2;
    let mut rgb = Vec::with_capacity(n_pixels * 3);
    for chunk in yuyv.chunks_exact(4) {
        let (y0, u, y1, v) = (
            chunk[0] as f32,
            chunk[1] as f32,
            chunk[2] as f32,
            chunk[3] as f32,
        );
        let u = u - 128.0;
        let v = v - 128.0;
        for y in [y0, y1] {
            let r = (y + 1.402 * v).clamp(0.0, 255.0) as u8;
            let g = (y - 0.344_136 * u - 0.714_136 * v).clamp(0.0, 255.0) as u8;
            let b = (y + 1.772 * u).clamp(0.0, 255.0) as u8;
            rgb.extend_from_slice(&[r, g, b]);
        }
    }
    rgb
}

/// Bilinear resize in float space. `src` is HxWx3 f32 in [0, 1], interleaved RGB.
/// Matches PyTorch `F.interpolate(..., mode='bilinear', align_corners=False)`.
fn bilinear_resize_f32(src: &[f32], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<f32> {
    let mut dst = vec![0.0f32; dh * dw * 3];
    let x_scale = sw as f32 / dw as f32;
    let y_scale = sh as f32 / dh as f32;

    for dy in 0..dh {
        for dx in 0..dw {
            // align_corners=False: pixel centres map as (i+0.5)*scale - 0.5
            let sx = (dx as f32 + 0.5) * x_scale - 0.5;
            let sy = (dy as f32 + 0.5) * y_scale - 0.5;

            let x0 = (sx.floor() as isize).clamp(0, sw as isize - 1) as usize;
            let y0 = (sy.floor() as isize).clamp(0, sh as isize - 1) as usize;
            let x1 = (x0 + 1).min(sw - 1);
            let y1 = (y0 + 1).min(sh - 1);

            let wx = sx - sx.floor();
            let wy = sy - sy.floor();

            for c in 0..3 {
                let p00 = src[(y0 * sw + x0) * 3 + c];
                let p01 = src[(y0 * sw + x1) * 3 + c];
                let p10 = src[(y1 * sw + x0) * 3 + c];
                let p11 = src[(y1 * sw + x1) * 3 + c];
                dst[(dy * dw + dx) * 3 + c] = p00 * (1.0 - wx) * (1.0 - wy)
                    + p01 * wx * (1.0 - wy)
                    + p10 * (1.0 - wx) * wy
                    + p11 * wx * wy;
            }
        }
    }
    dst
}

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

// ── tests ────────────────────────────────────────────────────────────────────
//
// Rgb8 end-to-end parity against PyTorch fixtures lives in tests/mel_parity.rs
// (dino_* tests); these unit tests cover the pixel-format conversions and the
// resize/normalise pieces themselves.

#[cfg(test)]
mod tests {
    use super::*;
    use somnus_sensory::Timestamp;

    fn frame(data: Vec<u8>, w: u32, h: u32, fmt: PixelFormat) -> VideoFrame {
        VideoFrame {
            data,
            width: w,
            height: h,
            format: fmt,
            timestamp: Timestamp::now(),
        }
    }

    // ── YUYV conversion ──────────────────────────────────────────────────────

    #[test]
    fn yuyv_gray_neutral_chroma() {
        // U = V = 128 (neutral chroma): RGB == Y for both pixels in the macroblock.
        let rgb = yuyv_to_rgb(&[128, 128, 128, 128]);
        assert_eq!(rgb, [128, 128, 128, 128, 128, 128]);
        let rgb = yuyv_to_rgb(&[0, 128, 255, 128]);
        assert_eq!(rgb, [0, 0, 0, 255, 255, 255]);
    }

    #[test]
    fn yuyv_chroma_and_clamping() {
        // Pure chroma offsets: U=0 → blue-ish, V=255 → red-ish.
        let rgb = yuyv_to_rgb(&[128, 0, 128, 128]);
        // u = -128: r = 128, g = 128 + 0.344136·128 ≈ 172, b = 128 - 1.772·128 → clamped 0.
        assert_eq!(rgb[0], 128);
        assert_eq!(rgb[1], (128.0 + 0.344_136 * 128.0) as u8);
        assert_eq!(rgb[2], 0); // clamped
        let rgb = yuyv_to_rgb(&[128, 128, 128, 255]);
        // v = 127: r = 128 + 1.402·127 ≈ 306 → clamped 255.
        assert_eq!(rgb[0], 255); // clamped
        assert_eq!(rgb[1], (128.0 - 0.714_136 * 127.0) as u8);
        assert_eq!(rgb[2], 128);
    }

    // ── to_rgb24 / pixel formats ─────────────────────────────────────────────

    #[test]
    fn bgr8_swaps_channels() {
        // A uniform BGR frame must normalise to exactly the same tensor as the
        // equivalent RGB frame.
        let w: usize = 4;
        let bgr = frame(
            [10u8, 20, 30].repeat(w * w),
            w as u32,
            w as u32,
            PixelFormat::Bgr8,
        );
        let rgb = frame(
            [30u8, 20, 10].repeat(w * w),
            w as u32,
            w as u32,
            PixelFormat::Rgb8,
        );
        let a = frame_to_tensor(&bgr).unwrap();
        let b = frame_to_tensor(&rgb).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn yuyv422_gray_matches_rgb_gray() {
        // Neutral-chroma YUYV at Y=96 is identical to RGB gray 96.
        let w: usize = 4; // 2 macroblocks per row
        let mut yuyv = Vec::new();
        for _ in 0..w * w / 2 {
            yuyv.extend_from_slice(&[96, 128, 96, 128]);
        }
        let a = frame_to_tensor(&frame(yuyv, w as u32, w as u32, PixelFormat::Yuyv422)).unwrap();
        let b = frame_to_tensor(&frame(
            vec![96; w * w * 3],
            w as u32,
            w as u32,
            PixelFormat::Rgb8,
        ))
        .unwrap();
        assert_eq!(a.len(), 3 * TARGET * TARGET);
        assert_eq!(a, b);
    }

    #[test]
    fn unsupported_format_errors() {
        let f = frame(vec![0xFF; 8], 4, 4, PixelFormat::Mjpeg);
        match frame_to_tensor(&f) {
            Err(EncodingError::UnsupportedPixelFormat(PixelFormat::Mjpeg)) => {}
            other => panic!("expected UnsupportedPixelFormat, got {other:?}"),
        }
    }

    // ── normalisation ────────────────────────────────────────────────────────

    #[test]
    fn uniform_frame_normalises_per_channel() {
        // Uniform 224×224 RGB(200, 100, 50): every spatial position of channel c
        // must equal (v/255 - mean[c]) / std[c] exactly (resize of a constant is
        // the constant).
        let f = frame(
            [200u8, 100, 50].repeat(TARGET * TARGET),
            TARGET as u32,
            TARGET as u32,
            PixelFormat::Rgb8,
        );
        let out = frame_to_tensor(&f).unwrap();
        assert_eq!(out.len(), 3 * TARGET * TARGET);
        let vals = [200.0f32, 100.0, 50.0];
        for c in 0..3 {
            let expect = (vals[c] / 255.0 - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
            let plane = &out[c * TARGET * TARGET..(c + 1) * TARGET * TARGET];
            assert!(
                plane.iter().all(|&v| (v - expect).abs() < 1e-6),
                "channel {c} deviates from {expect}"
            );
        }
    }

    // ── bilinear resize ──────────────────────────────────────────────────────

    #[test]
    fn bilinear_identity_size_is_exact() {
        // scale = 1 maps pixel centres onto pixel centres: exact copy.
        let src: Vec<f32> = (0..8 * 8 * 3).map(|i| (i % 251) as f32 / 251.0).collect();
        let dst = bilinear_resize_f32(&src, 8, 8, 8, 8);
        assert_eq!(dst, src);
    }

    #[test]
    fn bilinear_constant_image_stays_constant() {
        let src = vec![0.25f32; 5 * 7 * 3];
        let dst = bilinear_resize_f32(&src, 7, 5, 16, 16);
        assert!(dst.iter().all(|&v| (v - 0.25).abs() < 1e-6));
    }

    #[test]
    fn bilinear_2x1_to_4x1_matches_pytorch_align_corners_false() {
        // PyTorch F.interpolate(align_corners=False), 2→4: source coords are
        // (dx+0.5)·0.5 − 0.5 = −0.25, 0.25, 0.75, 1.25; out-of-range indices are
        // clamped but the weights keep the extrapolated fractional part.
        let src = [0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0]; // pixels p0=0, p1=1
        let dst = bilinear_resize_f32(&src, 2, 1, 4, 1);
        let expect = [
            0.75, // sx = -0.25: x0 clamped to 0, wx = 0.75
            0.25, // sx = 0.25
            0.75, // sx = 0.75
            1.0,  // sx = 1.25: both neighbours clamp to p1 (edge replication)
        ];
        for (i, &e) in expect.iter().enumerate() {
            assert!(
                (dst[i * 3] - e).abs() < 1e-6,
                "dx={i}: {} vs {e}",
                dst[i * 3]
            );
        }
    }
}

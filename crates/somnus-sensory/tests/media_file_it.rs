//! Integration tests for `MediaFile` — requires `ffmpeg` CLI on PATH.
//!
//! Each test generates its own temp file via ffmpeg so tests can run in parallel.
//! The generated files are H.264/AAC MP4, 3 seconds, 320×240 at 25 fps,
//! 44100 Hz mono sine-wave audio.

use somnus_sensory::{MediaFile, PixelFormat, SensorError};
use std::process::Command;
use std::time::Duration;

const W: u32 = 320;
const H: u32 = 240;
const FPS: u32 = 25;
const DURATION_SECS: u32 = 3;
const SAMPLE_RATE: u32 = 44100;

/// Generate a synthetic MP4 with video (testsrc) and audio (sine wave) at `path`.
fn generate(path: &str) {
    let status = Command::new("ffmpeg")
        .args([
            "-y",
            "-f",
            "lavfi",
            "-i",
            &format!("testsrc=duration={DURATION_SECS}:size={W}x{H}:rate={FPS}"),
            "-f",
            "lavfi",
            "-i",
            &format!("sine=frequency=440:duration={DURATION_SECS}:sample_rate={SAMPLE_RATE}"),
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-shortest",
            path,
        ])
        .stderr(std::process::Stdio::null())
        .status()
        .expect("ffmpeg must be on PATH for integration tests");
    assert!(status.success(), "ffmpeg failed to generate {path}");
}

// ── error paths ───────────────────────────────────────────────────────────────

#[test]
fn open_missing_file_returns_error() {
    let err = MediaFile::open("/tmp/somnus_it_no_such_file_xyz.mp4").expect_err("should fail");
    assert!(
        matches!(err, SensorError::MediaFileOpen { .. }),
        "unexpected variant: {err:?}"
    );
}

// ── video decoding ────────────────────────────────────────────────────────────

#[test]
fn first_video_frame_has_correct_geometry() {
    let path = "/tmp/somnus_it_vid_geom.mp4";
    generate(path);

    let streams = MediaFile::open(path).unwrap().start().unwrap();

    let frame = streams
        .video
        .recv_timeout(Duration::from_secs(10))
        .expect("timed out waiting for first video frame");

    assert_eq!(frame.width, W);
    assert_eq!(frame.height, H);
    assert_eq!(frame.format, PixelFormat::Rgb8);
    assert_eq!(frame.data.len(), (W * H * 3) as usize);
}

#[test]
fn drains_approximately_correct_video_frame_count() {
    let path = "/tmp/somnus_it_vid_count.mp4";
    generate(path);

    let streams = MediaFile::open(path).unwrap().start().unwrap();

    let mut count = 0u32;
    while streams
        .video
        .recv_timeout(Duration::from_secs(10))
        .is_ok()
    {
        count += 1;
    }

    // 3 s × 25 fps = 75 frames. Allow ±5 for codec/flush rounding.
    let expected = FPS * DURATION_SECS;
    assert!(
        count >= expected - 5 && count <= expected + 5,
        "expected ~{expected} video frames, got {count}"
    );
}

#[test]
fn all_video_frame_pixels_are_rgb8_packed() {
    let path = "/tmp/somnus_it_vid_pixels.mp4";
    generate(path);

    let streams = MediaFile::open(path).unwrap().start().unwrap();

    // Spot-check the first 5 frames: correct buffer size and format.
    for i in 0..5 {
        let frame = streams
            .video
            .recv_timeout(Duration::from_secs(10))
            .unwrap_or_else(|_| panic!("timed out on frame {i}"));
        assert_eq!(frame.format, PixelFormat::Rgb8);
        assert_eq!(
            frame.data.len(),
            (frame.width * frame.height * 3) as usize,
            "frame {i}: data length mismatch"
        );
    }
}

// ── audio decoding ────────────────────────────────────────────────────────────

#[test]
fn first_audio_frame_has_correct_metadata() {
    let path = "/tmp/somnus_it_aud_meta.mp4";
    generate(path);

    let streams = MediaFile::open(path).unwrap().start().unwrap();

    let frame = streams
        .audio
        .recv_timeout(Duration::from_secs(10))
        .expect("timed out waiting for first audio frame");

    assert_eq!(frame.sample_rate, SAMPLE_RATE);
    assert_eq!(frame.channels, 1, "sine source + aac default is mono");
    assert!(!frame.samples.is_empty());
}

#[test]
fn audio_samples_are_finite() {
    let path = "/tmp/somnus_it_aud_finite.mp4";
    generate(path);

    let streams = MediaFile::open(path).unwrap().start().unwrap();

    // Check the first 10 audio frames for NaN / Inf.
    for i in 0..10 {
        let frame = streams
            .audio
            .recv_timeout(Duration::from_secs(10))
            .unwrap_or_else(|_| panic!("timed out on frame {i}"));
        for (j, &s) in frame.samples.iter().enumerate() {
            assert!(s.is_finite(), "frame {i} sample {j} is non-finite: {s}");
        }
    }
}

#[test]
fn audio_total_duration_matches_file_length() {
    let path = "/tmp/somnus_it_aud_duration.mp4";
    generate(path);

    let streams = MediaFile::open(path).unwrap().start().unwrap();

    // Accumulate samples across all audio frames until the channel closes.
    let mut total_samples = 0usize;
    while let Ok(frame) = streams.audio.recv_timeout(Duration::from_secs(10)) {
        total_samples += frame.frame_count();
    }

    // 3 s × 44100 Hz = 132300 sample frames. Allow ±5 % for encoder rounding.
    let expected = SAMPLE_RATE as usize * DURATION_SECS as usize;
    let tolerance = expected / 20; // 5 %
    assert!(
        total_samples >= expected - tolerance && total_samples <= expected + tolerance,
        "expected ~{expected} sample frames, got {total_samples}"
    );
}

// ── metrics ───────────────────────────────────────────────────────────────────

#[test]
fn video_metrics_captured_equals_frames_received_plus_dropped() {
    let path = "/tmp/somnus_it_vid_metrics.mp4";
    generate(path);

    let streams = MediaFile::open(path).unwrap().start().unwrap();

    let mut received = 0u64;
    while streams
        .video
        .recv_timeout(Duration::from_secs(10))
        .is_ok()
    {
        received += 1;
    }

    // When the channel disconnects the decode thread has already exited; metrics are stable.
    let snap = streams.video_metrics.snapshot();
    assert_eq!(
        snap.frames_captured,
        received + snap.frames_dropped,
        "captured != received + dropped"
    );
    // In this test the consumer keeps up so no frames should be dropped.
    assert_eq!(snap.frames_dropped, 0, "unexpected drops");
}

#[test]
fn audio_metrics_captured_equals_frames_received_plus_dropped() {
    let path = "/tmp/somnus_it_aud_metrics.mp4";
    generate(path);

    let streams = MediaFile::open(path).unwrap().start().unwrap();

    let mut received = 0u64;
    while streams
        .audio
        .recv_timeout(Duration::from_secs(10))
        .is_ok()
    {
        received += 1;
    }

    let snap = streams.audio_metrics.snapshot();
    assert_eq!(snap.frames_captured, received + snap.frames_dropped);
    assert_eq!(snap.frames_dropped, 0, "unexpected audio drops");
}

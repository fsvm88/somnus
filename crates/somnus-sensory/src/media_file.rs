use crossbeam_channel::{bounded, Receiver, Sender, TrySendError};
use ffmpeg::{
    channel_layout::ChannelLayout,
    codec,
    format::{self, sample::Type as SampleType, Pixel},
    frame,
    media::Type as MediaType,
    software::{resampling, scaling},
};
use ffmpeg_next as ffmpeg;
use std::{path::PathBuf, sync::Arc};

use crate::{
    AudioFrame, PixelFormat, SensorError, SensorMetrics, SensorResult, Timestamp, VideoFrame,
};

// ── public types ──────────────────────────────────────────────────────────────

/// Decoded streams from a media file. Frames arrive on the two receivers.
/// Dropping this disconnects the channels; the decode thread exits when it
/// next tries to send and detects `Disconnected`.
pub struct MediaFileStreams {
    /// Decoded audio: f32 PCM interleaved, sample rate from the file's audio track.
    pub audio: Receiver<AudioFrame>,
    /// Decoded video: Rgb8 packed, resolution from the file's video track.
    pub video: Receiver<VideoFrame>,
    pub audio_metrics: Arc<SensorMetrics>,
    pub video_metrics: Arc<SensorMetrics>,
    // JoinHandles dropped here detach the thread; it shuts down via Disconnected.
    _handles: MediaFileHandles,
}

/// A media file path that can be opened for decoding.
///
/// Phase 1 limitation: both audio and video streams must be present.
/// Audio-only or video-only files are rejected at `start()`.
#[derive(Debug)]
pub struct MediaFile {
    path: PathBuf,
}

impl MediaFile {
    /// Verify the path exists. Does not open the format context yet.
    pub fn open(path: impl Into<PathBuf>) -> SensorResult<Self> {
        let path = path.into();
        if !path.exists() {
            return Err(SensorError::MediaFileOpen {
                path,
                reason: "file not found".into(),
            });
        }
        Ok(Self { path })
    }

    /// Open the format context, initialise decoders and format converters, then
    /// spawn the decode thread. All initialisation errors are surfaced before the
    /// thread starts.
    pub fn start(self) -> SensorResult<MediaFileStreams> {
        ffmpeg::init().map_err(|e| SensorError::MediaFileDecode(e.to_string()))?;

        // ── format context ────────────────────────────────────────────────────
        let mut ictx = format::input(&self.path).map_err(|e| SensorError::MediaFileOpen {
            path: self.path.clone(),
            reason: e.to_string(),
        })?;

        // ── audio decoder ─────────────────────────────────────────────────────
        // Nested scope ends the borrow on ictx before we look up the video stream.
        let (audio_idx, mut audio_decoder) = {
            let stream = ictx
                .streams()
                .best(MediaType::Audio)
                .ok_or(SensorError::MediaFileNoStream("audio"))?;
            let idx = stream.index();
            let dec = codec::context::Context::from_parameters(stream.parameters())
                .map_err(|e| SensorError::MediaFileDecode(e.to_string()))?
                .decoder()
                .audio()
                .map_err(|e| SensorError::MediaFileDecode(e.to_string()))?;
            (idx, dec)
        };

        // Some containers omit channel layout (bits == 0); fall back to channel count.
        let channel_layout = if audio_decoder.channel_layout().bits() == 0 {
            match audio_decoder.channels() {
                1 => ChannelLayout::MONO,
                _ => ChannelLayout::STEREO,
            }
        } else {
            audio_decoder.channel_layout()
        };

        // Resample any source format to interleaved f32 at the file's own sample rate.
        // Same-rate conversion introduces no buffering delay.
        let mut audio_resampler = resampling::Context::get(
            audio_decoder.format(),
            channel_layout,
            audio_decoder.rate(),
            format::Sample::F32(SampleType::Packed),
            channel_layout,
            audio_decoder.rate(),
        )
        .map_err(|e| SensorError::MediaFileDecode(e.to_string()))?;

        let sample_rate = audio_decoder.rate();
        let channels = audio_decoder.channels() as u8;

        // ── video decoder ─────────────────────────────────────────────────────
        let (video_idx, mut video_decoder) = {
            let stream = ictx
                .streams()
                .best(MediaType::Video)
                .ok_or(SensorError::MediaFileNoStream("video"))?;
            let idx = stream.index();
            let dec = codec::context::Context::from_parameters(stream.parameters())
                .map_err(|e| SensorError::MediaFileDecode(e.to_string()))?
                .decoder()
                .video()
                .map_err(|e| SensorError::MediaFileDecode(e.to_string()))?;
            (idx, dec)
        };

        // Capture parameters needed to build the scaler inside the thread.
        // scaling::Context is not Send, so it must be created on the decode thread.
        let src_pixel_fmt = video_decoder.format();
        let width = video_decoder.width();
        let height = video_decoder.height();

        // ── channels ──────────────────────────────────────────────────────────
        let (audio_tx, audio_rx) = bounded::<AudioFrame>(256);
        let (video_tx, video_rx) = bounded::<VideoFrame>(16);

        // Drain clones: used inside the decode thread for drop-oldest backpressure.
        let audio_drain = audio_rx.clone();
        let video_drain = video_rx.clone();

        let audio_metrics = Arc::new(SensorMetrics::default());
        let video_metrics = Arc::new(SensorMetrics::default());
        let audio_metrics_t = Arc::clone(&audio_metrics);
        let video_metrics_t = Arc::clone(&video_metrics);

        // ── decode thread ─────────────────────────────────────────────────────
        let handle = std::thread::spawn(move || {
            // scaling::Context is not Send; build it here on the decode thread.
            let mut video_scaler = match scaling::Context::get(
                src_pixel_fmt,
                width,
                height,
                Pixel::RGB24,
                width,
                height,
                scaling::flag::Flags::BILINEAR,
            ) {
                Ok(s) => s,
                Err(e) => {
                    tracing::error!(?e, "failed to create video scaler");
                    return;
                }
            };

            // Reuse raw decoded frames to avoid per-packet allocation.
            // Converter output frames (conv) are allocated fresh each call because
            // frame sizes can vary across VFR and codec flush boundaries.
            let mut audio_raw = frame::Audio::empty();
            let mut video_raw = frame::Video::empty();

            // Aggregate the whole decode thread's lifetime (main demux loop + both flush
            // passes) into one metric per file, per this crate's own guardrail against
            // per-packet instrumentation at decode-thread frequency. The inner closure lets
            // the early `return`s below exit just this closure, not the timed_core! call's
            // own post-body recording (a bare `return` inside the block would otherwise skip
            // the metric write entirely for files that hit the early-return paths).
            #[allow(clippy::redundant_closure_call)]
            let (_, _) = somnus_metrics::timed_core!(
                somnus_metrics::Metric::MediaFileDecode,
                (|| {
                    // Main demux + decode loop.
                    for (stream, packet) in ictx.packets() {
                        if stream.index() == audio_idx {
                            if audio_decoder.send_packet(&packet).is_err() {
                                break;
                            }
                            while audio_decoder.receive_frame(&mut audio_raw).is_ok() {
                                let mut conv = frame::Audio::empty();
                                if audio_resampler.run(&audio_raw, &mut conv).is_err() {
                                    continue;
                                }
                                if conv.samples() == 0 {
                                    continue;
                                }
                                let af = AudioFrame {
                                    samples: f32_samples_from_frame(&conv),
                                    sample_rate,
                                    channels,
                                    timestamp: Timestamp::now(),
                                };
                                audio_metrics_t.increment_captured();
                                if !send_drop_oldest(&audio_tx, &audio_drain, af, &audio_metrics_t)
                                {
                                    return; // consumer gone; skip flush
                                }
                            }
                        } else if stream.index() == video_idx {
                            if video_decoder.send_packet(&packet).is_err() {
                                break;
                            }
                            while video_decoder.receive_frame(&mut video_raw).is_ok() {
                                let mut conv = frame::Video::empty();
                                if video_scaler.run(&video_raw, &mut conv).is_err() {
                                    continue;
                                }
                                let vf = VideoFrame {
                                    data: rgb24_from_frame(&conv),
                                    width,
                                    height,
                                    format: PixelFormat::Rgb8,
                                    timestamp: Timestamp::now(),
                                };
                                video_metrics_t.increment_captured();
                                if !send_drop_oldest(&video_tx, &video_drain, vf, &video_metrics_t)
                                {
                                    return;
                                }
                            }
                        }
                    }

                    // ── flush audio decoder ───────────────────────────────────────────
                    audio_decoder.send_eof().ok();
                    while audio_decoder.receive_frame(&mut audio_raw).is_ok() {
                        let mut conv = frame::Audio::empty();
                        if audio_resampler.run(&audio_raw, &mut conv).is_ok() && conv.samples() > 0
                        {
                            let af = AudioFrame {
                                samples: f32_samples_from_frame(&conv),
                                sample_rate,
                                channels,
                                timestamp: Timestamp::now(),
                            };
                            audio_metrics_t.increment_captured();
                            send_drop_oldest(&audio_tx, &audio_drain, af, &audio_metrics_t);
                        }
                    }

                    // Drain resampler residue. Same-rate resampler holds ≤ 1 frame; loop runs at most twice.
                    loop {
                        let mut conv = frame::Audio::empty();
                        match audio_resampler.flush(&mut conv) {
                            Ok(delay) => {
                                if conv.samples() > 0 {
                                    let af = AudioFrame {
                                        samples: f32_samples_from_frame(&conv),
                                        sample_rate,
                                        channels,
                                        timestamp: Timestamp::now(),
                                    };
                                    audio_metrics_t.increment_captured();
                                    send_drop_oldest(&audio_tx, &audio_drain, af, &audio_metrics_t);
                                }
                                if delay.is_none() {
                                    break;
                                }
                            }
                            Err(_) => break,
                        }
                    }

                    // ── flush video decoder ───────────────────────────────────────────
                    video_decoder.send_eof().ok();
                    while video_decoder.receive_frame(&mut video_raw).is_ok() {
                        let mut conv = frame::Video::empty();
                        if video_scaler.run(&video_raw, &mut conv).is_ok() {
                            let vf = VideoFrame {
                                data: rgb24_from_frame(&conv),
                                width,
                                height,
                                format: PixelFormat::Rgb8,
                                timestamp: Timestamp::now(),
                            };
                            video_metrics_t.increment_captured();
                            send_drop_oldest(&video_tx, &video_drain, vf, &video_metrics_t);
                        }
                    }
                    // Video scaler is stateless; no flush needed.
                })()
            );
        });

        Ok(MediaFileStreams {
            audio: audio_rx,
            video: video_rx,
            audio_metrics,
            video_metrics,
            _handles: MediaFileHandles {
                _decode_thread: handle,
            },
        })
    }
}

// ── private helpers ───────────────────────────────────────────────────────────

struct MediaFileHandles {
    _decode_thread: std::thread::JoinHandle<()>,
}

/// Extract interleaved f32 samples from an `F32(Packed)` audio frame (plane 0).
fn f32_samples_from_frame(frame: &frame::Audio) -> Vec<f32> {
    frame
        .data(0)
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_ne_bytes(*b))
        .collect()
}

/// Copy RGB24 pixels from a video frame, stripping any row padding.
/// `stride(0)` may exceed `width * 3` on some decoders/platforms.
fn rgb24_from_frame(frame: &frame::Video) -> Vec<u8> {
    let w = frame.width() as usize;
    let h = frame.height() as usize;
    let stride = frame.stride(0);
    let src = frame.data(0);
    if stride == w * 3 {
        src[..w * h * 3].to_vec()
    } else {
        let mut out = Vec::with_capacity(w * h * 3);
        for row in 0..h {
            out.extend_from_slice(&src[row * stride..row * stride + w * 3]);
        }
        out
    }
}

/// Send `frame` to `tx` with drop-oldest backpressure via the `drain` clone.
///
/// Returns `true` if the channel is still alive, `false` if the receiver dropped.
/// Single-producer: after draining one slot the retry send is always safe.
pub(crate) fn send_drop_oldest<T>(
    tx: &Sender<T>,
    drain: &Receiver<T>,
    frame: T,
    metrics: &SensorMetrics,
) -> bool {
    match tx.try_send(frame) {
        Ok(()) => true,
        Err(TrySendError::Full(frame)) => {
            let _ = drain.try_recv();
            metrics.increment_dropped();
            tx.try_send(frame).is_ok()
        }
        Err(TrySendError::Disconnected(_)) => false,
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_metrics() -> SensorMetrics {
        SensorMetrics::default()
    }

    // Helper: returns (tx, rx, drain) for a channel of capacity `cap`.
    fn make_channel<T>(cap: usize) -> (Sender<T>, Receiver<T>, Receiver<T>) {
        let (tx, rx) = bounded(cap);
        let drain = rx.clone();
        (tx, rx, drain)
    }

    #[test]
    fn send_drop_oldest_sends_when_space_available() {
        let (tx, rx, drain) = make_channel(4);
        let m = dummy_metrics();
        assert!(send_drop_oldest(&tx, &drain, 1u32, &m));
        assert_eq!(rx.try_recv().unwrap(), 1);
        assert_eq!(m.snapshot().frames_dropped, 0);
    }

    #[test]
    fn send_drop_oldest_evicts_oldest_when_full() {
        let (tx, rx, drain) = make_channel(1);
        let m = dummy_metrics();

        // Fill the channel.
        tx.try_send(10u32).unwrap();

        // Channel is full; send_drop_oldest should evict 10, send 20.
        assert!(send_drop_oldest(&tx, &drain, 20u32, &m));
        assert_eq!(m.snapshot().frames_dropped, 1);
        assert_eq!(rx.try_recv().unwrap(), 20);
    }

    #[test]
    fn send_drop_oldest_returns_false_when_disconnected() {
        let (tx, rx, drain) = make_channel(4);
        // Drop BOTH receivers to fully disconnect the channel.
        // crossbeam channels disconnect only when every Receiver clone is gone.
        drop(rx);
        drop(drain);
        // `drain` is gone; pass a dummy receiver from an unrelated channel.
        // It is only used in the Full branch, which is never reached when tx
        // is already disconnected — the first try_send returns Disconnected.
        let (_, dummy_rx) = bounded::<u32>(1);
        let m = dummy_metrics();
        let result = send_drop_oldest(&tx, &dummy_rx, 99u32, &m);
        assert!(!result);
    }
}

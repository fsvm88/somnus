use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::Sample;
use crossbeam_channel::{bounded, Receiver, Sender};
use std::sync::Arc;

use crate::{AudioFrame, SensorError, SensorMetrics, SensorResult, Timestamp};

/// Frames buffered in the crossbeam channel before the callback starts
/// dropping the oldest. At a typical 10 ms chunk period this is ~2.5 s
/// of headroom before any frame is lost.
pub const AUDIO_CHANNEL_CAPACITY: usize = 256;

/// Live microphone capture handle. Dropping this stops the cpal stream.
///
/// The stream runs on a cpal-managed platform thread. Frames arrive on
/// `receiver` normalised to `[-1.0, 1.0]` f32 regardless of the device's
/// native sample format.
pub struct AudioCapture {
    /// Held only for its `Drop` impl; releasing this stops playback.
    _stream: cpal::Stream,
    pub receiver: Receiver<AudioFrame>,
    pub metrics: Arc<SensorMetrics>,
}

impl AudioCapture {
    /// Start capturing from `device`, or the system default input if `None`.
    /// Returns once the stream is playing; frames arrive on `self.receiver`.
    pub fn start(device: Option<cpal::Device>) -> SensorResult<Self> {
        let host = cpal::default_host();
        let device = device
            .or_else(|| host.default_input_device())
            .ok_or(SensorError::NoAudioDevice)?;

        let config = device
            .default_input_config()
            .map_err(SensorError::AudioConfig)?;

        let metrics = Arc::new(SensorMetrics::default());
        let (tx, rx) = bounded::<AudioFrame>(AUDIO_CHANNEL_CAPACITY);

        let stream = Self::build_stream(&device, config, tx, rx.clone(), Arc::clone(&metrics))?;
        stream.play().map_err(SensorError::AudioPlay)?;

        Ok(Self {
            _stream: stream,
            receiver: rx,
            metrics,
        })
    }

    // `rx_drain` is a clone of the consumer receiver used only for drop-oldest eviction.
    fn build_stream(
        device: &cpal::Device,
        config: cpal::SupportedStreamConfig,
        tx: Sender<AudioFrame>,
        rx_drain: Receiver<AudioFrame>,
        metrics: Arc<SensorMetrics>,
    ) -> SensorResult<cpal::Stream> {
        let sample_rate = config.sample_rate();
        let channels = config.channels() as u8;
        let fmt = config.sample_format();
        let stream_config: cpal::StreamConfig = config.into();

        // Generates a typed input callback for SizedSample type $T.
        // f32::from_sample converts any integer or float sample to [-1.0, 1.0].
        // Both tx and rx_drain are cloned into each closure; crossbeam Receivers are Clone.
        macro_rules! make_cb {
            ($T:ty) => {{
                let tx = tx.clone();
                let rx_drain = rx_drain.clone();
                let metrics = Arc::clone(&metrics);
                move |data: &[$T], _: &cpal::InputCallbackInfo| {
                    let frame = AudioFrame {
                        samples: data.iter().map(|&s| f32::from_sample(s)).collect(),
                        sample_rate,
                        channels,
                        timestamp: Timestamp::now(),
                    };
                    metrics.increment_captured();
                    if let Err(crossbeam_channel::TrySendError::Full(frame)) = tx.try_send(frame) {
                        let _ = rx_drain.try_recv(); // evict oldest
                        metrics.increment_dropped();
                        tracing::trace!("audio channel full; dropped oldest frame");
                        let _ = tx.try_send(frame); // retry; can only fail if rx dropped
                    }
                }
            }};
        }

        let err_cb = |err| tracing::error!(?err, "audio stream error");

        let stream = match fmt {
            cpal::SampleFormat::I8 => {
                device.build_input_stream(stream_config, make_cb!(i8), err_cb, None)
            }
            cpal::SampleFormat::I16 => {
                device.build_input_stream(stream_config, make_cb!(i16), err_cb, None)
            }
            cpal::SampleFormat::I32 => {
                device.build_input_stream(stream_config, make_cb!(i32), err_cb, None)
            }
            cpal::SampleFormat::I64 => {
                device.build_input_stream(stream_config, make_cb!(i64), err_cb, None)
            }
            cpal::SampleFormat::U8 => {
                device.build_input_stream(stream_config, make_cb!(u8), err_cb, None)
            }
            cpal::SampleFormat::U16 => {
                device.build_input_stream(stream_config, make_cb!(u16), err_cb, None)
            }
            cpal::SampleFormat::U32 => {
                device.build_input_stream(stream_config, make_cb!(u32), err_cb, None)
            }
            cpal::SampleFormat::U64 => {
                device.build_input_stream(stream_config, make_cb!(u64), err_cb, None)
            }
            cpal::SampleFormat::F32 => {
                device.build_input_stream(stream_config, make_cb!(f32), err_cb, None)
            }
            cpal::SampleFormat::F64 => {
                device.build_input_stream(stream_config, make_cb!(f64), err_cb, None)
            }
            other => return Err(SensorError::UnsupportedSampleFormat(other)),
        };

        stream.map_err(SensorError::AudioBuild)
    }
}

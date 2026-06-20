/// All errors that can arise in the sensory pipeline.
#[derive(Debug, thiserror::Error)]
pub enum SensorError {
    #[error("audio device unavailable")]
    NoAudioDevice,

    #[error("audio build stream: {0}")]
    AudioBuild(#[source] cpal::Error),

    #[error("audio play stream: {0}")]
    AudioPlay(#[source] cpal::Error),

    #[error("audio default config: {0}")]
    AudioConfig(#[source] cpal::Error),

    #[error("video device {index} unavailable")]
    NoVideoDevice { index: u32 },

    #[error("video frame capture: {0}")]
    VideoCapture(String),

    #[error("media file open: {path}: {reason}")]
    MediaFileOpen {
        path: std::path::PathBuf,
        reason: String,
    },

    #[error("media file has no {0} stream")]
    MediaFileNoStream(&'static str),

    #[error("media file decode: {0}")]
    MediaFileDecode(String),

    #[error("sensor channel closed")]
    ChannelClosed,

    #[error("unsupported audio sample format: {0:?}")]
    UnsupportedSampleFormat(cpal::SampleFormat),
}

/// Convenience alias for `Result<T, SensorError>`.
pub type SensorResult<T> = Result<T, SensorError>;

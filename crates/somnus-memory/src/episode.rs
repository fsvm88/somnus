/// A single perceived episode: the binding area's output at one moment in time.
#[derive(Debug, Clone)]
pub struct EpisodeRecord {
    /// Monotonic session-local id assigned by EpisodicBuffer::push.
    pub id: u64,
    /// Wall-clock seconds since Unix epoch at the time of perception.
    pub timestamp_secs: f64,
    /// 512-dim episode vector from the binding area.
    pub values: Vec<f32>,
}

/// Hot-tier episode trace: the episode record plus the raw encoder inputs for all regions.
///
/// `regional[i]` is the encoder embedding that was fed into region i's k-WTA layer:
///   0 = AudioSpeech (384-dim), 1 = AudioEnv (512-dim), 2 = Visual (384-dim), 3 = Symbolic (384-dim).
/// Empty vecs are stored for regions that had no input when the episode was created.
///
/// The cold archive stores EpisodeRecord only; per-region encoder inputs are hot-tier data.
#[derive(Debug, Clone)]
pub struct EpisodeTrace {
    pub record: EpisodeRecord,
    pub regional: [Vec<f32>; 4],
}

/// A single perceived episode: the binding area's output at one moment in time.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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
///   0 = AudioSpeech (384-dim), 1 = AudioEnv (512-dim), 2 = Visual (384-dim),
///   3 = Symbolic (384-dim), 4 = Language (896-dim, empty when no text was present).
/// Empty vecs are stored for regions that had no input when the episode was created.
///
/// `text_target` carries token IDs from a text-bypass query received while this episode was
/// current. Used as the decoder projection training signal during NREM (ADR 0012).
/// Not persisted to the cold archive or snapshot; lost on restart (hot-tier only).
///
/// `salience` is the L2 distance to the nearest warm-memory neighbour at encode time.
/// High salience = novel (far from anything previously consolidated). Used to prioritise
/// replay: novel traces are replayed first so they are not skipped on early NREM exit.
/// Defaults to f32::MAX when warm memory is empty (first session).
///
/// The cold archive stores EpisodeRecord only; per-region encoder inputs are hot-tier data.
#[derive(Debug, Clone)]
pub struct EpisodeTrace {
    pub record: EpisodeRecord,
    pub regional: [Vec<f32>; 5],
    pub text_target: Option<Vec<u32>>,
    pub salience: f32,
}

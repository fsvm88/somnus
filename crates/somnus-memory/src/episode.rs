/// Which modality (or combination) produced the episode vector.
///
/// Derived at push time from which regional encoder inputs were non-empty.
/// Stored in the cold archive (1 byte per record) so cross-modal gist analysis
/// can run across sessions once cold→warm promotion is added.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[repr(u8)]
pub enum IngestSource {
    Speech = 0,
    Env = 1,
    Visual = 2,
    Language = 3,
    Mixed = 4,
    Unknown = 255,
}

impl IngestSource {
    pub fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::Speech,
            1 => Self::Env,
            2 => Self::Visual,
            3 => Self::Language,
            4 => Self::Mixed,
            _ => Self::Unknown,
        }
    }
}

/// A single perceived episode: the binding area's output at one moment in time.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EpisodeRecord {
    /// Monotonic session-local id assigned by EpisodicBuffer::push.
    pub id: u64,
    /// Wall-clock seconds since Unix epoch at the time of perception.
    pub timestamp_secs: f64,
    /// 640-dim episode vector from the binding area.
    pub values: Vec<f32>,
    /// Which modality (or combination) produced this episode.
    pub ingest_source: IngestSource,
    /// L2 distance to nearest warm-memory neighbour at push time.
    /// High = novel (far from consolidated memory). `f32::MAX` when warm memory is empty.
    /// Persisted in the cold archive (SMNCOLD3) to enable salience-weighted eviction.
    pub salience: f32,
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
/// The cold archive stores EpisodeRecord (including salience); per-region encoder inputs
/// are hot-tier data only.
#[derive(Debug, Clone)]
pub struct EpisodeTrace {
    pub record: EpisodeRecord,
    pub regional: [Vec<f32>; 5],
    pub text_target: Option<Vec<u32>>,
}

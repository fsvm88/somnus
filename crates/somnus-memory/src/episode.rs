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

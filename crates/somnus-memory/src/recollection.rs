use crate::{
    cold::ColdArchive, episode::EpisodeRecord, error::MemoryResult, semantic::SemanticStore,
};

/// Cross-tier recollection index: warm query + cold promotion stub.
///
/// Phase 4: only warm queries are live; cold promotion returns Ok(()) immediately.
/// Phase 5 will fill in cold→warm promotion driven by the sleep loop.
pub struct RecollectionLayer {
    pub warm: SemanticStore,
    pub cold: ColdArchive,
}

impl RecollectionLayer {
    pub fn new(warm: SemanticStore, cold: ColdArchive) -> Self {
        Self { warm, cold }
    }

    /// Return up to `k` most similar episodes from warm (semantic) memory.
    pub fn query_warm(&self, cue: &[f32], k: usize) -> Vec<&EpisodeRecord> {
        self.warm.query(cue, k)
    }

    /// Promote a record from cold to warm by id.
    ///
    /// Phase 4 stub — always succeeds without doing anything.
    /// Phase 5: locate record in cold archive, dequantise, insert into warm.
    #[allow(unused_variables)]
    pub fn promote(&mut self, id: u64) -> MemoryResult<()> {
        Ok(())
    }
}

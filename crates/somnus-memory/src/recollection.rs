use crate::{
    cold::ColdArchive, episode::EpisodeRecord, error::MemoryResult, semantic::SemanticStore,
};

/// Cross-tier recollection index: warm query + cold→warm promotion.
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

    /// Locate a cold record by id and insert it into warm semantic memory.
    ///
    /// Linear scan over the cold archive — acceptable for Phase 5 (cold len < 10k).
    /// Phase 6: build a cold-side id index to replace this scan.
    pub fn promote(&mut self, id: u64) -> MemoryResult<()> {
        let record = self
            .cold
            .iter()?
            .find_map(|r| match r {
                Ok(rec) if rec.id == id => Some(Ok(rec)),
                Ok(_) => None,
                Err(e) => Some(Err(e)),
            })
            .transpose()?;

        if let Some(rec) = record {
            self.warm.insert(rec);
        }
        Ok(())
    }
}

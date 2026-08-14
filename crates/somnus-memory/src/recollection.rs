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

    /// Like `query_warm` but also returns L2 distance for each result.
    pub fn query_warm_with_dist(&self, cue: &[f32], k: usize) -> Vec<(f32, &EpisodeRecord)> {
        self.warm.query_with_dist(cue, k)
    }

    /// Locate a cold record by id and insert it into warm semantic memory.
    ///
    /// Linear scan over the cold archive — acceptable for Phase 5 (cold len < 10k).
    /// Phase 6: build a cold-side id index to replace this scan.
    pub fn promote(&mut self, id: u64) -> MemoryResult<()> {
        let (result, _) = somnus_metrics::timed_core!(
            somnus_metrics::Metric::RecollectionPromote,
            self.promote_inner(id)
        );
        result
    }

    fn promote_inner(&mut self, id: u64) -> MemoryResult<()> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cold::{ColdArchive, COLD_DIM};
    use crate::episode::{EpisodeRecord, IngestSource};
    use tempfile::TempDir;

    fn make_layer() -> (RecollectionLayer, TempDir) {
        let dir = TempDir::new().unwrap();
        let mut cold = ColdArchive::open(dir.path()).unwrap();
        for i in 0..3u64 {
            cold.append(&EpisodeRecord {
                id: i,
                timestamp_secs: i as f64,
                values: vec![i as f32 * 0.25; COLD_DIM],
                ingest_source: IngestSource::Unknown,
                salience: 0.5,
            })
            .unwrap();
        }
        let warm = SemanticStore::new(COLD_DIM);
        (RecollectionLayer::new(warm, cold), dir)
    }

    #[test]
    fn promote_moves_cold_record_into_warm() {
        let (mut layer, _dir) = make_layer();
        assert!(layer.warm.is_empty());

        layer.promote(1).unwrap();
        assert_eq!(layer.warm.len(), 1);

        // The promoted record must be retrievable by similarity query.
        let cue = vec![0.25f32; COLD_DIM];
        let hits = layer.query_warm(&cue, 1);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, 1);

        // Cold archive is a ledger: promote copies, it does not remove.
        assert_eq!(layer.cold.len(), 3);
    }

    #[test]
    fn promote_unknown_id_is_noop() {
        let (mut layer, _dir) = make_layer();
        layer.promote(999).unwrap();
        assert!(layer.warm.is_empty());
    }

    #[test]
    fn query_warm_with_dist_orders_by_distance() {
        let (mut layer, _dir) = make_layer();
        layer.promote(0).unwrap();
        layer.promote(2).unwrap();

        // Cue near record 2's vector (0.5 fill).
        let cue = vec![0.49f32; COLD_DIM];
        let hits = layer.query_warm_with_dist(&cue, 2);
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].1.id, 2, "nearest record first");
        assert!(hits[0].0 <= hits[1].0, "distances ascending");
    }
}

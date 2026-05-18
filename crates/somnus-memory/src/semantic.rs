use std::collections::HashMap;

use hnsw_rs::prelude::{DistL2, Hnsw};

use crate::episode::EpisodeRecord;

const HNSW_M: usize = 16;
const HNSW_MAX_ELEM: usize = 100_000;
const HNSW_MAX_LAYERS: usize = 16;
const HNSW_EF_C: usize = 400;
const SEARCH_EF: usize = 64;

/// Warm-tier semantic store: HNSW index over episode vectors + sidecar record map.
///
/// Insert policy in Phase 4: eager — every perceived episode is inserted immediately.
/// Sleep-gated consolidation is Phase 5.
pub struct SemanticStore {
    index: Hnsw<'static, f32, DistL2>,
    records: HashMap<usize, EpisodeRecord>,
    dim: usize,
}

impl SemanticStore {
    pub fn new(dim: usize) -> Self {
        let index = Hnsw::<'static, f32, DistL2>::new(
            HNSW_M,
            HNSW_MAX_ELEM,
            HNSW_MAX_LAYERS,
            HNSW_EF_C,
            DistL2 {},
        );
        Self {
            index,
            records: HashMap::new(),
            dim,
        }
    }

    /// Insert an episode into the HNSW index and the sidecar map.
    pub fn insert(&mut self, record: EpisodeRecord) {
        let id = record.id as usize;
        self.index.insert((&record.values, id));
        self.records.insert(id, record);
    }

    /// Return up to `k` nearest episodes to `cue`, sorted by ascending L2 distance.
    pub fn query(&self, cue: &[f32], k: usize) -> Vec<&EpisodeRecord> {
        debug_assert_eq!(cue.len(), self.dim);
        let neighbours = self.index.search(cue, k, SEARCH_EF);
        neighbours
            .iter()
            .filter_map(|n| self.records.get(&n.d_id))
            .collect()
    }

    /// Like `query` but also returns the L2 distance for each result.
    pub fn query_with_dist(&self, cue: &[f32], k: usize) -> Vec<(f32, &EpisodeRecord)> {
        debug_assert_eq!(cue.len(), self.dim);
        let neighbours = self.index.search(cue, k, SEARCH_EF);
        neighbours
            .iter()
            .filter_map(|n| self.records.get(&n.d_id).map(|r| (n.distance, r)))
            .collect()
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Return up to `max_n` episode vectors, evenly sampled from the records map.
    /// Used to build k-means centroids for gist-level probing during REM.
    pub fn sample_values(&self, max_n: usize) -> Vec<Vec<f32>> {
        let records: Vec<&EpisodeRecord> = self.records.values().collect();
        if records.len() <= max_n {
            records.iter().map(|r| r.values.clone()).collect()
        } else {
            let step = records.len() / max_n;
            (0..max_n)
                .map(|i| records[i * step].values.clone())
                .collect()
        }
    }

    /// Expose the underlying HNSW index (for snapshot serialisation).
    pub fn hnsw(&self) -> &Hnsw<'static, f32, DistL2> {
        &self.index
    }

    /// Reconstruct from a pre-loaded HNSW index (restored from snapshot).
    /// The sidecar `records` map is empty; callers should repopulate it from
    /// the episodic buffer's records or accept reduced retrieval fidelity.
    pub fn from_hnsw(index: Hnsw<'static, f32, DistL2>, dim: usize) -> Self {
        Self {
            index,
            records: HashMap::new(),
            dim,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_record(id: u64, val: f32) -> EpisodeRecord {
        EpisodeRecord {
            id,
            timestamp_secs: id as f64,
            values: vec![val; 8],
        }
    }

    #[test]
    fn query_returns_exact_match() {
        let mut store = SemanticStore::new(8);
        let rec = make_record(0, 1.0);
        let cue = rec.values.clone();
        store.insert(rec);
        let results = store.query(&cue, 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, 0);
    }

    #[test]
    fn query_top_k_nearest() {
        let mut store = SemanticStore::new(8);
        for i in 0..10u64 {
            store.insert(make_record(i, i as f32));
        }
        // Record 5 is most similar to the cue built from val=5.0.
        let cue = vec![5.0f32; 8];
        let results = store.query(&cue, 3);
        assert!(!results.is_empty());
        assert_eq!(results[0].id, 5);
    }
}

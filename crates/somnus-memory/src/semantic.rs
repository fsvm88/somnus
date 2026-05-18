use std::{collections::HashMap, fs, path::Path};

use hnsw_rs::{api::AnnT, hnswio::HnswIo, prelude::{DistL2, Hnsw}};

use crate::episode::EpisodeRecord;

const WARM_MAGIC: &[u8; 4] = b"SMNW";
// Bump when HNSW_M, HNSW_MAX_LAYERS, or the sidecar format changes.
const WARM_VERSION: u16 = 1;

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

    /// Persist the warm HNSW index and record sidecar to `warm_dir`.
    ///
    /// Writes three files:
    ///   `version`      — 6-byte magic + u16 version tag
    ///   `index.hnsw.data` / `index.hnsw.graph` — hnsw_rs graph dump
    ///   `records.bin`  — bincode-encoded HashMap<usize, EpisodeRecord>
    pub fn dump(&self, warm_dir: &Path) -> Result<(), std::io::Error> {
        fs::create_dir_all(warm_dir)?;

        // Version tag
        let mut vbuf = [0u8; 6];
        vbuf[..4].copy_from_slice(WARM_MAGIC);
        vbuf[4..6].copy_from_slice(&WARM_VERSION.to_le_bytes());
        fs::write(warm_dir.join("version"), vbuf)?;

        // HNSW graph
        self.index
            .file_dump(warm_dir, "index")
            .map_err(|e| std::io::Error::other(e.to_string()))?;

        // Sidecar
        let encoded = bincode::serialize(&self.records)
            .map_err(|e| std::io::Error::other(e.to_string()))?;
        fs::write(warm_dir.join("records.bin"), encoded)?;

        Ok(())
    }

    /// Try to reload from a previously dumped warm index.
    ///
    /// Returns `None` (falling back to cold-rebuild) if:
    ///   - the version file is absent or version-mismatched
    ///   - any of the three data files are absent
    ///   - any file is corrupt
    ///
    /// Uses `Box::leak` on the `HnswIo` reader so the returned `Hnsw` is
    /// `'static`. The allocation is fixed per process restart (~a few KB).
    pub fn load_warm(warm_dir: &Path, episode_dim: usize) -> Option<Self> {
        // Version check
        let vbuf = fs::read(warm_dir.join("version")).ok()?;
        if vbuf.len() < 6
            || &vbuf[..4] != WARM_MAGIC
            || u16::from_le_bytes([vbuf[4], vbuf[5]]) != WARM_VERSION
        {
            return None;
        }

        // Check all three data files exist before attempting to load
        let data_path = warm_dir.join("index.hnsw.data");
        let graph_path = warm_dir.join("index.hnsw.graph");
        let sidecar_path = warm_dir.join("records.bin");
        if !data_path.exists() || !graph_path.exists() || !sidecar_path.exists() {
            return None;
        }

        // Sidecar records
        let sidecar_bytes = fs::read(&sidecar_path).ok()?;
        let records: HashMap<usize, EpisodeRecord> =
            bincode::deserialize(&sidecar_bytes).ok()?;

        // HNSW — leak HnswIo so the returned Hnsw<'static, ...> is valid
        let io = Box::new(HnswIo::new(warm_dir, "index"));
        let io_static: &'static mut HnswIo = Box::leak(io);
        let index: Hnsw<'static, f32, DistL2> = io_static.load_hnsw().ok()?;

        Some(Self { index, records, dim: episode_dim })
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
    fn warm_dump_load_round_trip() {
        let dir = tempfile::TempDir::new().unwrap();
        let warm_dir = dir.path().join("warm");
        let dim = 8;

        let mut store = SemanticStore::new(dim);
        for i in 0..10u64 {
            store.insert(make_record(i, i as f32));
        }
        store.dump(&warm_dir).expect("dump failed");

        let loaded = SemanticStore::load_warm(&warm_dir, dim).expect("load_warm returned None");
        assert_eq!(loaded.len(), 10);

        // Query should still work
        let cue = vec![5.0f32; dim];
        let results = loaded.query(&cue, 1);
        assert_eq!(results[0].id, 5);
    }

    #[test]
    fn warm_load_returns_none_on_missing() {
        let dir = tempfile::TempDir::new().unwrap();
        assert!(SemanticStore::load_warm(dir.path(), 8).is_none());
    }

    #[test]
    fn warm_load_returns_none_on_version_mismatch() {
        let dir = tempfile::TempDir::new().unwrap();
        let warm_dir = dir.path().join("warm");
        std::fs::create_dir_all(&warm_dir).unwrap();
        // Write a corrupt version file
        std::fs::write(warm_dir.join("version"), b"SMNW\xff\xff").unwrap();
        assert!(SemanticStore::load_warm(&warm_dir, 8).is_none());
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

use std::{collections::HashMap, fs, path::Path};

use hnsw_rs::{
    api::AnnT,
    hnswio::HnswIo,
    prelude::{DistL2, Hnsw},
};

use crate::episode::EpisodeRecord;

const WARM_MAGIC: &[u8; 4] = b"SMNW";
// Bump when HNSW_M, HNSW_MAX_LAYERS, or the sidecar format changes.
const WARM_VERSION: u16 = 1;

const HNSW_M: usize = 16;
const HNSW_MAX_ELEM: usize = 100_000;
const HNSW_MAX_LAYERS: usize = 16;
// Reduced from 400: lower ef_construction speeds up spill inserts and warm rebuilds
// (the dominant sleep-phase HNSW cost) at the cost of slightly weaker graph quality.
// Recall impact is measurable via warm_recall_l2 and gist_recall_rate in NDJSON (ADR 0022).
// Revisit if gist_recall_rate drops below acceptable threshold after a calibration run.
const HNSW_EF_C: usize = 100;
const SEARCH_EF: usize = 64;

/// Warm-tier semantic store: HNSW index over episode vectors + sidecar record map.
///
/// Insert policy in Phase 4: eager — every perceived episode is inserted immediately.
/// Sleep-gated consolidation is Phase 5.
///
/// `max_records`: when > 0, inserts are dropped once this limit is reached.
/// Use `rebuild_from_recent` during sleep to replace the index with a fresh one
/// covering only the most recent N records from cold (proper LRU eviction).
pub struct SemanticStore {
    index: Hnsw<'static, f32, DistL2>,
    records: HashMap<usize, EpisodeRecord>,
    dim: usize,
    max_records: usize,
    search_ef: usize,
}

impl SemanticStore {
    pub fn new(dim: usize) -> Self {
        Self::with_max(dim, 0)
    }

    pub fn with_max(dim: usize, max_records: usize) -> Self {
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
            max_records,
            search_ef: SEARCH_EF,
        }
    }

    pub fn set_search_ef(&mut self, ef: usize) {
        self.search_ef = ef;
    }

    pub fn set_max_records(&mut self, max: usize) {
        self.max_records = max;
    }

    /// Insert an episode into the HNSW index and the sidecar map.
    /// Silently skips when `max_records > 0` and the cap has been reached.
    /// New records still land in cold; use `rebuild_from_recent` to refresh warm.
    pub fn insert(&mut self, record: EpisodeRecord) {
        if self.max_records > 0 && self.records.len() >= self.max_records {
            return;
        }
        let id = record.id as usize;
        self.index.insert((&record.values, id));
        self.records.insert(id, record);
    }

    /// Replace warm with a fresh HNSW built from the most recent `n` records in `cold`.
    /// Called during sleep when warm has exceeded its cap. Returns the number of
    /// records loaded. Does nothing (returns 0) if cold is empty or n == 0.
    pub fn rebuild_from_recent(&mut self, cold: &crate::cold::ColdArchive, n: usize) -> usize {
        if n == 0 || cold.is_empty() {
            return 0;
        }
        let total = cold.len();
        let skip = total.saturating_sub(n);

        let mut fresh = SemanticStore::with_max(self.dim, self.max_records);
        fresh.search_ef = self.search_ef;
        let mut loaded = 0usize;
        // iter_from seeks directly to the tail offset — no soft-skip loop over earlier records.
        if let Ok(iter) = cold.iter_from(skip) {
            for record in iter.flatten() {
                fresh.index.insert((&record.values, record.id as usize));
                fresh.records.insert(record.id as usize, record);
                loaded += 1;
            }
        }
        *self = fresh;
        loaded
    }

    /// Return up to `k` nearest episodes to `cue`, sorted by ascending L2 distance.
    pub fn query(&self, cue: &[f32], k: usize) -> Vec<&EpisodeRecord> {
        debug_assert_eq!(cue.len(), self.dim);
        let neighbours = self.index.search(cue, k, self.search_ef);
        neighbours
            .iter()
            .filter_map(|n| self.records.get(&n.d_id))
            .collect()
    }

    /// Like `query` but also returns the L2 distance for each result.
    pub fn query_with_dist(&self, cue: &[f32], k: usize) -> Vec<(f32, &EpisodeRecord)> {
        debug_assert_eq!(cue.len(), self.dim);
        let neighbours = self.index.search(cue, k, self.search_ef);
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

    /// Return all records whose `timestamp_secs >= since_secs`.
    /// Used by the within-session gist probe to exclude prior-session embeddings.
    pub fn records_since(&self, since_secs: f64) -> Vec<&EpisodeRecord> {
        self.records
            .values()
            .filter(|r| r.timestamp_secs >= since_secs)
            .collect()
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

    /// Return up to `max_n` records, evenly sampled from the warm store.
    /// Used to build cross-session source histograms in the gist entropy probe.
    pub fn sample_records(&self, max_n: usize) -> Vec<&EpisodeRecord> {
        let records: Vec<&EpisodeRecord> = self.records.values().collect();
        if records.len() <= max_n {
            records
        } else {
            let step = records.len() / max_n;
            (0..max_n).map(|i| records[i * step]).collect()
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
            max_records: 0,
            search_ef: SEARCH_EF,
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
        let encoded =
            bincode::serialize(&self.records).map_err(|e| std::io::Error::other(e.to_string()))?;
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
        let records: HashMap<usize, EpisodeRecord> = bincode::deserialize(&sidecar_bytes).ok()?;

        // HNSW — leak HnswIo so the returned Hnsw<'static, ...> is valid
        let io = Box::new(HnswIo::new(warm_dir, "index"));
        let io_static: &'static mut HnswIo = Box::leak(io);
        let index: Hnsw<'static, f32, DistL2> = io_static.load_hnsw().ok()?;

        Some(Self {
            index,
            records,
            dim: episode_dim,
            max_records: 0,
            search_ef: SEARCH_EF,
        })
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
            ingest_source: crate::episode::IngestSource::Unknown,
            salience: 0.5,
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

    /// query_with_dist returns (distance, record) pairs with finite distances.
    /// Used by sleep.rs for the gist probe and salience computation.
    #[test]
    fn query_with_dist_returns_finite_distances() {
        let mut store = SemanticStore::new(8);
        for i in 0..5u64 {
            store.insert(make_record(i, i as f32 * 0.1));
        }
        let cue = vec![0.2f32; 8];
        let results = store.query_with_dist(&cue, 2);
        assert_eq!(results.len(), 2);
        for (dist, _rec) in &results {
            assert!(
                dist.is_finite() && *dist >= 0.0,
                "distance not finite/non-negative: {dist}"
            );
        }
        // Nearest neighbour should be closer than the second.
        assert!(
            results[0].0 <= results[1].0,
            "results not sorted by distance"
        );
    }

    /// sample_values returns at most max_n items and never panics on small stores.
    /// Used by sleep.rs gist k-means.
    #[test]
    fn sample_values_respects_limit() {
        let mut store = SemanticStore::new(8);
        for i in 0..20u64 {
            store.insert(make_record(i, i as f32));
        }
        let sample = store.sample_values(5);
        assert!(sample.len() <= 5, "sample exceeds max_n limit");
        assert!(!sample.is_empty());
        for v in &sample {
            assert_eq!(v.len(), 8);
        }
        // Also works on empty store.
        let empty = SemanticStore::new(8);
        assert!(empty.sample_values(10).is_empty());
    }

    /// rebuild_from_recent fills warm from cold, capped at n.
    /// Used in sleep.rs when warm HNSW exceeds max_warm_records.
    #[test]
    fn rebuild_from_recent_fills_from_cold() {
        use crate::cold::{ColdArchive, COLD_DIM};
        let dir = tempfile::TempDir::new().unwrap();
        let mut cold = ColdArchive::open(dir.path()).unwrap();
        // Cold archive requires COLD_DIM=640 vectors.
        for i in 0..10u64 {
            let rec = EpisodeRecord {
                id: i,
                timestamp_secs: i as f64,
                values: (0..COLD_DIM).map(|j| i as f32 + j as f32 * 0.001).collect(),
                ingest_source: crate::episode::IngestSource::Unknown,
                salience: 0.5,
            };
            cold.append(&rec).unwrap();
        }
        let mut store = SemanticStore::new(COLD_DIM);
        let loaded = store.rebuild_from_recent(&cold, 5);
        assert_eq!(loaded, 5, "should load exactly 5 (the cap)");
        assert_eq!(store.len(), 5);
    }
}

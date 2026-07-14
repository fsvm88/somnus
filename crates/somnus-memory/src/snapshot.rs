/// Snapshot / restore for the hot memory tier.
///
/// Snapshot directory layout:
///   memory/snapshot/
///     episodic.bin          — binary dump of EpisodicBuffer traces
///     weights.safetensors   — perception weights (written by the somnus binary)
///     region_sizes.bin      — 4 × u64 LE: n_units per region after growth
///
/// The SemanticStore (warm HNSW) is NOT persisted as a file. On restore it is
/// rebuilt by reinserting records from (a) the cold archive (all historically
/// spilled episodes) and (b) the hot episodic buffer (any records not yet
/// spilled). Cold records are inserted first (lower IDs), hot after.
///
/// After a completed sleep cycle the hot buffer is cleared before the snapshot
/// is saved, so on restore the episodic buffer is typically empty and warm
/// recall coverage comes entirely from the cold archive.
///
/// Written atomically: temp dir → rename.
use std::{
    fs,
    io::{BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
};

use crate::{
    cold::ColdArchive,
    episodic::EpisodicBuffer,
    error::{MemoryError, MemoryResult},
    semantic::SemanticStore,
};

// ── episodic.bin format ───────────────────────────────────────────────────────
//
// Header: magic(8B) + count(u64,8B) = 16B
// Per trace (little-endian):
//   id(u64,8) + ts(f64,8) + n_ep_vals(u32,4) + ep_values([f32; n_ep_vals]) +
//   for each of 5 regions: n_reg_vals(u16,2) + reg_values([f32; n_reg_vals]) +
//   ingest_source(u8,1)
//
// Bumped to SMNHOT01 in Phase 6 (ADR 0013) to distinguish 4-region (Phase 5)
// from 5-region (Phase 6) snapshots. Bumped to SMNHOT02 (Gap 8) to add
// ingest_source byte per trace. Old snapshots fail with BadMagic.
const EPISODIC_MAGIC: &[u8; 8] = b"SMNHOT02";

pub fn save_episodic(buf: &EpisodicBuffer, path: &Path) -> MemoryResult<()> {
    let (result, _) = somnus_metrics::timed_core!(
        somnus_metrics::Metric::SnapshotSaveEpisodic,
        save_episodic_inner(buf, path)
    );
    result
}

fn save_episodic_inner(buf: &EpisodicBuffer, path: &Path) -> MemoryResult<()> {
    let file = fs::File::create(path)?;
    let mut w = BufWriter::new(file);

    w.write_all(EPISODIC_MAGIC)?;
    let count = buf.len() as u64;
    w.write_all(&count.to_le_bytes())?;

    for trace in buf.iter_traces() {
        let r = &trace.record;
        w.write_all(&r.id.to_le_bytes())?;
        w.write_all(&r.timestamp_secs.to_le_bytes())?;
        let n = r.values.len() as u32;
        w.write_all(&n.to_le_bytes())?;
        for &v in &r.values {
            w.write_all(&v.to_le_bytes())?;
        }
        for reg in &trace.regional {
            let rn = reg.len() as u16;
            w.write_all(&rn.to_le_bytes())?;
            for &v in reg {
                w.write_all(&v.to_le_bytes())?;
            }
        }
        w.write_all(&[r.ingest_source as u8])?;
    }
    w.flush()?;
    Ok(())
}

pub fn load_episodic(path: &Path, capacity: usize) -> MemoryResult<EpisodicBuffer> {
    let (result, _) = somnus_metrics::timed_core!(
        somnus_metrics::Metric::SnapshotLoadEpisodic,
        load_episodic_inner(path, capacity)
    );
    result
}

fn load_episodic_inner(path: &Path, capacity: usize) -> MemoryResult<EpisodicBuffer> {
    let file = fs::File::open(path)?;
    let mut r = BufReader::new(file);

    let mut magic = [0u8; 8];
    r.read_exact(&mut magic)?;
    if &magic != EPISODIC_MAGIC {
        return Err(MemoryError::BadMagic);
    }

    let count = read_u64(&mut r)?;
    let mut traces = Vec::with_capacity(count as usize);
    let mut max_id = 0u64;

    for _ in 0..count {
        let id = read_u64(&mut r)?;
        let ts = read_f64(&mut r)?;
        let n = read_u32(&mut r)? as usize;
        let ep_values = read_f32_vec(&mut r, n)?;

        let mut regional: [Vec<f32>; 5] = [vec![], vec![], vec![], vec![], vec![]];
        for slot in regional.iter_mut() {
            let rn = read_u16(&mut r)? as usize;
            *slot = read_f32_vec(&mut r, rn)?;
        }

        let mut src_byte = [0u8; 1];
        r.read_exact(&mut src_byte)?;
        let ingest_source = crate::episode::IngestSource::from_u8(src_byte[0]);

        if id > max_id {
            max_id = id;
        }
        traces.push((id, ts, ep_values, regional, ingest_source));
    }

    let mut buf = EpisodicBuffer::new(capacity);
    // Replay all traces with correct ids via the pub(crate) id-aware push.
    for (id, ts, values, regional, ingest_source) in traces {
        buf.push_with_id(id, ts, values, regional, ingest_source);
    }
    buf.set_next_id(max_id + 1);
    Ok(buf)
}

// ── atomic snapshot write/read ────────────────────────────────────────────────

/// Write episodic buffer to `memory_dir/snapshot/` atomically.
/// Also rebuilds the SemanticStore from all episode records in the buffer
/// (the warm tier is reconstructed on load, not persisted separately).
pub fn save_snapshot(episodic: &EpisodicBuffer, memory_dir: &Path) -> MemoryResult<()> {
    let (result, _) = somnus_metrics::timed_core!(
        somnus_metrics::Metric::SnapshotSave,
        save_snapshot_inner(episodic, memory_dir)
    );
    result
}

fn save_snapshot_inner(episodic: &EpisodicBuffer, memory_dir: &Path) -> MemoryResult<()> {
    let tmp = memory_dir.join("snapshot.tmp");
    let snap = memory_dir.join("snapshot");

    fs::create_dir_all(&tmp)?;
    save_episodic(episodic, &tmp.join("episodic.bin"))?;
    // Persist the global next_id so IDs remain monotonic across restarts even
    // when the hot buffer is cleared (which would otherwise reset next_id to 1).
    fs::write(
        tmp.join("next_id.bin"),
        episodic.next_id_counter().to_le_bytes(),
    )?;

    if snap.exists() {
        fs::remove_dir_all(&snap)?;
    }
    fs::rename(&tmp, &snap)?;
    Ok(())
}

/// Path where perception weights should be saved.
///
/// The stem `weights` is used; Burn's BinFileRecorder appends `.bin` via
/// `PathBuf::set_extension("bin")` on every save and load, so the actual
/// file on disk is `weights.bin`. Callers that need to probe existence must
/// check `.with_extension("bin")` (or use this path directly with Burn APIs
/// which handle the extension internally).
pub fn weights_path(memory_dir: &Path) -> PathBuf {
    memory_dir.join("snapshot").join("weights")
}

/// Path where region sizes are saved.
pub fn region_sizes_path(memory_dir: &Path) -> PathBuf {
    memory_dir.join("snapshot").join("region_sizes.bin")
}

/// Save per-region n_units (5 × u64 LE). Written after growth so the next
/// restore knows which regions to pre-expand before loading weights.
pub fn save_region_sizes(sizes: [usize; 5], memory_dir: &Path) -> MemoryResult<()> {
    let path = region_sizes_path(memory_dir);
    let mut buf = [0u8; 40];
    for (i, &s) in sizes.iter().enumerate() {
        buf[i * 8..i * 8 + 8].copy_from_slice(&(s as u64).to_le_bytes());
    }
    fs::write(path, buf)?;
    Ok(())
}

/// Load per-region n_units. Returns None if the file is absent or malformed.
pub fn load_region_sizes(memory_dir: &Path) -> Option<[usize; 5]> {
    let data = fs::read(region_sizes_path(memory_dir)).ok()?;
    if data.len() < 40 {
        return None;
    }
    let mut sizes = [0usize; 5];
    for (i, s) in sizes.iter_mut().enumerate() {
        let bytes: [u8; 8] = data[i * 8..i * 8 + 8].try_into().ok()?;
        *s = u64::from_le_bytes(bytes) as usize;
    }
    Some(sizes)
}

/// Save the number of cycles remaining in the growth cooldown (single u64 LE).
pub fn save_growth_cooldown_remaining(remaining: usize, memory_dir: &Path) -> MemoryResult<()> {
    let path = memory_dir.join("snapshot").join("growth_cooldown.bin");
    fs::write(path, (remaining as u64).to_le_bytes())?;
    Ok(())
}

/// Load the growth cooldown remaining counter. Returns 0 if absent or malformed.
pub fn load_growth_cooldown_remaining(memory_dir: &Path) -> usize {
    let path = memory_dir.join("snapshot").join("growth_cooldown.bin");
    let data = fs::read(path).ok().unwrap_or_default();
    if data.len() < 8 {
        return 0;
    }
    let bytes: [u8; 8] = data[..8].try_into().unwrap_or([0; 8]);
    u64::from_le_bytes(bytes) as usize
}

/// Save the salience EWMA state (μ, σ as f32 LE; n as u64 LE = 16 bytes; ADR 0024 S1).
pub fn save_sal_ewma(mu: f32, sigma: f32, n: u64, memory_dir: &Path) -> MemoryResult<()> {
    let path = memory_dir.join("snapshot").join("sal_ewma.bin");
    let mut buf = [0u8; 16];
    buf[0..4].copy_from_slice(&mu.to_le_bytes());
    buf[4..8].copy_from_slice(&sigma.to_le_bytes());
    buf[8..16].copy_from_slice(&n.to_le_bytes());
    fs::write(path, buf)?;
    Ok(())
}

/// Load the salience EWMA state. Returns (0.8, 0.2, 0) on absent/malformed file.
pub fn load_sal_ewma(memory_dir: &Path) -> (f32, f32, u64) {
    let path = memory_dir.join("snapshot").join("sal_ewma.bin");
    let data = fs::read(path).ok().unwrap_or_default();
    if data.len() < 16 {
        return (0.8, 0.2, 0);
    }
    let mu = f32::from_le_bytes(data[0..4].try_into().unwrap_or([0; 4]));
    let sigma = f32::from_le_bytes(data[4..8].try_into().unwrap_or([0; 4]));
    let n = u64::from_le_bytes(data[8..16].try_into().unwrap_or([0; 8]));
    (mu, sigma, n)
}

/// Save the settle-cosine EWMA state (μ f32 LE + n u64 LE = 12 bytes; ADR 0024 S2).
pub fn save_settle_ewma(mu: f32, n: u64, memory_dir: &Path) -> MemoryResult<()> {
    let path = memory_dir.join("snapshot").join("settle_ewma.bin");
    let mut buf = [0u8; 12];
    buf[0..4].copy_from_slice(&mu.to_le_bytes());
    buf[4..12].copy_from_slice(&n.to_le_bytes());
    fs::write(path, buf)?;
    Ok(())
}

/// Load the settle-cosine EWMA state. Returns (0.85, 0) on absent/malformed file.
pub fn load_settle_ewma(memory_dir: &Path) -> (f32, u64) {
    let path = memory_dir.join("snapshot").join("settle_ewma.bin");
    let data = fs::read(path).ok().unwrap_or_default();
    if data.len() < 12 {
        return (0.85, 0);
    }
    let mu = f32::from_le_bytes(data[0..4].try_into().unwrap_or([0; 4]));
    let n = u64::from_le_bytes(data[4..12].try_into().unwrap_or([0; 8]));
    (mu, n)
}

/// Save the wake-duration EWMA state (μ f32 LE + n u64 LE = 12 bytes; ADR 0024 S6).
pub fn save_wake_ewma(mu: f32, n: u64, memory_dir: &Path) -> MemoryResult<()> {
    let path = memory_dir.join("snapshot").join("wake_ewma.bin");
    let mut buf = [0u8; 12];
    buf[0..4].copy_from_slice(&mu.to_le_bytes());
    buf[4..12].copy_from_slice(&n.to_le_bytes());
    fs::write(path, buf)?;
    Ok(())
}

/// Load the wake-duration EWMA state. Returns (0.0, 0) on absent/malformed file.
pub fn load_wake_ewma(memory_dir: &Path) -> (f32, u64) {
    let path = memory_dir.join("snapshot").join("wake_ewma.bin");
    let data = fs::read(path).ok().unwrap_or_default();
    if data.len() < 12 {
        return (0.0, 0);
    }
    let mu = f32::from_le_bytes(data[0..4].try_into().unwrap_or([0; 4]));
    let n = u64::from_le_bytes(data[4..12].try_into().unwrap_or([0; 8]));
    (mu, n)
}

/// Restore episodic buffer + rebuild SemanticStore from `memory_dir/snapshot/`.
///
/// Warm HNSW is loaded from the persisted dump in `memory_dir/warm/` when available
/// (fast path, < 5 s for 100 k records). Falls back to cold-rebuild when the dump is
/// absent, version-mismatched, or corrupt (original slow path, ~3 min at 164 k records).
///
/// Returns None if no snapshot exists.
pub fn load_snapshot(
    memory_dir: &Path,
    episodic_capacity: usize,
    episode_dim: usize,
) -> MemoryResult<Option<(EpisodicBuffer, SemanticStore)>> {
    let (result, _) = somnus_metrics::timed_core!(
        somnus_metrics::Metric::SnapshotLoad,
        load_snapshot_inner(memory_dir, episodic_capacity, episode_dim)
    );
    result
}

fn load_snapshot_inner(
    memory_dir: &Path,
    episodic_capacity: usize,
    episode_dim: usize,
) -> MemoryResult<Option<(EpisodicBuffer, SemanticStore)>> {
    let snap = memory_dir.join("snapshot");
    let episodic_path = snap.join("episodic.bin");
    if !episodic_path.exists() {
        return Ok(None);
    }

    let mut episodic = load_episodic(&episodic_path, episodic_capacity)?;

    // ── Fast path: load persisted warm index ─────────────────────────────────
    let warm_dir = memory_dir.join("warm");
    if let Some(semantic) = SemanticStore::load_warm(&warm_dir, episode_dim) {
        let warm_len = semantic.len();

        // Still need to insert any hot-buffer records that were not yet spilled.
        // (Typically empty after a completed sleep cycle, but handle the case.)
        // We rebuild a fresh SemanticStore to insert them after loading warm.
        // NOTE: load_warm already has the sidecar with all records; hot records
        // are appended here if present.
        let mut semantic = semantic;
        for record in episodic.iter_records() {
            semantic.insert(record.clone());
        }
        let hot_count = semantic.len() - warm_len;

        // Restore monotonic next_id.
        let persisted_next_id = {
            let path = snap.join("next_id.bin");
            fs::read(&path)
                .ok()
                .and_then(|b| b.try_into().ok())
                .map(u64::from_le_bytes)
                .unwrap_or(0)
        };
        let floor = [persisted_next_id, episodic.next_id_counter()]
            .into_iter()
            .max()
            .unwrap_or(0);
        if floor > episodic.next_id_counter() {
            episodic.set_next_id(floor);
        }

        log::info!(
            "warm HNSW loaded from dump: warm_records={warm_len} hot_records={hot_count} next_id={}",
            episodic.next_id_counter(),
        );
        return Ok(Some((episodic, semantic)));
    }

    // ── Slow path: rebuild from cold archive ──────────────────────────────────
    log::info!("warm dump absent or stale — rebuilding from cold archive");
    let mut semantic = SemanticStore::new(episode_dim);

    let mut cold_count = 0usize;
    let mut max_cold_id: u64 = 0;
    match ColdArchive::open(memory_dir) {
        Ok(archive) => match archive.iter() {
            Ok(iter) => {
                for result in iter {
                    match result {
                        Ok(record) => {
                            if record.id > max_cold_id {
                                max_cold_id = record.id;
                            }
                            semantic.insert(record);
                            cold_count += 1;
                        }
                        Err(e) => {
                            log::warn!("skipping corrupt cold record during warm rebuild: {e}");
                        }
                    }
                }
            }
            Err(e) => log::warn!("could not iterate cold archive during warm rebuild: {e}"),
        },
        Err(e) => log::warn!("could not open cold archive during warm rebuild: {e}"),
    }

    let mut hot_count = 0usize;
    for record in episodic.iter_records() {
        semantic.insert(record.clone());
        hot_count += 1;
    }

    let persisted_next_id = {
        let path = snap.join("next_id.bin");
        fs::read(&path)
            .ok()
            .and_then(|b| b.try_into().ok())
            .map(u64::from_le_bytes)
            .unwrap_or(0)
    };
    let floor = [
        persisted_next_id,
        if cold_count > 0 { max_cold_id + 1 } else { 0 },
        episodic.next_id_counter(),
    ]
    .into_iter()
    .max()
    .unwrap_or(0);
    if floor > episodic.next_id_counter() {
        episodic.set_next_id(floor);
    }

    log::info!(
        "warm HNSW rebuilt: cold_records={cold_count} hot_records={hot_count} total={} next_id={}",
        cold_count + hot_count,
        episodic.next_id_counter(),
    );

    Ok(Some((episodic, semantic)))
}

// ── read helpers ─────────────────────────────────────────────────────────────

fn read_u16(r: &mut impl Read) -> MemoryResult<u16> {
    let mut b = [0u8; 2];
    r.read_exact(&mut b)?;
    Ok(u16::from_le_bytes(b))
}

fn read_u32(r: &mut impl Read) -> MemoryResult<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64(r: &mut impl Read) -> MemoryResult<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn read_f64(r: &mut impl Read) -> MemoryResult<f64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(f64::from_le_bytes(b))
}

fn read_f32_vec(r: &mut impl Read, n: usize) -> MemoryResult<Vec<f32>> {
    let mut out = vec![0.0f32; n];
    for v in &mut out {
        let mut b = [0u8; 4];
        r.read_exact(&mut b)?;
        *v = f32::from_le_bytes(b);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn episodic_round_trip() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("episodic.bin");

        let mut buf = EpisodicBuffer::new(10);
        buf.push(
            1.0,
            vec![0.1f32; 640],
            [vec![1.0f32; 384], vec![2.0f32; 512], vec![], vec![], vec![]],
            f32::MAX,
            crate::episode::IngestSource::Unknown,
        );
        buf.push(
            2.0,
            vec![0.2f32; 640],
            [vec![], vec![], vec![3.0f32; 384], vec![4.0f32; 384], vec![]],
            f32::MAX,
            crate::episode::IngestSource::Visual,
        );

        save_episodic(&buf, &path).unwrap();
        let restored = load_episodic(&path, 10).unwrap();

        assert_eq!(restored.len(), 2);
        let traces: Vec<_> = restored.iter_traces().collect();
        assert_eq!(traces[0].record.timestamp_secs, 1.0);
        assert_eq!(traces[0].regional[0], vec![1.0f32; 384]);
        assert_eq!(traces[0].regional[1], vec![2.0f32; 512]);
        assert!(traces[0].regional[2].is_empty());
        assert_eq!(
            traces[0].record.ingest_source,
            crate::episode::IngestSource::Unknown
        );
        assert_eq!(traces[1].record.timestamp_secs, 2.0);
        assert_eq!(traces[1].regional[2], vec![3.0f32; 384]);
        assert_eq!(
            traces[1].record.ingest_source,
            crate::episode::IngestSource::Visual
        );
    }

    #[test]
    fn snapshot_round_trip() {
        let dir = TempDir::new().unwrap();
        let mut buf = EpisodicBuffer::new(4);
        buf.push(
            1.0,
            vec![0.1f32; 4],
            [vec![1.0f32; 2], vec![], vec![], vec![], vec![]],
            f32::MAX,
            crate::episode::IngestSource::Unknown,
        );
        buf.push(
            2.0,
            vec![0.2f32; 4],
            [vec![], vec![2.0f32; 2], vec![], vec![], vec![]],
            f32::MAX,
            crate::episode::IngestSource::Unknown,
        );

        save_snapshot(&buf, dir.path()).unwrap();
        let result = load_snapshot(dir.path(), 4, 4).unwrap();
        assert!(result.is_some());
        let (restored_buf, restored_sem) = result.unwrap();
        assert_eq!(restored_buf.len(), 2);
        assert_eq!(restored_sem.len(), 2);
        // next_id should be at least 2 (the two records had ids 0 and 1).
        assert!(restored_buf.next_id_counter() >= 2);
    }

    #[test]
    fn warm_rebuilt_from_cold_after_hot_cleared() {
        use crate::cold::{ColdArchive, COLD_DIM};
        let dir = TempDir::new().unwrap();
        let dim = COLD_DIM;

        // Simulate: records spilled to cold, then hot buffer cleared, snapshot saved empty.
        let mut archive = ColdArchive::open(dir.path()).unwrap();
        for i in 0..3u64 {
            archive
                .append(&crate::episode::EpisodeRecord {
                    id: i,
                    timestamp_secs: i as f64,
                    values: vec![i as f32 * 0.1; dim],
                    ingest_source: crate::episode::IngestSource::Unknown,
                    salience: 0.5,
                })
                .unwrap();
        }
        // Save snapshot with an empty hot buffer (post-clear state).
        let empty_buf = EpisodicBuffer::new(4);
        save_snapshot(&empty_buf, dir.path()).unwrap();

        let result = load_snapshot(dir.path(), 4, dim).unwrap();
        assert!(result.is_some());
        let (restored_buf, restored_sem) = result.unwrap();
        assert_eq!(restored_buf.len(), 0, "hot buffer should be empty");
        // Warm HNSW must be populated from cold records despite empty hot buffer.
        assert_eq!(
            restored_sem.len(),
            3,
            "warm HNSW should contain all 3 cold records"
        );
        // next_id must be above the max cold ID (2) so new records don't collide.
        assert!(
            restored_buf.next_id_counter() > 2,
            "next_id should be above max cold ID"
        );
    }
}

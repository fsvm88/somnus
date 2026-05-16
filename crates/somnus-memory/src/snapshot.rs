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
//   for each of 4 regions: n_reg_vals(u16,2) + reg_values([f32; n_reg_vals])

const EPISODIC_MAGIC: &[u8; 8] = b"SMNHOT\0\0";

pub fn save_episodic(buf: &EpisodicBuffer, path: &Path) -> MemoryResult<()> {
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
    }
    w.flush()?;
    Ok(())
}

pub fn load_episodic(path: &Path, capacity: usize) -> MemoryResult<EpisodicBuffer> {
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

        let mut regional: [Vec<f32>; 4] = [vec![], vec![], vec![], vec![]];
        for slot in regional.iter_mut() {
            let rn = read_u16(&mut r)? as usize;
            *slot = read_f32_vec(&mut r, rn)?;
        }

        if id > max_id {
            max_id = id;
        }
        traces.push((id, ts, ep_values, regional));
    }

    let mut buf = EpisodicBuffer::new(capacity);
    // Replay all traces with correct ids via the pub(crate) id-aware push.
    for (id, ts, values, regional) in traces {
        buf.push_with_id(id, ts, values, regional);
    }
    buf.set_next_id(max_id + 1);
    Ok(buf)
}

// ── atomic snapshot write/read ────────────────────────────────────────────────

/// Write episodic buffer to `memory_dir/snapshot/` atomically.
/// Also rebuilds the SemanticStore from all episode records in the buffer
/// (the warm tier is reconstructed on load, not persisted separately).
pub fn save_snapshot(episodic: &EpisodicBuffer, memory_dir: &Path) -> MemoryResult<()> {
    let tmp = memory_dir.join("snapshot.tmp");
    let snap = memory_dir.join("snapshot");

    fs::create_dir_all(&tmp)?;
    save_episodic(episodic, &tmp.join("episodic.bin"))?;

    if snap.exists() {
        fs::remove_dir_all(&snap)?;
    }
    fs::rename(&tmp, &snap)?;
    Ok(())
}

/// Path where perception weights should be saved.
pub fn weights_path(memory_dir: &Path) -> PathBuf {
    memory_dir.join("snapshot").join("weights.safetensors")
}

/// Path where region sizes are saved.
pub fn region_sizes_path(memory_dir: &Path) -> PathBuf {
    memory_dir.join("snapshot").join("region_sizes.bin")
}

/// Save per-region n_units (4 × u64 LE). Written after growth so the next
/// restore knows which regions to pre-expand before loading weights.
pub fn save_region_sizes(sizes: [usize; 4], memory_dir: &Path) -> MemoryResult<()> {
    let path = region_sizes_path(memory_dir);
    let mut buf = [0u8; 32];
    for (i, &s) in sizes.iter().enumerate() {
        buf[i * 8..i * 8 + 8].copy_from_slice(&(s as u64).to_le_bytes());
    }
    fs::write(path, buf)?;
    Ok(())
}

/// Load per-region n_units. Returns None if the file is absent or malformed.
pub fn load_region_sizes(memory_dir: &Path) -> Option<[usize; 4]> {
    let data = fs::read(region_sizes_path(memory_dir)).ok()?;
    if data.len() < 32 {
        return None;
    }
    let mut sizes = [0usize; 4];
    for (i, s) in sizes.iter_mut().enumerate() {
        let bytes: [u8; 8] = data[i * 8..i * 8 + 8].try_into().ok()?;
        *s = u64::from_le_bytes(bytes) as usize;
    }
    Some(sizes)
}

/// Restore episodic buffer + rebuild SemanticStore from `memory_dir/snapshot/`.
///
/// Warm HNSW is rebuilt by inserting cold-archive records first (all historical
/// spilled episodes), then hot-buffer records (anything not yet spilled).
/// Cold records have lower IDs by construction; inserting cold first avoids ID
/// collisions in the common case. Returns None if no snapshot exists.
pub fn load_snapshot(
    memory_dir: &Path,
    episodic_capacity: usize,
    episode_dim: usize,
) -> MemoryResult<Option<(EpisodicBuffer, SemanticStore)>> {
    let snap = memory_dir.join("snapshot");
    let episodic_path = snap.join("episodic.bin");
    if !episodic_path.exists() {
        return Ok(None);
    }

    let episodic = load_episodic(&episodic_path, episodic_capacity)?;
    let mut semantic = SemanticStore::new(episode_dim);

    // Rebuild warm from cold archive first (all historically spilled records).
    let mut cold_count = 0usize;
    match ColdArchive::open(memory_dir) {
        Ok(archive) => match archive.iter() {
            Ok(iter) => {
                for result in iter {
                    match result {
                        Ok(record) => {
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

    // Then insert hot-buffer records (not yet spilled; typically empty after sleep).
    let mut hot_count = 0usize;
    for record in episodic.iter_records() {
        semantic.insert(record.clone());
        hot_count += 1;
    }

    log::info!(
        "warm HNSW rebuilt: cold_records={cold_count} hot_records={hot_count} total={}",
        cold_count + hot_count
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
            vec![0.1f32; 512],
            [vec![1.0f32; 384], vec![2.0f32; 512], vec![], vec![]],
        );
        buf.push(
            2.0,
            vec![0.2f32; 512],
            [vec![], vec![], vec![3.0f32; 384], vec![4.0f32; 384]],
        );

        save_episodic(&buf, &path).unwrap();
        let restored = load_episodic(&path, 10).unwrap();

        assert_eq!(restored.len(), 2);
        let traces: Vec<_> = restored.iter_traces().collect();
        assert_eq!(traces[0].record.timestamp_secs, 1.0);
        assert_eq!(traces[0].regional[0], vec![1.0f32; 384]);
        assert_eq!(traces[0].regional[1], vec![2.0f32; 512]);
        assert!(traces[0].regional[2].is_empty());
        assert_eq!(traces[1].record.timestamp_secs, 2.0);
        assert_eq!(traces[1].regional[2], vec![3.0f32; 384]);
    }

    #[test]
    fn snapshot_round_trip() {
        let dir = TempDir::new().unwrap();
        let mut buf = EpisodicBuffer::new(4);
        buf.push(
            1.0,
            vec![0.1f32; 4],
            [vec![1.0f32; 2], vec![], vec![], vec![]],
        );
        buf.push(
            2.0,
            vec![0.2f32; 4],
            [vec![], vec![2.0f32; 2], vec![], vec![]],
        );

        save_snapshot(&buf, dir.path()).unwrap();
        let result = load_snapshot(dir.path(), 4, 4).unwrap();
        assert!(result.is_some());
        let (restored_buf, restored_sem) = result.unwrap();
        assert_eq!(restored_buf.len(), 2);
        assert_eq!(restored_sem.len(), 2);
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
    }
}

use std::{
    fs::{File, OpenOptions},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
};

use memmap2::Mmap;

use crate::{
    episode::EpisodeRecord,
    error::{MemoryError, MemoryResult},
};

/// Dimension of episode vectors — must match `somnus-perception::binding::EPISODE_DIM`.
pub const COLD_DIM: usize = 512;

const MAGIC: &[u8; 8] = b"SMNCOLD\0";

// Per-record layout (little-endian):
//   id:        u64   8 B
//   timestamp: f64   8 B
//   scale:     f32   4 B
//   values:    [i8; COLD_DIM]  512 B
//   ──────────────────────
//   total:            532 B
const RECORD_BYTES: usize = 8 + 8 + 4 + COLD_DIM;
const HEADER_BYTES: usize = 8; // just the magic

/// Append-only cold archive for episode vectors.
///
/// Vectors are INT8-quantised with a per-vector scale factor.
/// In Phase 4 `append` is implemented but not called from the main loop;
/// cold→warm promotion is Phase 5 work.
pub struct ColdArchive {
    path: PathBuf,
    len: usize,
}

impl ColdArchive {
    /// Open (or create) the archive file.
    pub fn open(dir: &Path) -> MemoryResult<Self> {
        std::fs::create_dir_all(dir)?;
        let path = dir.join("episodes.cold");
        if path.exists() {
            let meta = std::fs::metadata(&path)?;
            let file_len = meta.len() as usize;
            if file_len < HEADER_BYTES {
                return Err(MemoryError::BadMagic);
            }
            // Verify magic.
            let file = File::open(&path)?;
            let mmap = unsafe { Mmap::map(&file)? };
            if &mmap[..8] != MAGIC {
                return Err(MemoryError::BadMagic);
            }
            let body = file_len - HEADER_BYTES;
            let len = body / RECORD_BYTES;
            Ok(Self { path, len })
        } else {
            // Create and write header.
            let mut w = BufWriter::new(File::create(&path)?);
            w.write_all(MAGIC)?;
            w.flush()?;
            Ok(Self { path, len: 0 })
        }
    }

    /// Quantise and append one episode record.
    pub fn append(&mut self, record: &EpisodeRecord) -> MemoryResult<()> {
        if record.values.len() != COLD_DIM {
            return Err(MemoryError::DimMismatch {
                expected: COLD_DIM,
                got: record.values.len(),
            });
        }
        let max_abs = record.values.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
        let quantised: Vec<i8> = record
            .values
            .iter()
            .map(|&v| (v / scale).round().clamp(-127.0, 127.0) as i8)
            .collect();

        let mut file = OpenOptions::new().append(true).open(&self.path)?;
        let mut w = BufWriter::new(&mut file);
        w.write_all(&record.id.to_le_bytes())?;
        w.write_all(&record.timestamp_secs.to_le_bytes())?;
        w.write_all(&scale.to_le_bytes())?;
        // Safety: i8 and u8 have same size/alignment; transmuting to &[u8] for writing is fine.
        w.write_all(unsafe {
            std::slice::from_raw_parts(quantised.as_ptr() as *const u8, COLD_DIM)
        })?;
        w.flush()?;
        self.len += 1;
        Ok(())
    }

    /// Iterate all records, dequantising each vector.
    pub fn iter(&self) -> MemoryResult<ColdIter> {
        let file = File::open(&self.path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        Ok(ColdIter {
            mmap,
            pos: HEADER_BYTES,
            len: self.len,
            yielded: 0,
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

pub struct ColdIter {
    mmap: Mmap,
    pos: usize,
    len: usize,
    yielded: usize,
}

impl Iterator for ColdIter {
    type Item = MemoryResult<EpisodeRecord>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.yielded >= self.len {
            return None;
        }
        let end = self.pos + RECORD_BYTES;
        if end > self.mmap.len() {
            return Some(Err(MemoryError::CorruptRecord {
                offset: self.pos as u64,
            }));
        }
        let bytes = &self.mmap[self.pos..end];
        let id = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
        let ts = f64::from_le_bytes(bytes[8..16].try_into().unwrap());
        let scale = f32::from_le_bytes(bytes[16..20].try_into().unwrap());
        let qbytes = &bytes[20..20 + COLD_DIM];
        let values: Vec<f32> = qbytes.iter().map(|&b| (b as i8) as f32 * scale).collect();

        self.pos += RECORD_BYTES;
        self.yielded += 1;
        Some(Ok(EpisodeRecord {
            id,
            timestamp_secs: ts,
            values,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn make_record(id: u64, val: f32) -> EpisodeRecord {
        EpisodeRecord {
            id,
            timestamp_secs: id as f64,
            values: vec![val; COLD_DIM],
        }
    }

    #[test]
    fn round_trip_single_record() {
        let dir = TempDir::new().unwrap();
        let mut archive = ColdArchive::open(dir.path()).unwrap();
        let rec = make_record(42, 0.5);
        archive.append(&rec).unwrap();
        assert_eq!(archive.len(), 1);

        let results: Vec<_> = archive.iter().unwrap().collect();
        assert_eq!(results.len(), 1);
        let back = results[0].as_ref().unwrap();
        assert_eq!(back.id, 42);
        // INT8 at scale 0.5/127 ≈ 0.00394; max error per element < 0.005.
        for v in &back.values {
            assert!((v - 0.5).abs() < 0.005, "dequant error too large: {v}");
        }
    }

    #[test]
    fn round_trip_zero_vector() {
        let dir = TempDir::new().unwrap();
        let mut archive = ColdArchive::open(dir.path()).unwrap();
        archive.append(&make_record(0, 0.0)).unwrap();
        let results: Vec<_> = archive.iter().unwrap().collect();
        let back = results[0].as_ref().unwrap();
        assert!(back.values.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn bad_magic_detected() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("episodes.cold");
        std::fs::write(&path, b"BADMAGIC").unwrap();
        let result = ColdArchive::open(dir.path());
        assert!(matches!(result, Err(MemoryError::BadMagic)));
    }

    #[test]
    fn multiple_records_preserved() {
        let dir = TempDir::new().unwrap();
        let mut archive = ColdArchive::open(dir.path()).unwrap();
        for i in 0..5u64 {
            archive.append(&make_record(i, i as f32 * 0.1)).unwrap();
        }
        assert_eq!(archive.len(), 5);
        let ids: Vec<u64> = archive.iter().unwrap().map(|r| r.unwrap().id).collect();
        assert_eq!(ids, vec![0, 1, 2, 3, 4]);
    }
}

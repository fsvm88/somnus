use std::{
    fs::{File, OpenOptions},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
};

use memmap2::Mmap;

use crate::{
    episode::{EpisodeRecord, IngestSource},
    error::{MemoryError, MemoryResult},
};

/// Dimension of episode vectors — must match `somnus-perception::binding::EPISODE_DIM`.
/// Bumped to 640 (5 regions × 128) in Phase 6 (ADR 0013). Old 512-dim archives
/// are incompatible; MAGIC bumped so they fail gracefully.
pub const COLD_DIM: usize = 640;

// Bumped from SMNCOLD2: added salience f32 per record (ADR 0020 salience-weighted eviction).
const MAGIC: &[u8; 8] = b"SMNCOLD3";

// Per-record layout (little-endian):
//   id:            u64   8 B
//   timestamp:     f64   8 B
//   scale:         f32   4 B
//   values:        [i8; COLD_DIM]  640 B
//   ingest_source: u8    1 B
//   salience:      f32   4 B
//   ──────────────────────
//   total:               665 B
/// On-disk bytes per record: id(8) + ts(8) + scale(4) + values(`COLD_DIM`) + src(1) +
/// salience(4). Exposed so callers (the spill loop) can derive bytes-written from a record
/// count instead of duplicating this layout arithmetic.
pub const RECORD_BYTES: usize = 8 + 8 + 4 + COLD_DIM + 1 + 4;
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
        let (result, _) = somnus_metrics::timed_core!(
            somnus_metrics::Metric::ColdArchiveOpen,
            Self::open_inner(dir)
        );
        result
    }

    fn open_inner(dir: &Path) -> MemoryResult<Self> {
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
        self.append_batch(std::iter::once(record))
    }

    /// Quantise and append many records through one file handle and one buffered writer.
    ///
    /// Records are written in order; on the first error (dimension mismatch or I/O) the records
    /// already written stay in the archive and `len()` reflects exactly those, so a caller can
    /// compare `len()` before and after to see how many landed.
    pub fn append_batch<'a>(
        &mut self,
        records: impl IntoIterator<Item = &'a EpisodeRecord>,
    ) -> MemoryResult<()> {
        let mut file = OpenOptions::new().append(true).open(&self.path)?;
        let mut w = BufWriter::new(&mut file);
        let mut written = 0usize;
        let result = (|| -> MemoryResult<()> {
            for record in records {
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
                w.write_all(&record.id.to_le_bytes())?;
                w.write_all(&record.timestamp_secs.to_le_bytes())?;
                w.write_all(&scale.to_le_bytes())?;
                // Safety: i8 and u8 have same size/alignment; transmuting to &[u8] for writing is fine.
                w.write_all(unsafe {
                    std::slice::from_raw_parts(quantised.as_ptr() as *const u8, COLD_DIM)
                })?;
                w.write_all(&[record.ingest_source as u8])?;
                w.write_all(&record.salience.to_le_bytes())?;
                written += 1;
            }
            Ok(())
        })();
        // Flush what was buffered even when a later record failed, so len() matches the file.
        let flushed = w.flush();
        match (&result, flushed) {
            (_, Ok(())) => self.len += written,
            (Ok(()), Err(e)) => return Err(e.into()),
            (Err(_), Err(_)) => {} // file state unknown; len stays, original error returned
        }
        result
    }

    /// Iterate all records, dequantising each vector.
    pub fn iter(&self) -> MemoryResult<ColdIter> {
        self.iter_from(0)
    }

    /// Iterate records starting at `start_record` (0-based index), skipping earlier ones.
    /// Seeks directly to `HEADER_BYTES + start_record * RECORD_BYTES` — no soft-skip loop.
    pub fn iter_from(&self, start_record: usize) -> MemoryResult<ColdIter> {
        let start = start_record.min(self.len);
        let file = File::open(&self.path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        Ok(ColdIter {
            mmap,
            pos: HEADER_BYTES + start * RECORD_BYTES,
            len: self.len - start, // remaining records, not total
            yielded: 0,
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Rewrite the archive keeping only the most recent `n` records.
    ///
    /// Cheap for large archives: reads only the tail slice and rewrites in one pass.
    /// No-op if `n >= self.len`.
    pub fn truncate_to_recent(&mut self, n: usize) -> MemoryResult<()> {
        let (result, _) = somnus_metrics::timed_core!(
            somnus_metrics::Metric::ColdArchiveTruncate,
            self.truncate_to_recent_inner(n)
        );
        result
    }

    fn truncate_to_recent_inner(&mut self, n: usize) -> MemoryResult<()> {
        if n >= self.len {
            return Ok(());
        }
        let keep_start = HEADER_BYTES + (self.len - n) * RECORD_BYTES;
        let keep_bytes = {
            let file = File::open(&self.path)?;
            let mmap = unsafe { Mmap::map(&file)? };
            mmap[keep_start..keep_start + n * RECORD_BYTES].to_vec()
        };
        let mut w = BufWriter::new(File::create(&self.path)?);
        w.write_all(MAGIC)?;
        w.write_all(&keep_bytes)?;
        w.flush()?;
        self.len = n;
        Ok(())
    }

    /// Rewrite the archive keeping the `n` highest-salience records.
    ///
    /// Reads the salience field from each record (bytes 661–664 of the SMNCOLD3 layout),
    /// selects the top-n by salience descending, then rewrites them in their original
    /// chronological order. High salience = novel (large L2 from warm memory) = keep.
    /// `f32::MAX` sentinel (empty warm memory) sorts as maximally salient.
    /// No-op if `n >= self.len`.
    pub fn evict_lowest_salience(&mut self, n: usize) -> MemoryResult<()> {
        let (result, _) = somnus_metrics::timed_core!(
            somnus_metrics::Metric::ColdArchiveEvict,
            self.evict_lowest_salience_inner(n)
        );
        result
    }

    fn evict_lowest_salience_inner(&mut self, n: usize) -> MemoryResult<()> {
        if n >= self.len {
            return Ok(());
        }
        somnus_metrics::counted_core!(
            somnus_metrics::Metric::ColdArchiveEvictedRecords,
            (self.len - n) as u64
        );
        // Salience field offset within each record: after id(8)+ts(8)+scale(4)+values(640)+src(1).
        const SAL_OFFSET: usize = 8 + 8 + 4 + COLD_DIM + 1;
        let file = File::open(&self.path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        // Collect (record_index, salience) pairs.
        let mut index_sal: Vec<(usize, f32)> = (0..self.len)
            .map(|i| {
                let base = HEADER_BYTES + i * RECORD_BYTES + SAL_OFFSET;
                let sal = f32::from_le_bytes(mmap[base..base + 4].try_into().unwrap());
                (i, sal)
            })
            .collect();

        // Partition: find the n highest saliences.
        // Use partial_sort equivalent: select top-n by descending salience.
        index_sal.select_nth_unstable_by(self.len - n, |a, b| {
            a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut keep: Vec<usize> = index_sal[self.len - n..].iter().map(|&(i, _)| i).collect();
        // Restore chronological order.
        keep.sort_unstable();

        // Copy kept record bytes before releasing the mmap — File::create truncates the file,
        // which would cause SIGBUS if the mmap were still live during writes.
        let kept_bytes: Vec<u8> = keep
            .iter()
            .flat_map(|&idx| {
                let start = HEADER_BYTES + idx * RECORD_BYTES;
                mmap[start..start + RECORD_BYTES].to_vec()
            })
            .collect();
        drop(mmap);
        drop(file);

        let mut w = BufWriter::new(File::create(&self.path)?);
        w.write_all(MAGIC)?;
        w.write_all(&kept_bytes)?;
        w.flush()?;
        self.len = n;
        Ok(())
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
        let ingest_source = IngestSource::from_u8(bytes[20 + COLD_DIM]);
        let salience = f32::from_le_bytes(
            bytes[20 + COLD_DIM + 1..20 + COLD_DIM + 5]
                .try_into()
                .unwrap(),
        );

        self.pos += RECORD_BYTES;
        self.yielded += 1;
        Some(Ok(EpisodeRecord {
            id,
            timestamp_secs: ts,
            values,
            ingest_source,
            salience,
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
            ingest_source: IngestSource::Unknown,
            salience: 0.5,
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

    #[test]
    fn iter_from_seeks_to_offset_and_clamps() {
        let dir = TempDir::new().unwrap();
        let mut archive = ColdArchive::open(dir.path()).unwrap();
        for i in 0..5u64 {
            archive.append(&make_record(i, i as f32 * 0.1)).unwrap();
        }

        let ids: Vec<u64> = archive
            .iter_from(2)
            .unwrap()
            .map(|r| r.unwrap().id)
            .collect();
        assert_eq!(ids, vec![2, 3, 4], "iter_from(2) skips records 0-1");

        // Data at the seek offset must be the real record, not a misaligned read.
        let first = archive.iter_from(3).unwrap().next().unwrap().unwrap();
        assert_eq!(first.id, 3);
        assert!((first.values[0] - 0.3).abs() < 0.005);

        assert_eq!(
            archive.iter_from(5).unwrap().count(),
            0,
            "start == len yields nothing"
        );
        assert_eq!(
            archive.iter_from(100).unwrap().count(),
            0,
            "start > len clamps to len"
        );
    }

    #[test]
    fn truncate_to_recent_keeps_tail() {
        let dir = TempDir::new().unwrap();
        let mut archive = ColdArchive::open(dir.path()).unwrap();
        for i in 0..10u64 {
            archive.append(&make_record(i, i as f32 * 0.1)).unwrap();
        }
        archive.truncate_to_recent(4).unwrap();
        assert_eq!(archive.len(), 4);
        let ids: Vec<u64> = archive.iter().unwrap().map(|r| r.unwrap().id).collect();
        assert_eq!(ids, vec![6, 7, 8, 9]);
    }

    #[test]
    fn truncate_noop_when_n_gte_len() {
        let dir = TempDir::new().unwrap();
        let mut archive = ColdArchive::open(dir.path()).unwrap();
        for i in 0..3u64 {
            archive.append(&make_record(i, 0.5)).unwrap();
        }
        archive.truncate_to_recent(5).unwrap();
        assert_eq!(archive.len(), 3);
        archive.truncate_to_recent(3).unwrap();
        assert_eq!(archive.len(), 3);
    }

    #[test]
    fn truncate_file_survives_reopen() {
        let dir = TempDir::new().unwrap();
        let mut archive = ColdArchive::open(dir.path()).unwrap();
        for i in 0..8u64 {
            archive.append(&make_record(i, i as f32 * 0.05)).unwrap();
        }
        archive.truncate_to_recent(3).unwrap();
        // Reopen and verify length + ids.
        let archive2 = ColdArchive::open(dir.path()).unwrap();
        assert_eq!(archive2.len(), 3);
        let ids: Vec<u64> = archive2.iter().unwrap().map(|r| r.unwrap().id).collect();
        assert_eq!(ids, vec![5, 6, 7]);
    }

    #[test]
    fn salience_round_trips() {
        let dir = TempDir::new().unwrap();
        let mut archive = ColdArchive::open(dir.path()).unwrap();
        let saliences = [0.0f32, 0.25, 0.5, 1.0, f32::MAX];
        for (i, &sal) in saliences.iter().enumerate() {
            let mut rec = make_record(i as u64, 0.1);
            rec.salience = sal;
            archive.append(&rec).unwrap();
        }
        let back: Vec<f32> = archive
            .iter()
            .unwrap()
            .map(|r| r.unwrap().salience)
            .collect();
        assert_eq!(back, saliences);
    }

    #[test]
    fn evict_lowest_salience_keeps_high_salience_records() {
        let dir = TempDir::new().unwrap();
        let mut archive = ColdArchive::open(dir.path()).unwrap();
        // Records: id 0..9, salience = id * 0.1 (id 9 = highest, id 0 = lowest).
        for i in 0..10u64 {
            let mut rec = make_record(i, 0.5);
            rec.salience = i as f32 * 0.1;
            archive.append(&rec).unwrap();
        }
        archive.evict_lowest_salience(4).unwrap();
        assert_eq!(archive.len(), 4);
        // Expect the 4 highest-salience records (ids 6,7,8,9), in chronological order.
        let ids: Vec<u64> = archive.iter().unwrap().map(|r| r.unwrap().id).collect();
        assert_eq!(ids, vec![6, 7, 8, 9]);
    }

    #[test]
    fn evict_lowest_salience_noop_when_n_gte_len() {
        let dir = TempDir::new().unwrap();
        let mut archive = ColdArchive::open(dir.path()).unwrap();
        for i in 0..3u64 {
            archive.append(&make_record(i, 0.5)).unwrap();
        }
        archive.evict_lowest_salience(5).unwrap();
        assert_eq!(archive.len(), 3);
    }

    #[test]
    fn evict_lowest_salience_sentinel_max_kept() {
        // Records with f32::MAX salience should never be evicted over real values.
        let dir = TempDir::new().unwrap();
        let mut archive = ColdArchive::open(dir.path()).unwrap();
        // 5 records: first 3 have tiny salience, last 2 have f32::MAX (cold-start sentinel).
        for i in 0..3u64 {
            let mut rec = make_record(i, 0.5);
            rec.salience = 0.01 * (i + 1) as f32;
            archive.append(&rec).unwrap();
        }
        for i in 3..5u64 {
            let mut rec = make_record(i, 0.5);
            rec.salience = f32::MAX;
            archive.append(&rec).unwrap();
        }
        archive.evict_lowest_salience(3).unwrap();
        assert_eq!(archive.len(), 3);
        let ids: Vec<u64> = archive.iter().unwrap().map(|r| r.unwrap().id).collect();
        // Keep: id 2 (0.03) and ids 3,4 (MAX) — evict ids 0 (0.01) and 1 (0.02).
        assert_eq!(ids, vec![2, 3, 4]);
    }

    #[test]
    fn evict_lowest_salience_file_survives_reopen() {
        let dir = TempDir::new().unwrap();
        let mut archive = ColdArchive::open(dir.path()).unwrap();
        for i in 0..8u64 {
            let mut rec = make_record(i, 0.5);
            rec.salience = i as f32 * 0.1;
            archive.append(&rec).unwrap();
        }
        archive.evict_lowest_salience(3).unwrap();
        let archive2 = ColdArchive::open(dir.path()).unwrap();
        assert_eq!(archive2.len(), 3);
        let ids: Vec<u64> = archive2.iter().unwrap().map(|r| r.unwrap().id).collect();
        assert_eq!(ids, vec![5, 6, 7]);
    }

    #[test]
    fn ingest_source_round_trips() {
        let dir = TempDir::new().unwrap();
        let mut archive = ColdArchive::open(dir.path()).unwrap();
        let sources = [
            IngestSource::Speech,
            IngestSource::Env,
            IngestSource::Visual,
            IngestSource::Language,
            IngestSource::Mixed,
            IngestSource::Unknown,
        ];
        for (i, &src) in sources.iter().enumerate() {
            let mut rec = make_record(i as u64, 0.1);
            rec.ingest_source = src;
            archive.append(&rec).unwrap();
        }
        let back: Vec<_> = archive
            .iter()
            .unwrap()
            .map(|r| r.unwrap().ingest_source)
            .collect();
        assert_eq!(back, sources);
    }

    #[test]
    fn append_batch_matches_sequential_append_and_reports_partial_progress() {
        let a_dir = TempDir::new().unwrap();
        let b_dir = TempDir::new().unwrap();
        let mut seq = ColdArchive::open(a_dir.path()).unwrap();
        let mut bat = ColdArchive::open(b_dir.path()).unwrap();
        let recs: Vec<_> = (0..5)
            .map(|i| make_record(i, 0.1 * (i as f32 + 1.0)))
            .collect();
        for r in &recs {
            seq.append(r).unwrap();
        }
        bat.append_batch(&recs).unwrap();
        assert_eq!(seq.len(), bat.len());
        let a = std::fs::read(a_dir.path().join("episodes.cold")).unwrap();
        let b = std::fs::read(b_dir.path().join("episodes.cold")).unwrap();
        assert_eq!(
            a, b,
            "batched file must be byte-identical to sequential appends"
        );

        // A bad record mid-batch: earlier records land, len() says exactly how many.
        let mut bad = make_record(99, 0.5);
        bad.values.truncate(3);
        let before = bat.len();
        let batch = [
            make_record(10, 0.2),
            make_record(11, 0.3),
            bad,
            make_record(12, 0.4),
        ];
        assert!(bat.append_batch(&batch).is_err());
        assert_eq!(bat.len(), before + 2);
        assert_eq!(bat.iter().unwrap().count(), before + 2);
    }
}

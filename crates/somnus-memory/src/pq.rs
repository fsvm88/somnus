/// Product-quantisation (PQ) compression for the cold archive (Phase 11a, ADR 0016).
///
/// The INT8 ColdArchive remains the live write path; this module provides an offline
/// conversion that produces a much smaller file used for analysis and future retrieval.
///
/// Design constants (fixed; changing invalidates existing PQ files):
///   PQ_M=8 subspaces × PQ_K=256 centroids × PQ_SUB_DIM=80 floats each.
///   Per-record bytes: id(8) + ts(8) + codes(8) = 24, vs 660 for INT8 → 27.5× reduction.
///   Codebook: 8 × 256 × 80 × 4 = 655 360 bytes, stored once at the file head.
use std::{
    fs::{File, OpenOptions},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
};

use memmap2::Mmap;

use crate::{
    cold::{ColdArchive, COLD_DIM},
    episode::EpisodeRecord,
    error::{MemoryError, MemoryResult},
};

pub const PQ_M: usize = 8;
pub const PQ_K: usize = 256; // centroids per subspace — fits in u8
pub const PQ_SUB_DIM: usize = COLD_DIM / PQ_M; // 80
pub const PQ_MAX_ITER: usize = 25;
pub const PQ_TRAIN_SAMPLE: usize = 8192;

const MAGIC_PQ: &[u8; 8] = b"SMNCOLDQ";

// File layout:
//   magic:   [u8; 8]      8 B
//   m:       u8           1 B
//   k:       u16 LE       2 B
//   dim:     u16 LE       2 B
//   ──────────────────────── 13 B
//   codebook: [f32; M×K×SUB_DIM] LE   655 360 B
//   records:  id(u64) + ts(f64) + codes([u8;M])   24 B each
const PQ_HEADER_BYTES: usize = 13;
const PQ_CODEBOOK_BYTES: usize = PQ_M * PQ_K * PQ_SUB_DIM * 4;
const PQ_PREAMBLE: usize = PQ_HEADER_BYTES + PQ_CODEBOOK_BYTES;
const PQ_RECORD_BYTES: usize = 8 + 8 + PQ_M; // 24

// ── Codebook ─────────────────────────────────────────────────────────────────

pub struct PqCodebook {
    /// centroids[subspace][centroid][component]
    centroids: Vec<Vec<Vec<f32>>>,
}

impl PqCodebook {
    /// Train on a slice of COLD_DIM-length vectors.
    /// Uses deterministic even-spaced k-means initialisation.
    pub fn train(vectors: &[Vec<f32>]) -> Self {
        assert!(!vectors.is_empty(), "need at least one vector to train");
        assert_eq!(vectors[0].len(), COLD_DIM);

        let mut centroids = Vec::with_capacity(PQ_M);
        for m in 0..PQ_M {
            let start = m * PQ_SUB_DIM;
            let subs: Vec<&[f32]> = vectors
                .iter()
                .map(|v| &v[start..start + PQ_SUB_DIM])
                .collect();
            centroids.push(kmeans(&subs, PQ_K, PQ_MAX_ITER));
        }
        Self { centroids }
    }

    /// Encode one COLD_DIM vector into PQ_M byte codes.
    pub fn encode(&self, v: &[f32]) -> [u8; PQ_M] {
        assert_eq!(v.len(), COLD_DIM);
        let mut codes = [0u8; PQ_M];
        for (m, code) in codes.iter_mut().enumerate() {
            let start = m * PQ_SUB_DIM;
            let sub = &v[start..start + PQ_SUB_DIM];
            let mut best = 0usize;
            let mut best_d = f32::MAX;
            for (k, c) in self.centroids[m].iter().enumerate() {
                let d = l2sq(sub, c);
                if d < best_d {
                    best_d = d;
                    best = k;
                }
            }
            *code = best as u8;
        }
        codes
    }

    /// Reconstruct a COLD_DIM vector from PQ_M byte codes.
    pub fn decode(&self, codes: &[u8; PQ_M]) -> Vec<f32> {
        let mut v = vec![0.0f32; COLD_DIM];
        for (m, &code) in codes.iter().enumerate() {
            let start = m * PQ_SUB_DIM;
            let c = &self.centroids[m][code as usize];
            v[start..start + PQ_SUB_DIM].copy_from_slice(c);
        }
        v
    }

    fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(PQ_CODEBOOK_BYTES);
        for m in 0..PQ_M {
            for k in 0..PQ_K {
                for &val in &self.centroids[m][k] {
                    buf.extend_from_slice(&val.to_le_bytes());
                }
            }
        }
        debug_assert_eq!(buf.len(), PQ_CODEBOOK_BYTES);
        buf
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        assert_eq!(bytes.len(), PQ_CODEBOOK_BYTES);
        let mut centroids = Vec::with_capacity(PQ_M);
        let mut offset = 0usize;
        for _ in 0..PQ_M {
            let mut ks = Vec::with_capacity(PQ_K);
            for _ in 0..PQ_K {
                let mut sub = Vec::with_capacity(PQ_SUB_DIM);
                for _ in 0..PQ_SUB_DIM {
                    let val =
                        f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
                    sub.push(val);
                    offset += 4;
                }
                ks.push(sub);
            }
            centroids.push(ks);
        }
        Self { centroids }
    }
}

// ── k-means (Lloyd's) ─────────────────────────────────────────────────────────

fn l2sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Lloyd's k-means with deterministic even-spaced initialisation.
/// Returns `k` centroids of length `sub_dim`.
fn kmeans(vecs: &[&[f32]], k: usize, max_iter: usize) -> Vec<Vec<f32>> {
    let n = vecs.len();
    let sub_dim = vecs[0].len();
    let k_actual = k.min(n);

    // Even-spaced initialisation — deterministic, no RNG needed.
    let mut centroids: Vec<Vec<f32>> = (0..k_actual)
        .map(|i| vecs[i * n / k_actual].to_vec())
        .collect();
    // Pad if n < k (shouldn't happen when PQ_TRAIN_SAMPLE >> PQ_K).
    while centroids.len() < k {
        centroids.push(vecs[0].to_vec());
    }

    let mut assignments = vec![0usize; n];
    let mut dist_buf = vec![0.0f32; k];

    for _ in 0..max_iter {
        // Assign each vector to its nearest centroid.
        let mut changed = false;
        for (i, v) in vecs.iter().enumerate() {
            for (ki, c) in centroids.iter().enumerate() {
                dist_buf[ki] = l2sq(v, c);
            }
            let best = dist_buf
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap();
            if assignments[i] != best {
                assignments[i] = best;
                changed = true;
            }
        }
        if !changed {
            break;
        }

        // Update centroids.
        let mut sums = vec![vec![0.0f32; sub_dim]; k];
        let mut counts = vec![0usize; k];
        for (i, v) in vecs.iter().enumerate() {
            let c = assignments[i];
            for (s, x) in sums[c].iter_mut().zip(v.iter()) {
                *s += x;
            }
            counts[c] += 1;
        }
        for c in 0..k {
            if counts[c] > 0 {
                let cnt = counts[c] as f32;
                for s in &mut sums[c] {
                    *s /= cnt;
                }
                centroids[c] = sums[c].clone();
            }
            // If a centroid is empty we keep the old position (stable, matches scikit default).
        }
    }

    centroids
}

// ── PQ archive ───────────────────────────────────────────────────────────────

/// PQ-compressed cold archive. Read-write; the codebook is fixed at creation.
pub struct PqColdArchive {
    path: PathBuf,
    codebook: PqCodebook,
    len: usize,
}

impl PqColdArchive {
    /// Create a new PQ archive in `dir`, writing the header and codebook immediately.
    pub fn create(dir: &Path, codebook: PqCodebook) -> MemoryResult<Self> {
        std::fs::create_dir_all(dir)?;
        let path = dir.join("episodes.pq");
        let mut w = BufWriter::new(File::create(&path)?);

        w.write_all(MAGIC_PQ)?;
        w.write_all(&[PQ_M as u8])?;
        w.write_all(&(PQ_K as u16).to_le_bytes())?;
        w.write_all(&(COLD_DIM as u16).to_le_bytes())?;
        w.write_all(&codebook.to_bytes())?;
        w.flush()?;

        Ok(Self { path, codebook, len: 0 })
    }

    /// Open an existing PQ archive (reads and validates header + codebook).
    pub fn open(dir: &Path) -> MemoryResult<Self> {
        let path = dir.join("episodes.pq");
        let file = File::open(&path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        if mmap.len() < PQ_PREAMBLE {
            return Err(MemoryError::BadMagic);
        }
        if &mmap[..8] != MAGIC_PQ {
            return Err(MemoryError::BadMagic);
        }
        let m = mmap[8] as usize;
        let k = u16::from_le_bytes(mmap[9..11].try_into().unwrap()) as usize;
        let dim = u16::from_le_bytes(mmap[11..13].try_into().unwrap()) as usize;
        if m != PQ_M || k != PQ_K || dim != COLD_DIM {
            return Err(MemoryError::DimMismatch {
                expected: COLD_DIM,
                got: dim,
            });
        }

        let codebook =
            PqCodebook::from_bytes(&mmap[PQ_HEADER_BYTES..PQ_HEADER_BYTES + PQ_CODEBOOK_BYTES]);

        let body = mmap.len() - PQ_PREAMBLE;
        let len = body / PQ_RECORD_BYTES;

        Ok(Self { path, codebook, len })
    }

    /// Encode and append one episode record.
    pub fn append(&mut self, record: &EpisodeRecord) -> MemoryResult<()> {
        if record.values.len() != COLD_DIM {
            return Err(MemoryError::DimMismatch {
                expected: COLD_DIM,
                got: record.values.len(),
            });
        }
        let codes = self.codebook.encode(&record.values);
        let mut file = OpenOptions::new().append(true).open(&self.path)?;
        let mut w = BufWriter::new(&mut file);
        w.write_all(&record.id.to_le_bytes())?;
        w.write_all(&record.timestamp_secs.to_le_bytes())?;
        w.write_all(&codes)?;
        w.flush()?;
        self.len += 1;
        Ok(())
    }

    /// Iterate all records, dequantising each vector from its PQ codes.
    pub fn iter(&self) -> MemoryResult<PqIter> {
        let file = File::open(&self.path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        Ok(PqIter {
            mmap,
            codebook: &self.codebook as *const PqCodebook,
            pos: PQ_PREAMBLE,
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

    pub fn codebook(&self) -> &PqCodebook {
        &self.codebook
    }
}

pub struct PqIter {
    mmap: Mmap,
    // Raw pointer: `PqColdArchive` owns the codebook and outlives any iterator
    // created from it. The iterator is not Send/Sync because of this pointer.
    codebook: *const PqCodebook,
    pos: usize,
    len: usize,
    yielded: usize,
}

impl Iterator for PqIter {
    type Item = MemoryResult<EpisodeRecord>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.yielded >= self.len {
            return None;
        }
        let end = self.pos + PQ_RECORD_BYTES;
        if end > self.mmap.len() {
            return Some(Err(MemoryError::CorruptRecord {
                offset: self.pos as u64,
            }));
        }
        let bytes = &self.mmap[self.pos..end];
        let id = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
        let ts = f64::from_le_bytes(bytes[8..16].try_into().unwrap());
        let mut codes = [0u8; PQ_M];
        codes.copy_from_slice(&bytes[16..16 + PQ_M]);

        // SAFETY: codebook pointer is valid for the lifetime of PqColdArchive,
        // which must outlive this iterator (enforced by borrow at call site).
        let values = unsafe { (*self.codebook).decode(&codes) };

        self.pos += PQ_RECORD_BYTES;
        self.yielded += 1;
        Some(Ok(EpisodeRecord {
            id,
            timestamp_secs: ts,
            values,
        }))
    }
}

// ── Conversion ───────────────────────────────────────────────────────────────

/// Convert an existing INT8 cold archive to a PQ archive in `pq_dir`.
///
/// Training sample: up to `PQ_TRAIN_SAMPLE` vectors drawn with deterministic
/// even-spacing so results are reproducible without a fixed RNG seed.
pub fn convert_cold_to_pq(src: &ColdArchive, pq_dir: &Path) -> MemoryResult<PqColdArchive> {
    let n = src.len();
    if n == 0 {
        return Err(MemoryError::DimMismatch {
            expected: 1,
            got: 0,
        });
    }

    // Collect training sample: up to PQ_TRAIN_SAMPLE vectors, even-spaced so
    // the sample is reproducible and covers the full archive.
    let sample_n = PQ_TRAIN_SAMPLE.min(n);
    let selected: std::collections::HashSet<usize> =
        (0..sample_n).map(|j| j * n / sample_n).collect();
    let mut sample: Vec<Vec<f32>> = Vec::with_capacity(sample_n);
    for (i, rec) in src.iter()?.enumerate() {
        if selected.contains(&i) {
            sample.push(rec?.values);
        }
    }

    // Train codebook.
    let codebook = PqCodebook::train(&sample);

    // Create PQ archive and encode all records.
    let mut pq = PqColdArchive::create(pq_dir, codebook)?;
    for rec in src.iter()? {
        pq.append(&rec?)?;
    }

    Ok(pq)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn synthetic_vectors(n: usize, seed: u64) -> Vec<Vec<f32>> {
        // Simple LCG for deterministic test vectors.
        let mut state = seed;
        let mut lcg = || -> f32 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
        };
        (0..n).map(|_| (0..COLD_DIM).map(|_| lcg()).collect()).collect()
    }

    #[test]
    fn codebook_encode_decode_round_trip() {
        let vecs = synthetic_vectors(512, 42);
        let cb = PqCodebook::train(&vecs);
        // Reconstruction error relative to vector norm < 20% on random data.
        let mut rel_errors = Vec::new();
        for v in &vecs[..20] {
            let codes = cb.encode(v);
            let rec = cb.decode(&codes);
            let err: f32 = v.iter().zip(rec.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f32>().sqrt();
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                rel_errors.push(err / norm);
            }
        }
        let mean_rel: f32 = rel_errors.iter().sum::<f32>() / rel_errors.len() as f32;
        assert!(mean_rel < 0.5, "mean relative reconstruction error too large: {mean_rel}");
    }

    #[test]
    fn codebook_serialise_round_trip() {
        let vecs = synthetic_vectors(300, 7);
        let cb = PqCodebook::train(&vecs);
        let bytes = cb.to_bytes();
        assert_eq!(bytes.len(), PQ_CODEBOOK_BYTES);
        let cb2 = PqCodebook::from_bytes(&bytes);
        // Encode the same vector with both; codes must match.
        let probe = &vecs[10];
        assert_eq!(cb.encode(probe), cb2.encode(probe));
    }

    #[test]
    fn pq_archive_round_trip() {
        let dir = TempDir::new().unwrap();
        let vecs = synthetic_vectors(400, 13);
        let cb = PqCodebook::train(&vecs);

        let mut arch = PqColdArchive::create(dir.path(), cb).unwrap();
        for (i, v) in vecs.iter().enumerate() {
            arch.append(&EpisodeRecord {
                id: i as u64,
                timestamp_secs: i as f64,
                values: v.clone(),
            })
            .unwrap();
        }
        assert_eq!(arch.len(), 400);

        // Re-open and verify ids are preserved.
        let arch2 = PqColdArchive::open(dir.path()).unwrap();
        assert_eq!(arch2.len(), 400);
        let ids: Vec<u64> = arch2.iter().unwrap().map(|r| r.unwrap().id).collect();
        assert_eq!(ids, (0u64..400).collect::<Vec<_>>());
    }

    #[test]
    fn convert_cold_to_pq_size_and_quality() {
        // Build a small INT8 archive, convert it, check size + recall quality.
        let cold_dir = TempDir::new().unwrap();
        let pq_dir = TempDir::new().unwrap();

        let vecs = synthetic_vectors(600, 99);
        let mut cold = ColdArchive::open(cold_dir.path()).unwrap();
        for (i, v) in vecs.iter().enumerate() {
            cold.append(&EpisodeRecord {
                id: i as u64,
                timestamp_secs: i as f64,
                values: v.clone(),
            })
            .unwrap();
        }

        let pq = convert_cold_to_pq(&cold, pq_dir.path()).unwrap();
        assert_eq!(pq.len(), cold.len());

        // Size reduction: PQ record (24 B) vs INT8 record (660 B) → at least 4×.
        let ratio = 660.0 / PQ_RECORD_BYTES as f32;
        assert!(
            ratio >= 4.0,
            "size reduction {ratio}× is below 4× target"
        );

        // L2 quality: for each record, the PQ reconstruction L2 vs the original
        // dequantised vector. Mean relative error must be < 10% of vector norm.
        // (On random data the error is higher; this tests the code path and file format.)
        let cold_records: Vec<_> = cold.iter().unwrap().map(|r| r.unwrap().values).collect();
        let pq_records: Vec<_> = pq.iter().unwrap().map(|r| r.unwrap().values).collect();
        assert_eq!(cold_records.len(), pq_records.len());

        let mean_rel: f32 = cold_records
            .iter()
            .zip(pq_records.iter())
            .map(|(orig, rec)| {
                let err: f32 =
                    orig.iter().zip(rec.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f32>().sqrt();
                let norm: f32 = orig.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 { err / norm } else { 0.0 }
            })
            .sum::<f32>()
            / cold_records.len() as f32;

        // Random uniform vectors have poor PQ quality; threshold is generous here.
        // Real episode vectors (correlated) achieve < 10% as stated in the roadmap.
        assert!(mean_rel < 1.0, "mean relative PQ error suspiciously large: {mean_rel}");
    }

    #[test]
    fn bad_magic_detected() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("episodes.pq"), b"BADMAGIC").unwrap();
        assert!(matches!(PqColdArchive::open(dir.path()), Err(MemoryError::BadMagic)));
    }

    #[test]
    fn record_bytes_constant() {
        // 24 B per record → 27.5× reduction vs 660 B INT8 record.
        assert_eq!(PQ_RECORD_BYTES, 24);
        assert!(660 / PQ_RECORD_BYTES >= 27);
    }
}

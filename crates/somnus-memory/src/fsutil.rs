//! Crash-safe file replacement. A kill (SIGKILL, OOM, power loss) mid-write must leave either
//! the previous file or the new one on disk, never a truncated mix: snapshots are the only copy
//! of days of training.

use std::{
    fs::{self, File},
    io::{self, Write},
    path::{Path, PathBuf},
};

/// `<path>.tmp` next to `path` (same directory, so the final rename never crosses filesystems).
pub fn tmp_path(path: &Path) -> PathBuf {
    let mut s = path.as_os_str().to_owned();
    s.push(".tmp");
    PathBuf::from(s)
}

/// Flush `tmp` to disk, then atomically rename it over `dst`.
///
/// Call after a writer that is not under our control (e.g. a library's `save_file`) has
/// finished writing `tmp`.
pub fn commit_tmp(tmp: &Path, dst: &Path) -> io::Result<()> {
    File::open(tmp)?.sync_all()?;
    fs::rename(tmp, dst)?;
    // Persist the rename itself; best effort (not every filesystem supports fsync on a dir).
    if let Some(dir) = dst.parent().filter(|d| !d.as_os_str().is_empty()) {
        if let Ok(d) = File::open(dir) {
            let _ = d.sync_all();
        }
    }
    Ok(())
}

/// Write `bytes` to `path` atomically (temp file, fsync, rename).
pub fn write_atomic(path: &Path, bytes: impl AsRef<[u8]>) -> io::Result<()> {
    let tmp = tmp_path(path);
    let mut f = File::create(&tmp)?;
    f.write_all(bytes.as_ref())?;
    drop(f);
    commit_tmp(&tmp, path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_atomic_replaces_and_leaves_no_tmp() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("state.bin");
        write_atomic(&p, b"one").unwrap();
        write_atomic(&p, b"two!").unwrap();
        assert_eq!(fs::read(&p).unwrap(), b"two!");
        assert!(!tmp_path(&p).exists());
    }

    #[test]
    fn stale_tmp_from_a_crash_does_not_affect_the_committed_file() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("state.bin");
        write_atomic(&p, b"good").unwrap();
        // Simulates dying mid-write of the next version: a partial tmp is left behind.
        fs::write(tmp_path(&p), b"par").unwrap();
        assert_eq!(fs::read(&p).unwrap(), b"good");
        // The next save overwrites the stale tmp and succeeds.
        write_atomic(&p, b"next").unwrap();
        assert_eq!(fs::read(&p).unwrap(), b"next");
    }
}

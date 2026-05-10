use std::io;

#[derive(Debug, thiserror::Error)]
pub enum MemoryError {
    #[error("io error: {0}")]
    Io(#[from] io::Error),

    #[error("cold archive bad magic")]
    BadMagic,

    #[error("cold archive corrupt record at byte offset {offset}")]
    CorruptRecord { offset: u64 },

    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimMismatch { expected: usize, got: usize },
}

pub type MemoryResult<T> = Result<T, MemoryError>;

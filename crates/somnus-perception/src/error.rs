use crate::region::RegionId;

#[derive(Debug, thiserror::Error)]
pub enum PerceptionError {
    #[error("candle error: {0}")]
    Candle(#[from] candle_core::Error),

    #[error("region mismatch: expected {expected:?}, got {got:?}")]
    RegionMismatch { expected: RegionId, got: RegionId },

    #[error("binding area requires all four regions to have activations")]
    BindingIncomplete,
}

pub type PerceptionResult<T> = Result<T, PerceptionError>;

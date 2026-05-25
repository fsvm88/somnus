use crate::region::RegionId;

#[derive(Debug, thiserror::Error)]
pub enum PerceptionError {
    #[error("burn error: {0}")]
    Burn(String),

    #[error("region mismatch: expected {expected:?}, got {got:?}")]
    RegionMismatch { expected: RegionId, got: RegionId },

    #[error("binding area requires all four regions to have activations")]
    BindingIncomplete,
}

pub type PerceptionResult<T> = Result<T, PerceptionError>;

use candle_core::{Device, Module, Tensor};
use candle_nn::{linear, Linear, VarBuilder};

use crate::error::PerceptionResult;

pub const N_UNITS: usize = 1024;
pub const K_ACTIVE: usize = 20;

const BOOST_STRENGTH: f32 = 1.0;
const DUTY_CYCLE_ALPHA: f32 = 1.0 / 1000.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RegionId {
    AudioSpeech,
    AudioEnv,
    Visual,
    Symbolic,
    Language,
}

impl RegionId {
    pub fn as_index(self) -> usize {
        match self {
            RegionId::AudioSpeech => 0,
            RegionId::AudioEnv => 1,
            RegionId::Visual => 2,
            RegionId::Symbolic => 3,
            RegionId::Language => 4,
        }
    }
}

/// Sparse output of a k-WTA layer.
/// Indices are sorted ascending; values are the raw post-linear activations.
#[derive(Debug, Clone)]
pub struct SparseActivation {
    pub region: RegionId,
    pub indices: Vec<u16>,
    pub values: Vec<f32>,
    /// Total number of units in the layer (N_UNITS at init; grows after unit addition).
    pub n_units: usize,
}

impl SparseActivation {
    /// Dense vector of size `self.n_units` — full layer representation.
    pub fn to_dense(&self) -> Vec<f32> {
        let mut out = vec![0.0f32; self.n_units];
        for (&idx, &val) in self.indices.iter().zip(self.values.iter()) {
            out[idx as usize] = val;
        }
        out
    }

    /// Dense vector clamped to `n` elements — used by the binding area's fixed-size interface.
    /// Units with index >= n are dropped (they exist in the extended layer but not in binding yet).
    pub fn to_dense_n(&self, n: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; n];
        for (&idx, &val) in self.indices.iter().zip(self.values.iter()) {
            let i = idx as usize;
            if i < n {
                out[i] = val;
            }
        }
        out
    }
}

/// k-WTA linear layer with per-unit duty-cycle boosting.
pub struct KWtaLayer {
    pub region: RegionId,
    pub linear: Linear,
    pub n_units: usize,
    device: Device,
    pub duty_cycle: Vec<f32>,
}

impl KWtaLayer {
    pub fn new(
        region: RegionId,
        n_in: usize,
        vb: VarBuilder,
        device: Device,
    ) -> PerceptionResult<Self> {
        let linear = linear(n_in, N_UNITS, vb)?;
        Ok(Self {
            region,
            linear,
            n_units: N_UNITS,
            device,
            duty_cycle: vec![0.0f32; N_UNITS],
        })
    }

    pub fn forward(&mut self, input: &[f32]) -> PerceptionResult<SparseActivation> {
        let t = Tensor::from_vec(input.to_vec(), (1, input.len()), &self.device)?;
        let z = self.linear.forward(&t)?.squeeze(0)?;
        let z_vec: Vec<f32> = z.to_vec1()?;
        self.select_top_k(z_vec)
    }

    /// Forward pass without updating the duty-cycle moving average.
    ///
    /// Use during settling iterations so that multiple passes per encoder input
    /// do not inflate the effective duty-cycle learning rate (ADR 0011).
    pub fn forward_no_dc(&self, input: &[f32]) -> PerceptionResult<SparseActivation> {
        let t = Tensor::from_vec(input.to_vec(), (1, input.len()), &self.device)?;
        let z = self.linear.forward(&t)?.squeeze(0)?;
        let z_vec: Vec<f32> = z.to_vec1()?;
        self.select_top_k_no_dc(z_vec)
    }

    /// Forward pass that also returns the intermediate tracked tensor `z` for backprop.
    /// Returns `(activation, z_tensor)` where z_tensor has shape `[n_units]`.
    pub fn forward_tracked(
        &mut self,
        input: &[f32],
    ) -> PerceptionResult<(SparseActivation, Tensor)> {
        let t = Tensor::from_vec(input.to_vec(), (1, input.len()), &self.device)?;
        let z = self.linear.forward(&t)?.squeeze(0)?; // tracked [n_units]
        let z_vec: Vec<f32> = z.to_vec1()?; // read values — does NOT detach z

        let act = self.select_top_k(z_vec)?;
        Ok((act, z))
    }

    /// Batched forward for NREM replay.
    ///
    /// `input_batch` has shape `[B, n_in]`. Returns `(z_batch, mask_batch)` both
    /// shape `[B, n_units]`. `z_batch` is tracked; `mask_batch` is a detached
    /// constant. Caller computes `sparse_z = z_batch * mask_batch` to preserve
    /// gradient flow through the encoder weights.
    ///
    /// Duty cycles are updated per-sample in batch order (same semantics as
    /// the sequential path).
    pub fn forward_tracked_batch(
        &mut self,
        input_batch: &Tensor,
        b: usize,
    ) -> PerceptionResult<(Tensor, Tensor)> {
        let z_batch = self.linear.forward(input_batch)?; // [B, n_units]
        let z_vec2: Vec<Vec<f32>> = z_batch.to_vec2()?;

        let n = self.n_units;
        let mut flat_masks = vec![0.0f32; b * n];

        for (si, z_vec) in z_vec2.iter().enumerate() {
            let target_dc = K_ACTIVE as f32 / n as f32;
            let mut boosted = z_vec.clone();
            for (i, dc) in self.duty_cycle.iter().enumerate() {
                boosted[i] += BOOST_STRENGTH * (target_dc - dc);
            }
            let mut indexed: Vec<(usize, f32)> = boosted.iter().copied().enumerate().collect();
            let split = n - K_ACTIVE;
            indexed.select_nth_unstable_by(split, |a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            let top_k = &indexed[split..];

            let active: std::collections::HashSet<usize> =
                top_k.iter().map(|&(i, _)| i).collect();
            for i in 0..n {
                let fired = if active.contains(&i) { 1.0 } else { 0.0 };
                self.duty_cycle[i] =
                    (1.0 - DUTY_CYCLE_ALPHA) * self.duty_cycle[i] + DUTY_CYCLE_ALPHA * fired;
                if fired > 0.0 {
                    flat_masks[si * n + i] = 1.0;
                }
            }
        }

        let mask_t = Tensor::from_vec(flat_masks, (b, n), &self.device)?;
        Ok((z_batch, mask_t))
    }

    fn select_top_k(&mut self, z_vec: Vec<f32>) -> PerceptionResult<SparseActivation> {
        let n = self.n_units;

        let target_dc = K_ACTIVE as f32 / n as f32;
        let mut boosted = z_vec.clone();
        for (i, dc) in self.duty_cycle.iter().enumerate() {
            boosted[i] += BOOST_STRENGTH * (target_dc - dc);
        }

        let mut indexed: Vec<(usize, f32)> = boosted.iter().copied().enumerate().collect();
        let split = n - K_ACTIVE;
        indexed.select_nth_unstable_by(split, |a, b| {
            a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        let top_k = &indexed[split..];

        let mut pairs: Vec<(u16, f32)> = top_k.iter().map(|&(i, _)| (i as u16, z_vec[i])).collect();
        pairs.sort_unstable_by_key(|(i, _)| *i);

        let active_set: std::collections::HashSet<usize> = top_k.iter().map(|&(i, _)| i).collect();
        for i in 0..n {
            let fired = if active_set.contains(&i) { 1.0 } else { 0.0 };
            self.duty_cycle[i] =
                (1.0 - DUTY_CYCLE_ALPHA) * self.duty_cycle[i] + DUTY_CYCLE_ALPHA * fired;
        }

        Ok(SparseActivation {
            region: self.region,
            indices: pairs.iter().map(|(i, _)| *i).collect(),
            values: pairs.iter().map(|(_, v)| *v).collect(),
            n_units: n,
        })
    }

    /// Same selection logic as `select_top_k` but does not update duty cycles.
    fn select_top_k_no_dc(&self, z_vec: Vec<f32>) -> PerceptionResult<SparseActivation> {
        let n = self.n_units;
        let target_dc = K_ACTIVE as f32 / n as f32;
        let mut boosted = z_vec.clone();
        for (i, dc) in self.duty_cycle.iter().enumerate() {
            boosted[i] += BOOST_STRENGTH * (target_dc - dc);
        }
        let mut indexed: Vec<(usize, f32)> = boosted.iter().copied().enumerate().collect();
        let split = n - K_ACTIVE;
        indexed.select_nth_unstable_by(split, |a, b| {
            a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        let top_k = &indexed[split..];
        let mut pairs: Vec<(u16, f32)> = top_k.iter().map(|&(i, _)| (i as u16, z_vec[i])).collect();
        pairs.sort_unstable_by_key(|(i, _)| *i);
        Ok(SparseActivation {
            region: self.region,
            indices: pairs.iter().map(|(i, _)| *i).collect(),
            values: pairs.iter().map(|(_, v)| *v).collect(),
            n_units: n,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;
    use candle_nn::VarMap;

    fn make_layer(region: RegionId, n_in: usize) -> KWtaLayer {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        KWtaLayer::new(region, n_in, vb, device).unwrap()
    }

    #[test]
    fn kwta_produces_exactly_k_active() {
        let mut layer = make_layer(RegionId::Visual, 384);
        let input = vec![0.5f32; 384];
        let act = layer.forward(&input).unwrap();
        assert_eq!(act.indices.len(), K_ACTIVE);
        assert_eq!(act.values.len(), K_ACTIVE);
    }

    #[test]
    fn kwta_indices_sorted_and_in_range() {
        let mut layer = make_layer(RegionId::AudioEnv, 512);
        let input: Vec<f32> = (0..512).map(|i| i as f32 / 512.0).collect();
        let act = layer.forward(&input).unwrap();
        for &idx in &act.indices {
            assert!((idx as usize) < N_UNITS);
        }
        for w in act.indices.windows(2) {
            assert!(w[0] < w[1], "indices not sorted");
        }
    }

    #[test]
    fn to_dense_has_correct_nonzero_count() {
        let mut layer = make_layer(RegionId::AudioSpeech, 384);
        let input = vec![1.0f32; 384];
        let act = layer.forward(&input).unwrap();
        let dense = act.to_dense();
        assert_eq!(dense.len(), N_UNITS);
        let nonzero = dense.iter().filter(|&&v| v != 0.0).count();
        assert_eq!(nonzero, K_ACTIVE);
    }

    #[test]
    fn to_dense_n_truncates_to_n() {
        let act = SparseActivation {
            region: RegionId::Visual,
            indices: vec![0, 512, 1023],
            values: vec![1.0, 2.0, 3.0],
            n_units: 1024,
        };
        let dense = act.to_dense_n(512);
        assert_eq!(dense.len(), 512);
        assert_eq!(dense[0], 1.0);
        // index 512 is out of range for n=512, should be dropped
        assert!(dense.iter().skip(1).all(|&v| v == 0.0));
    }
}

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
}

impl RegionId {
    pub fn as_index(self) -> usize {
        match self {
            RegionId::AudioSpeech => 0,
            RegionId::AudioEnv => 1,
            RegionId::Visual => 2,
            RegionId::Symbolic => 3,
        }
    }
}

/// Sparse output of a k-WTA layer: k active units out of N_UNITS total.
/// Indices are sorted ascending; values are the raw post-linear activations.
#[derive(Debug, Clone)]
pub struct SparseActivation {
    pub region: RegionId,
    pub indices: Vec<u16>,
    pub values: Vec<f32>,
}

impl SparseActivation {
    /// Reconstruct a dense N_UNITS-dim vector from the sparse code.
    pub fn to_dense(&self) -> Vec<f32> {
        let mut out = vec![0.0f32; N_UNITS];
        for (&idx, &val) in self.indices.iter().zip(self.values.iter()) {
            out[idx as usize] = val;
        }
        out
    }
}

/// k-WTA linear layer with per-unit duty-cycle boosting.
pub struct KWtaLayer {
    pub region: RegionId,
    linear: Linear,
    device: Device,
    // Running mean activation frequency per unit (exponential moving average).
    duty_cycle: Vec<f32>,
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
            device,
            duty_cycle: vec![0.0f32; N_UNITS],
        })
    }

    pub fn forward(&mut self, input: &[f32]) -> PerceptionResult<SparseActivation> {
        let t = Tensor::from_vec(input.to_vec(), (1, input.len()), &self.device)?;
        let z = self.linear.forward(&t)?.squeeze(0)?;
        let z_vec: Vec<f32> = z.to_vec1()?;

        // Apply boosting: raise pre-activations of chronically under-used units.
        let target_dc = K_ACTIVE as f32 / N_UNITS as f32;
        let mut boosted = z_vec.clone();
        for (i, dc) in self.duty_cycle.iter().enumerate() {
            boosted[i] += BOOST_STRENGTH * (target_dc - dc);
        }

        // Hard top-k on boosted activations; store original (unboosted) values.
        let mut indexed: Vec<(usize, f32)> = boosted.iter().copied().enumerate().collect();
        // Partial sort: move top-K to the back.
        let split = N_UNITS - K_ACTIVE;
        indexed.select_nth_unstable_by(split, |a, b| {
            a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        let top_k = &indexed[split..];

        // Collect and sort by index for a canonical representation.
        let mut pairs: Vec<(u16, f32)> = top_k.iter().map(|&(i, _)| (i as u16, z_vec[i])).collect();
        pairs.sort_unstable_by_key(|(i, _)| *i);

        // Update duty cycles.
        let active_set: std::collections::HashSet<usize> = top_k.iter().map(|&(i, _)| i).collect();
        for i in 0..N_UNITS {
            let fired = if active_set.contains(&i) { 1.0 } else { 0.0 };
            self.duty_cycle[i] =
                (1.0 - DUTY_CYCLE_ALPHA) * self.duty_cycle[i] + DUTY_CYCLE_ALPHA * fired;
        }

        Ok(SparseActivation {
            region: self.region,
            indices: pairs.iter().map(|(i, _)| *i).collect(),
            values: pairs.iter().map(|(_, v)| *v).collect(),
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
}

use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{
        backend::{AutodiffBackend, Backend},
        Tensor, TensorData,
    },
};

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
///
/// Generic over backend B. The training-path methods (forward_tracked,
/// forward_tracked_batch) are in a separate impl block requiring AutodiffBackend.
#[derive(Module, Debug)]
pub struct KWtaLayer<B: Backend> {
    pub linear: Linear<B>,
    #[module(skip)]
    pub region: RegionId,
    #[module(skip)]
    pub n_units: usize,
    #[module(skip)]
    pub duty_cycle: Vec<f32>,
}

impl<B: Backend> KWtaLayer<B> {
    pub fn new(region: RegionId, n_in: usize, device: &B::Device) -> Self {
        let linear = LinearConfig::new(n_in, N_UNITS)
            .with_bias(false)
            .init(device);
        Self {
            linear,
            region,
            n_units: N_UNITS,
            duty_cycle: vec![0.0f32; N_UNITS],
        }
    }

    pub fn forward(&mut self, input: &[f32], device: &B::Device) -> SparseActivation {
        let data = TensorData::new(input.to_vec(), [1, input.len()]);
        let t = Tensor::<B, 2>::from_data(data, device);
        let z: Tensor<B, 2> = self.linear.forward(t);
        let z_vec: Vec<f32> = z.into_data().convert::<f32>().to_vec().unwrap();
        self.select_top_k(z_vec)
    }

    /// Forward pass without updating the duty-cycle moving average.
    ///
    /// Use during settling iterations so multiple passes per encoder input
    /// do not inflate the effective duty-cycle learning rate (ADR 0011).
    pub fn forward_no_dc(&self, input: &[f32], device: &B::Device) -> SparseActivation {
        let data = TensorData::new(input.to_vec(), [1, input.len()]);
        let t = Tensor::<B, 2>::from_data(data, device);
        let z: Tensor<B, 2> = self.linear.forward(t);
        let z_vec: Vec<f32> = z.into_data().convert::<f32>().to_vec().unwrap();
        self.select_top_k_no_dc(z_vec)
    }

    /// Apply k-WTA topk + duty-cycle update to a pre-computed z vector.
    ///
    /// Use when the linear matmul runs externally (e.g. encoder ORT session in Phase 1.5).
    /// Duty-cycle cadence note: when the encoder ORT is active, this is called once per
    /// binding event for all 5 regions simultaneously, vs. the burn path which updates
    /// per-region per-ingest. The rate differs; validate against burn-only runs.
    pub fn topk_from_z(&mut self, z_vec: Vec<f32>) -> SparseActivation {
        self.select_top_k(z_vec)
    }

    /// topk selection from a pre-computed z vector, without duty-cycle update.
    /// Used by the settle loop: settle is a refinement pass, not a real ingest,
    /// so dc should not be advanced.
    pub fn topk_from_z_no_dc(&self, z_vec: Vec<f32>) -> SparseActivation {
        self.select_top_k_no_dc(z_vec)
    }

    fn select_top_k(&mut self, z_vec: Vec<f32>) -> SparseActivation {
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

        // Update duty cycles: decay all, then boost active units.
        // Avoids a HashSet allocation + O(n) membership-test loop.
        for dc in self.duty_cycle.iter_mut() {
            *dc *= 1.0 - DUTY_CYCLE_ALPHA;
        }
        for &(i, _) in top_k.iter() {
            self.duty_cycle[i] += DUTY_CYCLE_ALPHA;
        }

        SparseActivation {
            region: self.region,
            indices: pairs.iter().map(|(i, _)| *i).collect(),
            values: pairs.iter().map(|(_, v)| *v).collect(),
            n_units: n,
        }
    }

    fn select_top_k_no_dc(&self, z_vec: Vec<f32>) -> SparseActivation {
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
        SparseActivation {
            region: self.region,
            indices: pairs.iter().map(|(i, _)| *i).collect(),
            values: pairs.iter().map(|(_, v)| *v).collect(),
            n_units: n,
        }
    }
}

/// Training-path methods: require AutodiffBackend so `.detach()` is available.
impl<B: AutodiffBackend> KWtaLayer<B> {
    /// Forward pass that returns the tracked z tensor for backprop.
    ///
    /// Returns `(activation, z_tensor)` where z_tensor has shape `[1, n_units]`.
    /// `.detach()` is called on a clone for the readback; the original `z` participates
    /// in the gradient graph.
    pub fn forward_tracked(
        &mut self,
        input: &[f32],
        device: &B::Device,
    ) -> (SparseActivation, Tensor<B, 2>) {
        let data = TensorData::new(input.to_vec(), [1, input.len()]);
        let t = Tensor::<B, 2>::from_data(data, device);
        let z: Tensor<B, 2> = self.linear.forward(t); // [1, n_units], tracked

        let z_vec: Vec<f32> = z
            .clone()
            .detach()
            .into_data()
            .convert::<f32>()
            .to_vec()
            .unwrap();

        let act = self.select_top_k(z_vec);
        (act, z)
    }

    /// Batched forward for NREM replay.
    ///
    /// `input_batch` has shape `[B, n_in]`. Returns `(z_batch, flat_masks)` where
    /// `z_batch` is `[B, n_units]` tracked and `flat_masks` is a flat `Vec<f32>` of
    /// shape `[B * n_units]` for the caller to upload as an untracked mask tensor.
    ///
    /// Duty cycles are updated per-sample in batch order.
    pub fn forward_tracked_batch(
        &mut self,
        input_batch: Tensor<B, 2>,
        b: usize,
        batch_topk: bool,
    ) -> (Tensor<B, 2>, Vec<f32>) {
        let z_batch: Tensor<B, 2> = self.linear.forward(input_batch); // [B, n_units]

        let z_vals: Vec<f32> = z_batch
            .clone()
            .detach()
            .into_data()
            .convert::<f32>()
            .to_vec()
            .unwrap();

        let n = self.n_units;
        let mut flat_masks = vec![0.0f32; b * n];
        let target_dc = K_ACTIVE as f32 / n as f32;

        if batch_topk {
            // BatchTopK: compute one threshold over the entire batch
            // Snapshot duty-cycle and compute boosted values
            let boosted_snapshot: Vec<f32> = z_vals
                .iter()
                .enumerate()
                .map(|(idx, &z)| {
                    let dc = self.duty_cycle[idx % n];
                    z + BOOST_STRENGTH * (target_dc - dc)
                })
                .collect();

            // Find threshold τ such that top (b * K_ACTIVE) values exceed it
            let target_k = b * K_ACTIVE;
            let mut indexed: Vec<(usize, f32)> =
                boosted_snapshot.iter().copied().enumerate().collect();
            let split = boosted_snapshot.len() - target_k;
            indexed.select_nth_unstable_by(split, |a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            let threshold = indexed[split].1;

            // Apply threshold: set mask for all activations >= threshold
            for (idx, &boosted) in boosted_snapshot.iter().enumerate() {
                if boosted >= threshold {
                    flat_masks[idx] = 1.0;
                }
            }

            // Now update duty cycles: decay all once per sample, boost actives
            for sample_idx in 0..b {
                for unit_idx in 0..n {
                    self.duty_cycle[unit_idx] *= 1.0 - DUTY_CYCLE_ALPHA;
                    if flat_masks[sample_idx * n + unit_idx] > 0.0 {
                        self.duty_cycle[unit_idx] += DUTY_CYCLE_ALPHA;
                    }
                }
            }
        } else {
            // Per-sample top-k: original behavior
            for (si, z_row) in z_vals.chunks(n).enumerate() {
                let mut boosted: Vec<f32> = z_row.to_vec();
                for (i, dc) in self.duty_cycle.iter().enumerate() {
                    boosted[i] += BOOST_STRENGTH * (target_dc - dc);
                }
                let mut indexed: Vec<(usize, f32)> = boosted.iter().copied().enumerate().collect();
                let split = n - K_ACTIVE;
                indexed.select_nth_unstable_by(split, |a, b| {
                    a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                });
                let top_k = &indexed[split..];

                // Decay all duty cycles, then boost active units and set mask bits.
                for dc in self.duty_cycle.iter_mut() {
                    *dc *= 1.0 - DUTY_CYCLE_ALPHA;
                }
                for &(i, _) in top_k.iter() {
                    self.duty_cycle[i] += DUTY_CYCLE_ALPHA;
                    flat_masks[si * n + i] = 1.0;
                }
            }
        }

        (z_batch, flat_masks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArrayDevice;
    use burn::backend::NdArray;

    fn make_layer(region: RegionId, n_in: usize) -> KWtaLayer<NdArray> {
        let device = NdArrayDevice::Cpu;
        KWtaLayer::new(region, n_in, &device)
    }

    #[test]
    fn kwta_produces_exactly_k_active() {
        let device = NdArrayDevice::Cpu;
        let mut layer = make_layer(RegionId::Visual, 384);
        let input = vec![0.5f32; 384];
        let act = layer.forward(&input, &device);
        assert_eq!(act.indices.len(), K_ACTIVE);
        assert_eq!(act.values.len(), K_ACTIVE);
    }

    #[test]
    fn kwta_indices_sorted_and_in_range() {
        let device = NdArrayDevice::Cpu;
        let mut layer = make_layer(RegionId::AudioEnv, 512);
        let input: Vec<f32> = (0..512).map(|i| i as f32 / 512.0).collect();
        let act = layer.forward(&input, &device);
        for &idx in &act.indices {
            assert!((idx as usize) < N_UNITS);
        }
        for w in act.indices.windows(2) {
            assert!(w[0] < w[1], "indices not sorted");
        }
    }

    #[test]
    fn to_dense_has_correct_nonzero_count() {
        let device = NdArrayDevice::Cpu;
        let mut layer = make_layer(RegionId::AudioSpeech, 384);
        let input = vec![1.0f32; 384];
        let act = layer.forward(&input, &device);
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
        assert!(dense.iter().skip(1).all(|&v| v == 0.0));
    }
}

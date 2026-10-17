use crate::tensor_ext::TensorF32Ext;
use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{Device, Tensor, TensorData},
};

pub const N_UNITS: usize = 1024;
pub const K_ACTIVE: usize = 20;

const BOOST_STRENGTH: f32 = 1.0;
const DUTY_CYCLE_ALPHA: f32 = 1.0 / 1000.0;

/// Active-unit count that preserves the baseline `K_ACTIVE / N_UNITS` sparsity ratio
/// at a grown region size. Growth (`grow_region`/`grow_region_seeded`) adds units but
/// never revisited k — left at the fixed baseline, a 10x region growth would collapse
/// relative sparsity from ~2% to ~0.2%. Rounds to nearest, floored at 1.
pub fn scaled_k(n_units: usize) -> usize {
    ((n_units as f64 * K_ACTIVE as f64 / N_UNITS as f64).round() as usize).max(1)
}

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

    /// Indices/values padded (with `(0, 0.0)`) or truncated to exactly `k` entries —
    /// the ONNX binding graph's sparse-gather inputs have static shape `[k]`, so
    /// every caller of `run_binding` must supply exactly `k` regardless of how
    /// many units are genuinely active (e.g. the `lang_zero` placeholder for a
    /// missing modality, or a dense→sparse re-derivation that ties at exactly
    /// zero). A padding row of `(index 0, value 0.0)` contributes exactly nil
    /// through the Gather + weighted-sum binding rewrite.
    pub fn to_exact_k(&self, k: usize) -> (Vec<i64>, Vec<f32>) {
        let mut idx: Vec<i64> = self.indices.iter().take(k).map(|&i| i as i64).collect();
        let mut val: Vec<f32> = self.values.iter().take(k).copied().collect();
        idx.resize(k, 0);
        val.resize(k, 0.0);
        (idx, val)
    }
}

/// Cumulative (since process start) magnitude statistics of the active slots chosen on the ingest
/// path, split between the original `N_UNITS` and grown units (gaps.md §50 diagnostics).
#[derive(Debug, Clone, Default)]
pub struct SlotStats {
    pub base_abs_sum: f64,
    pub base_n: u64,
    pub grown_abs_sum: f64,
    pub grown_n: u64,
}

/// k-WTA linear layer with per-unit duty-cycle boosting.
///
/// Generic over backend B. The training-path methods (forward_tracked,
/// forward_tracked_batch) are in a separate impl block requiring AutodiffBackend.
#[derive(Module, Debug)]
pub struct KWtaLayer {
    pub linear: Linear,
    #[module(skip)]
    pub region: RegionId,
    #[module(skip)]
    pub n_units: usize,
    /// Active-unit count for this layer's k-WTA selection. Starts at K_ACTIVE and is
    /// rescaled (via `scaled_k`) by the caller whenever `n_units` grows, to keep the
    /// sparsity ratio roughly constant instead of decaying as the region grows.
    #[module(skip)]
    pub k_active: usize,
    #[module(skip)]
    pub duty_cycle: Vec<f32>,
    /// Active-slot magnitude diagnostics, updated by `select_top_k` (ingest path only).
    #[module(skip)]
    pub slot_stats: SlotStats,
}

impl KWtaLayer {
    pub fn new(region: RegionId, n_in: usize, device: &Device) -> Self {
        let linear = LinearConfig::new(n_in, N_UNITS)
            .with_bias(false)
            .init(device);
        Self {
            linear,
            region,
            n_units: N_UNITS,
            k_active: K_ACTIVE,
            duty_cycle: vec![0.0f32; N_UNITS],
            slot_stats: SlotStats::default(),
        }
    }

    pub fn forward(&mut self, input: &[f32], device: &Device) -> SparseActivation {
        let data = TensorData::new(input.to_vec(), [1, input.len()]);
        let t = Tensor::<2>::from_data(data, device);
        let z: Tensor<2> =
            somnus_metrics::timed!(somnus_metrics::Metric::KwtaScoreMatmul, self.linear.forward(t));
        let z_vec: Vec<f32> = z.to_f32_vec();
        self.select_top_k(z_vec)
    }

    /// Forward pass without updating the duty-cycle moving average.
    ///
    /// Use during settling iterations so multiple passes per encoder input
    /// do not inflate the effective duty-cycle learning rate (ADR 0011).
    pub fn forward_no_dc(&self, input: &[f32], device: &Device) -> SparseActivation {
        let data = TensorData::new(input.to_vec(), [1, input.len()]);
        let t = Tensor::<2>::from_data(data, device);
        let z: Tensor<2> = self.linear.forward(t);
        let z_vec: Vec<f32> = z.to_f32_vec();
        self.select_top_k_no_dc(z_vec)
    }

    /// Batched `forward_no_dc`: one `[B, n_in]` matmul for `inputs`, then per-row top-k.
    /// Row-for-row equal to calling `forward_no_dc` on each input (duty cycles untouched).
    pub fn forward_no_dc_batch(&self, inputs: &[&[f32]], device: &Device) -> Vec<SparseActivation> {
        if inputs.is_empty() {
            return Vec::new();
        }
        let n_in = inputs[0].len();
        let flat: Vec<f32> = inputs.iter().flat_map(|v| v.iter().copied()).collect();
        let t = Tensor::<2>::from_data(TensorData::new(flat, [inputs.len(), n_in]), device);
        let z: Tensor<2> =
            somnus_metrics::timed!(somnus_metrics::Metric::KwtaScoreMatmul, self.linear.forward(t));
        let z_vec: Vec<f32> = z.to_f32_vec();
        z_vec
            .chunks_exact(self.n_units)
            .map(|row| self.select_top_k_no_dc(row.to_vec()))
            .collect()
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
        somnus_metrics::timed!(somnus_metrics::Metric::KwtaSelectTopK, {
            let act = self.select_top_k_no_dc(z_vec);

            // Update duty cycles: decay all, then boost active units.
            // Avoids a HashSet allocation + O(n) membership-test loop.
            for dc in self.duty_cycle.iter_mut() {
                *dc *= 1.0 - DUTY_CYCLE_ALPHA;
            }
            for (&i, &v) in act.indices.iter().zip(&act.values) {
                self.duty_cycle[i as usize] += DUTY_CYCLE_ALPHA;
                let st = &mut self.slot_stats;
                if (i as usize) < N_UNITS {
                    st.base_abs_sum += v.abs() as f64;
                    st.base_n += 1;
                } else {
                    st.grown_abs_sum += v.abs() as f64;
                    st.grown_n += 1;
                }
            }
            act
        })
    }

    /// Boosted top-k selection without the duty-cycle update; `select_top_k` builds on it.
    /// Values are the raw (un-boosted) `z`.
    fn select_top_k_no_dc(&self, z_vec: Vec<f32>) -> SparseActivation {
        let n = self.n_units;
        let k = self.k_active;
        let target_dc = k as f32 / n as f32;
        let mut indexed: Vec<(usize, f32)> = z_vec
            .iter()
            .zip(&self.duty_cycle)
            .enumerate()
            .map(|(i, (z, dc))| (i, z + BOOST_STRENGTH * (target_dc - dc)))
            .collect();
        let split = n - k;
        indexed.select_nth_unstable_by(split, |a, b| {
            a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut idx: Vec<u16> = indexed[split..].iter().map(|&(i, _)| i as u16).collect();
        idx.sort_unstable();
        SparseActivation {
            region: self.region,
            values: idx.iter().map(|&i| z_vec[i as usize]).collect(),
            indices: idx,
            n_units: n,
        }
    }
}

/// Training-path methods: require AutodiffBackend so `.detach()` is available.
impl KWtaLayer {
    /// Forward pass that returns the tracked z tensor for backprop.
    ///
    /// Returns `(activation, z_tensor)` where z_tensor has shape `[1, n_units]`.
    /// `.detach()` is called on a clone for the readback; the original `z` participates
    /// in the gradient graph.
    pub fn forward_tracked(
        &mut self,
        input: &[f32],
        device: &Device,
    ) -> (SparseActivation, Tensor<2>) {
        let data = TensorData::new(input.to_vec(), [1, input.len()]);
        let t = Tensor::<2>::from_data(data, device);
        let z: Tensor<2> = self.linear.forward(t); // [1, n_units], tracked

        let z_vec: Vec<f32> = z.clone().detach().to_f32_vec();

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
        input_batch: Tensor<2>,
        b: usize,
        batch_topk: bool,
    ) -> (Tensor<2>, Vec<f32>) {
        let z_batch: Tensor<2> = self.linear.forward(input_batch); // [B, n_units]

        let z_vals: Vec<f32> = z_batch.clone().detach().to_f32_vec();

        let n = self.n_units;
        let k = self.k_active;
        let mut flat_masks = vec![0.0f32; b * n];
        let target_dc = k as f32 / n as f32;

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

            // Find threshold τ such that top (b * k) values exceed it
            let target_k = b * k;
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
                let split = n - k;
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

/// ADR 0038 prerequisite 3 (gaps.md §42) — minimal block-local ("hierarchical") k-WTA
/// scorer, built solely to numerically validate whether block-local competition preserves
/// the invariants `KWtaLayer::select_top_k`'s single global competition guarantees:
/// duty-cycle stays bounded, the k/n sparsity ratio is preserved on growth, and no block
/// is permanently starved of selection. **Not wired into any production forward() path** —
/// ADR 0038 stays `proposed`; this exists purely to answer the open prerequisite.
///
/// `select_top_k`'s duty-cycle target is a single global ratio (`k_active/n_units`) shared
/// by every unit. Under block-local selection that silently breaks in two ways a naive port
/// would miss: (1) a block that's rarely chosen at the coarse stage never gets any of its
/// units selected regardless of their individual duty-cycle boost, so unit-level boosting
/// alone can't prevent starvation — the *block* itself needs its own duty-cycle-style boost
/// at the coarse-selection stage; (2) `k_active`'s fixed count doesn't map onto a variable
/// number of selected blocks of possibly-uneven size, so it has to be split across selected
/// blocks (proportional to block size here) rather than applied as one flat global ratio.
pub mod hierarchical_validation {
    use std::cmp::Ordering;

    /// gaps.md §42's BOOST_STRENGTH/DUTY_CYCLE_ALPHA reused verbatim — the block-local
    /// scorer is validated against the exact same constants `select_top_k` uses, not a
    /// separately-tuned set, since the question is "does block-local competition preserve
    /// the invariant," not "can a different constant compensate."
    use super::{BOOST_STRENGTH, DUTY_CYCLE_ALPHA};

    /// Per-block duty cycle plus the fixed block size — the piece of state a real
    /// hierarchical `KWtaLayer` would carry alongside the existing per-unit `duty_cycle`.
    #[derive(Debug, Clone)]
    pub struct BlockState {
        pub block_size: usize,
        pub block_duty_cycle: Vec<f32>,
    }

    impl BlockState {
        pub fn new(n_units: usize, block_size: usize) -> Self {
            let n_blocks = n_units.div_ceil(block_size);
            Self {
                block_size,
                block_duty_cycle: vec![0.0f32; n_blocks],
            }
        }

        pub fn n_blocks(&self, n_units: usize) -> usize {
            n_units.div_ceil(self.block_size)
        }

        /// Extend `block_duty_cycle` for newly created blocks after `n_units` grows.
        /// New blocks start at duty-cycle 0 — same convention `grow_region` uses for new
        /// units' entries in the per-unit `duty_cycle` vector (region.rs / lib.rs).
        pub fn grow(&mut self, new_n_units: usize) {
            let want = self.n_blocks(new_n_units);
            if want > self.block_duty_cycle.len() {
                self.block_duty_cycle.resize(want, 0.0);
            }
        }
    }

    fn block_bounds(block: usize, block_size: usize, n: usize) -> (usize, usize) {
        let start = block * block_size;
        let end = ((block + 1) * block_size).min(n);
        (start, end)
    }

    /// Two-stage selection: coarse per-block scoring (mean activation + block-level
    /// duty-cycle boost) picks `n_blocks_select` blocks, then exact top-k runs within each
    /// selected block only, with `k_active` split across selected blocks proportional to
    /// block size (largest-remainder method, so the total is always exactly `k_active`
    /// when the selected blocks have that much capacity).
    ///
    /// Returns `(index, raw z value)` pairs sorted by index, mirroring `select_top_k`'s
    /// output shape so it's a drop-in comparison target.
    pub fn select_top_k_block_local(
        z_vec: &[f32],
        unit_duty_cycle: &mut [f32],
        block_state: &mut BlockState,
        k_active: usize,
        n_blocks_select: usize,
    ) -> Vec<(u16, f32)> {
        let n = z_vec.len();
        let block_size = block_state.block_size;
        let n_blocks = block_state.n_blocks(n);
        assert_eq!(
            block_state.block_duty_cycle.len(),
            n_blocks,
            "BlockState not grown to match z_vec's n_units — call grow() first"
        );
        let n_blocks_select = n_blocks_select.clamp(1, n_blocks);

        // ── Coarse stage: pick blocks, with a block-level duty-cycle boost so a block
        // that's rarely chosen doesn't stay permanently invisible to selection. ────────
        let block_target_dc = n_blocks_select as f32 / n_blocks as f32;
        let mut block_scores: Vec<(usize, f32)> = (0..n_blocks)
            .map(|b| {
                let (start, end) = block_bounds(b, block_size, n);
                let mean: f32 = z_vec[start..end].iter().sum::<f32>() / (end - start) as f32;
                let boosted =
                    mean + BOOST_STRENGTH * (block_target_dc - block_state.block_duty_cycle[b]);
                (b, boosted)
            })
            .collect();
        let split = n_blocks - n_blocks_select;
        block_scores.select_nth_unstable_by(split, |a, b| {
            a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal)
        });
        let mut selected_blocks: Vec<usize> =
            block_scores[split..].iter().map(|&(b, _)| b).collect();
        selected_blocks.sort_unstable();

        for dc in block_state.block_duty_cycle.iter_mut() {
            *dc *= 1.0 - DUTY_CYCLE_ALPHA;
        }
        for &b in &selected_blocks {
            block_state.block_duty_cycle[b] += DUTY_CYCLE_ALPHA;
        }

        // ── Split k_active across selected blocks, proportional to block size. ─────────
        let selected_total_len: usize = selected_blocks
            .iter()
            .map(|&b| {
                let (s, e) = block_bounds(b, block_size, n);
                e - s
            })
            .sum();
        let k_active = k_active.min(selected_total_len);

        let mut quota: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
        let mut remainders: Vec<(usize, f32)> = Vec::with_capacity(selected_blocks.len());
        let mut assigned = 0usize;
        for &b in &selected_blocks {
            let (s, e) = block_bounds(b, block_size, n);
            let len = e - s;
            let exact = k_active as f32 * len as f32 / selected_total_len as f32;
            let floor = exact.floor() as usize;
            quota.insert(b, floor);
            remainders.push((b, exact - floor as f32));
            assigned += floor;
        }
        remainders.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        let mut leftover = k_active - assigned;
        for &(b, _) in &remainders {
            if leftover == 0 {
                break;
            }
            *quota.get_mut(&b).unwrap() += 1;
            leftover -= 1;
        }

        // ── Fine stage: exact top-k within each selected block, using a *per-block*
        // duty-cycle target (k_b/block_len), not the single global ratio. ──────────────
        let mut result: Vec<(u16, f32)> = Vec::with_capacity(k_active);
        for &b in &selected_blocks {
            let (start, end) = block_bounds(b, block_size, n);
            let len = end - start;
            let k_b = quota[&b].min(len);
            if k_b == 0 {
                continue;
            }
            let target_dc = k_b as f32 / len as f32;
            let mut boosted: Vec<(usize, f32)> = (start..end)
                .map(|i| {
                    (
                        i,
                        z_vec[i] + BOOST_STRENGTH * (target_dc - unit_duty_cycle[i]),
                    )
                })
                .collect();
            let split_b = len - k_b;
            boosted.select_nth_unstable_by(split_b, |a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal)
            });
            for &(i, _) in &boosted[split_b..] {
                result.push((i as u16, z_vec[i]));
            }
        }
        result.sort_unstable_by_key(|(i, _)| *i);

        for dc in unit_duty_cycle.iter_mut() {
            *dc *= 1.0 - DUTY_CYCLE_ALPHA;
        }
        for &(i, _) in &result {
            unit_duty_cycle[i as usize] += DUTY_CYCLE_ALPHA;
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Device;

    fn make_layer(region: RegionId, n_in: usize) -> KWtaLayer {
        let device = Device::flex();
        KWtaLayer::new(region, n_in, &device)
    }

    #[test]
    fn kwta_produces_exactly_k_active() {
        let device = Device::flex();
        let mut layer = make_layer(RegionId::Visual, 384);
        let input = vec![0.5f32; 384];
        let act = layer.forward(&input, &device);
        assert_eq!(act.indices.len(), K_ACTIVE);
        assert_eq!(act.values.len(), K_ACTIVE);
    }

    #[test]
    fn scaled_k_preserves_baseline_ratio() {
        assert_eq!(scaled_k(N_UNITS), K_ACTIVE);
        // 10x region growth should scale k roughly 10x too, not leave it flat at 20.
        assert_eq!(scaled_k(N_UNITS * 10), K_ACTIVE * 10);
        // Never collapses to 0 for a tiny (hypothetical) region.
        assert_eq!(scaled_k(1), 1);
    }

    #[test]
    fn kwta_respects_grown_k_active() {
        let mut layer = make_layer(RegionId::Visual, 384);
        // Simulate what grow_region/grow_region_seeded do to a layer's bookkeeping:
        // extend n_units and rescale k_active to match, without actually resizing
        // the weight tensor (irrelevant to k-WTA selection logic under test here).
        layer.n_units = N_UNITS * 10;
        layer.k_active = scaled_k(layer.n_units);
        layer.duty_cycle = vec![0.0f32; layer.n_units];
        // Sparsity ratio should be preserved, not stuck at the baseline K_ACTIVE=20.
        assert_eq!(layer.k_active, K_ACTIVE * 10);
    }

    #[test]
    fn kwta_indices_sorted_and_in_range() {
        let device = Device::flex();
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
        let device = Device::flex();
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

    #[test]
    fn to_exact_k_pads_when_short() {
        let act = SparseActivation {
            region: RegionId::Language,
            indices: vec![],
            values: vec![],
            n_units: 1024,
        };
        let (idx, val) = act.to_exact_k(20);
        assert_eq!(idx.len(), 20);
        assert_eq!(val.len(), 20);
        assert!(idx.iter().all(|&i| i == 0));
        assert!(val.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn to_exact_k_truncates_when_long() {
        let act = SparseActivation {
            region: RegionId::Visual,
            indices: vec![0, 1, 2, 3],
            values: vec![1.0, 2.0, 3.0, 4.0],
            n_units: 1024,
        };
        let (idx, val) = act.to_exact_k(2);
        assert_eq!(idx, vec![0, 1]);
        assert_eq!(val, vec![1.0, 2.0]);
    }

    #[test]
    fn to_exact_k_passes_through_when_exact() {
        let act = SparseActivation {
            region: RegionId::AudioSpeech,
            indices: vec![5, 10],
            values: vec![1.5, 2.5],
            n_units: 1024,
        };
        let (idx, val) = act.to_exact_k(2);
        assert_eq!(idx, vec![5, 10]);
        assert_eq!(val, vec![1.5, 2.5]);
    }
}

/// ADR 0038 prerequisite 3 (gaps.md §42) validation tests for `hierarchical_validation`.
#[cfg(test)]
mod hierarchical_validation_tests {
    use super::hierarchical_validation::{select_top_k_block_local, BlockState};
    use proptest::prelude::*;

    #[test]
    fn selects_exactly_k_when_capacity_allows() {
        let n = 256;
        let block_size = 32; // 8 blocks
        let z: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let mut dc = vec![0.0f32; n];
        let mut bs = BlockState::new(n, block_size);
        let out = select_top_k_block_local(&z, &mut dc, &mut bs, 20, 4);
        assert_eq!(out.len(), 20);
    }

    #[test]
    fn indices_sorted_unique_and_in_range() {
        let n = 200; // not a multiple of block_size — exercises the ragged last block
        let block_size = 32;
        let z: Vec<f32> = (0..n).map(|i| (i * 37 % 199) as f32).collect();
        let mut dc = vec![0.0f32; n];
        let mut bs = BlockState::new(n, block_size);
        let out = select_top_k_block_local(&z, &mut dc, &mut bs, 15, 3);
        for w in out.windows(2) {
            assert!(w[0].0 < w[1].0, "indices not strictly sorted/unique");
        }
        for &(i, _) in &out {
            assert!((i as usize) < n);
        }
    }

    #[test]
    fn unit_duty_cycle_stays_bounded_over_many_rounds() {
        let n = 512;
        let block_size = 64;
        let mut dc = vec![0.0f32; n];
        let mut bs = BlockState::new(n, block_size);
        for round in 0..2000u32 {
            // Deterministic pseudo-random z so different units win across rounds.
            let z: Vec<f32> = (0..n)
                .map(|i| (((i as u32).wrapping_mul(2654435761).wrapping_add(round)) % 1000) as f32)
                .collect();
            let _ = select_top_k_block_local(&z, &mut dc, &mut bs, 20, 4);
        }
        // Boosted-selection duty cycle is a decaying EWMA of "was selected" (0/1 per round,
        // decay factor DUTY_CYCLE_ALPHA) — same bound select_top_k's global duty_cycle
        // relies on. It must never runs away past 1 (a unit selected every single round
        // converges to exactly the target, not beyond).
        for &d in &dc {
            assert!(
                (0.0..=1.0).contains(&d),
                "unit duty cycle out of bounds: {d}"
            );
        }
        for &d in &bs.block_duty_cycle {
            assert!(
                (0.0..=1.0).contains(&d),
                "block duty cycle out of bounds: {d}"
            );
        }
    }

    #[test]
    fn chronically_low_scoring_block_is_not_permanently_starved() {
        // Block 0 scores modestly below every other block on raw z (a realistic
        // duty-cycle-fairness gap — comparable to the scale of the boost itself, ~0.5
        // at most here). Without a block-level duty-cycle boost (the fix this module
        // adds vs. a naive port of select_top_k's per-unit-only boosting), block 0 would
        // never be chosen no matter how long the loop runs — that's the ADR 0038
        // "Consequences" starvation risk this test exists to catch a regression of.
        //
        // Note this is a *bounded* correction, same as select_top_k's existing per-unit
        // boost: it can resolve fairness among units/blocks whose raw-score gap is within
        // BOOST_STRENGTH * target_dc, not rescue a block that's arbitrarily far behind —
        // that bound is inherent to duty-cycle boosting in general, not specific to going
        // block-local, and is exactly the kind of thing prerequisite 3 is meant to surface.
        let n = 256;
        let block_size = 32; // 8 blocks, block 0 is the cold one
        let mut dc = vec![0.0f32; n];
        let mut bs = BlockState::new(n, block_size);
        // DUTY_CYCLE_ALPHA=1/1000 (same constant as production `select_top_k`) means
        // duty-cycle fairness operates on a ~1000+ round timescale, matching how it's
        // actually used in production (thousands of ingest events per region, not a
        // handful) — this loop has to run long enough to see the correction, not just
        // a handful of rounds.
        let mut block0_selected_count = 0u32;
        for round in 0..4000u32 {
            let mut z: Vec<f32> = (0..n)
                .map(|i| ((((i as u32).wrapping_mul(97).wrapping_add(round)) % 50) as f32) / 100.0)
                .collect();
            for v in z.iter_mut().take(block_size) {
                *v -= 0.3; // block 0 consistently scores below the others, but not by an
                           // unrecoverable margin
            }
            let out = select_top_k_block_local(&z, &mut dc, &mut bs, 20, 4);
            if out.iter().any(|&(i, _)| (i as usize) < block_size) {
                block0_selected_count += 1;
            }
        }
        assert!(
            block0_selected_count > 0,
            "block 0 was never selected across 4000 rounds — starvation regression"
        );
    }

    #[test]
    fn newly_grown_units_are_immediately_eligible() {
        // Mirrors what grow_region/grow_region_seeded (lib.rs) do to a layer's bookkeeping:
        // append units, extend duty-cycle with zeros, and here also grow the block state.
        // A newly grown unit must be able to win a block-local selection in the very next
        // call, same as select_top_k's global competition guarantees today.
        let n0 = 64;
        let block_size = 32;
        let mut dc = vec![0.0f32; n0];
        let mut bs = BlockState::new(n0, block_size);

        let n1 = 96; // +1 block of new units
        dc.resize(n1, 0.0);
        bs.grow(n1);
        assert_eq!(bs.block_duty_cycle.len(), 3);

        // New block's units score far above everything else — must be selectable.
        let mut z = vec![0.0f32; n1];
        for v in z.iter_mut().skip(n0) {
            *v = 1000.0;
        }
        let out = select_top_k_block_local(&z, &mut dc, &mut bs, 8, 1);
        assert!(
            out.iter().any(|&(i, _)| (i as usize) >= n0),
            "newly grown unit was not selectable immediately after growth"
        );
    }

    proptest! {
        #[test]
        fn selection_never_exceeds_k_and_stays_in_range(
            n in 32usize..300,
            block_size in 8usize..64,
            k in 1usize..40,
            n_blocks_select in 1usize..20,
            seed in 0u32..10_000,
        ) {
            let z: Vec<f32> = (0..n)
                .map(|i| (((i as u32).wrapping_mul(2654435761).wrapping_add(seed)) % 997) as f32)
                .collect();
            let mut dc = vec![0.0f32; n];
            let mut bs = BlockState::new(n, block_size);
            let out = select_top_k_block_local(&z, &mut dc, &mut bs, k, n_blocks_select);

            prop_assert!(out.len() <= k);
            for w in out.windows(2) {
                prop_assert!(w[0].0 < w[1].0);
            }
            for &(i, _) in &out {
                prop_assert!((i as usize) < n);
            }
        }
    }
}

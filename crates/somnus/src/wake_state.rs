//! Mutable state of the wake loop in `run()`: adaptive EWMAs, per-cycle accumulators and the
//! scheduling/flag variables that used to be ~56 loose locals.

use std::{
    collections::VecDeque,
    time::{Duration, Instant},
};

/// All mutable scalars/buffers the wake loop and the sleep trigger carry between iterations.
/// Heavy resources (perception, episodic buffer, decoder, channels, sinks) stay as locals of
/// `run()`; this holds only plain state, so helper functions can take `&mut WakeState`.
pub(crate) struct WakeState {
    pub(crate) cooldown_remaining: usize,
    pub(crate) sal_mu: f32,
    pub(crate) sal_sigma: f32,
    pub(crate) sal_n: u64,
    pub(crate) settle_mu: f32,
    pub(crate) settle_n: u64,
    pub(crate) wake_ewma_mu: f32,
    pub(crate) wake_ewma_n: u64,
    pub(crate) onnx_failed: bool,
    pub(crate) wake_recon_dropped: u64,
    pub(crate) max_wake: Duration,
    pub(crate) wake_start: Instant,
    pub(crate) sleep_cycle_count: usize,
    pub(crate) startup_query_fired: bool,
    pub(crate) wake_episodes: usize,
    pub(crate) wake_output_mse_sum: f64,
    pub(crate) wake_output_mse_n: usize,
    pub(crate) prior_cosine_sums: [f64; 5],
    pub(crate) prior_cosine_counts: [usize; 5],
    pub(crate) lang_ingest_count: usize,
    pub(crate) gain_episode_counter: usize,
    pub(crate) wake_reconstruct_counter: usize,
    pub(crate) salience_sum: f64,
    pub(crate) salience_n: usize,
    pub(crate) salience_min: f32,
    pub(crate) high_salience_episodes: usize,
    pub(crate) blend_count: usize,
    pub(crate) hnsw_query_us_sum: u64,
    pub(crate) hnsw_query_n: u64,
    pub(crate) wake_reconstruct_throttle_passed: usize,
    pub(crate) episodes_speech: usize,
    pub(crate) episodes_env: usize,
    pub(crate) episodes_visual: usize,
    pub(crate) episodes_language: usize,
    pub(crate) episodes_mixed: usize,
    pub(crate) settle_trigger_count: usize,
    pub(crate) settle_iters_sum: usize,
    pub(crate) settle_delta_sum: f64,
    pub(crate) speech_encode_total_ms: u64,
    pub(crate) env_encode_total_ms: u64,
    pub(crate) video_encode_total_ms: u64,
    pub(crate) perception_total_ms: u64,
    pub(crate) cosine_ewma: Option<f64>,
    pub(crate) cosine_baseline: Option<f64>,
    pub(crate) cosine_baseline_accum: f64,
    pub(crate) cosine_baseline_n: usize,
    pub(crate) perf_sleep_flag: bool,
    pub(crate) wakefulness_pressure: f32,
    pub(crate) s4_ring: [Option<f64>; 3],
    pub(crate) s4_ring_pos: usize,
    pub(crate) s4_beta_active: bool,
    pub(crate) sleep_dur_secs_history: VecDeque<f64>,
    pub(crate) s7_mse_ewma: f64,
    pub(crate) s7_ewma_n: usize,
    pub(crate) s7_stagnant: u32,
    pub(crate) effective_wake_interval: usize,
}

impl WakeState {
    /// Zero the per-wake-cycle accumulators after the cycle's metrics line has been written.
    /// Adaptive state (EWMAs, cooldown, flags that outlive a cycle) is deliberately untouched.
    pub(crate) fn reset_cycle(&mut self) {
        self.wake_episodes = 0;
        self.wake_output_mse_sum = 0.0;
        self.wake_output_mse_n = 0;
        self.prior_cosine_sums = [0.0; 5];
        self.prior_cosine_counts = [0; 5];
        self.lang_ingest_count = 0;
        self.gain_episode_counter = 0;
        self.wake_reconstruct_counter = 0;
        self.salience_sum = 0.0;
        self.salience_n = 0;
        self.salience_min = f32::MAX;
        self.high_salience_episodes = 0;
        self.blend_count = 0;
        self.hnsw_query_us_sum = 0;
        self.hnsw_query_n = 0;
        self.wake_reconstruct_throttle_passed = 0;
        self.wake_recon_dropped = 0;
        self.episodes_speech = 0;
        self.episodes_env = 0;
        self.episodes_visual = 0;
        self.episodes_language = 0;
        self.episodes_mixed = 0;
        self.settle_trigger_count = 0;
        self.settle_iters_sum = 0;
        self.settle_delta_sum = 0.0;
        self.speech_encode_total_ms = 0;
        self.env_encode_total_ms = 0;
        self.video_encode_total_ms = 0;
        self.perception_total_ms = 0;
        self.cosine_ewma = None;
        self.cosine_baseline = None;
        self.cosine_baseline_accum = 0.0;
        self.cosine_baseline_n = 0;
        self.perf_sleep_flag = false;
    }
}

impl WakeState {
    /// S4 prior-beta scale-up controller (ADR 0024). Called once per sleep cycle: records the
    /// cycle's global prior cosine and returns the new prior beta to apply, if it changed.
    /// Scales from the base `--prior-beta` to 0.05 when the global prior cosine stays above 0.65
    /// for 3 consecutive cycles; drops back to base when any cycle falls below 0.40.
    pub(crate) fn s4_step(&mut self, base_beta: f32) -> Option<f32> {
        let global_cosine: Option<f64> =
            crate::global_prior_cosine(&self.prior_cosine_sums, &self.prior_cosine_counts);
        self.s4_ring[self.s4_ring_pos % 3] = global_cosine;
        self.s4_ring_pos += 1;
        let c = global_cosine?;
        if c < 0.40 && self.s4_beta_active {
            self.s4_beta_active = false;
            tracing::info!(
                prior_cosine = c,
                "S4: prior-beta scaled down to base (< 0.40)"
            );
            Some(base_beta)
        } else if !self.s4_beta_active
            && self.s4_ring_pos >= 3
            && crate::s4_ring_all_high(&self.s4_ring)
        {
            self.s4_beta_active = true;
            tracing::info!(
                prior_cosine = c,
                "S4: prior-beta scaled up to 0.05 (> 0.65 for 3 cycles)"
            );
            Some(0.05)
        } else {
            None
        }
    }

    /// S7: adapt `effective_wake_interval` from the cycle-level wake MSE trend. Shrinks the
    /// interval (more frequent wake reconstruction) after 3 consecutive cycles in which the MSE
    /// improved by less than 1 %, and restores it toward `base_interval` once the MSE improves
    /// again. `None` (no wake reconstruction ran this cycle) leaves everything unchanged.
    pub(crate) fn s7_step(&mut self, mean_mse_wake: Option<f64>, base_interval: usize) {
        const S7_STAGNANT_THRESHOLD: f64 = 0.01;
        const S7_STAGNANT_CYCLES: u32 = 3;
        const S7_MSE_ALPHA: f64 = 0.3;
        let Some(mse) = mean_mse_wake else {
            return;
        };
        if self.s7_ewma_n == 0 {
            self.s7_mse_ewma = mse;
        } else {
            let improvement = (self.s7_mse_ewma - mse) / self.s7_mse_ewma.max(1e-9);
            if improvement < S7_STAGNANT_THRESHOLD {
                self.s7_stagnant += 1;
            } else {
                self.s7_stagnant = 0;
                if self.effective_wake_interval < base_interval {
                    self.effective_wake_interval =
                        (self.effective_wake_interval + 1).min(base_interval);
                    tracing::info!(
                        effective_wake_interval = self.effective_wake_interval,
                        "S7: restored wake_reconstruct_interval"
                    );
                }
            }
            self.s7_mse_ewma = S7_MSE_ALPHA * mse + (1.0 - S7_MSE_ALPHA) * self.s7_mse_ewma;
        }
        self.s7_ewma_n += 1;
        if self.s7_stagnant >= S7_STAGNANT_CYCLES && self.effective_wake_interval > 1 {
            self.effective_wake_interval -= 1;
            self.s7_stagnant = 0;
            tracing::info!(
                effective_wake_interval = self.effective_wake_interval,
                s7_mse_ewma = self.s7_mse_ewma,
                "S7: reduced wake_reconstruct_interval (stagnant MSE)"
            );
        }
    }
}

#[cfg(test)]
impl WakeState {
    /// Neutral state for unit tests: cold-start EWMAs, empty accumulators.
    pub(crate) fn for_test() -> Self {
        Self {
            cooldown_remaining: 0,
            sal_mu: 0.8,
            sal_sigma: 0.2,
            sal_n: 0,
            settle_mu: 0.85,
            settle_n: 0,
            wake_ewma_mu: 0.0,
            wake_ewma_n: 0,
            onnx_failed: false,
            wake_recon_dropped: 0,
            max_wake: Duration::from_secs(300),
            wake_start: Instant::now(),
            sleep_cycle_count: 1,
            startup_query_fired: false,
            wake_episodes: 0,
            wake_output_mse_sum: 0.0,
            wake_output_mse_n: 0,
            prior_cosine_sums: [0.0; 5],
            prior_cosine_counts: [0; 5],
            lang_ingest_count: 0,
            gain_episode_counter: 0,
            wake_reconstruct_counter: 0,
            salience_sum: 0.0,
            salience_n: 0,
            salience_min: f32::MAX,
            high_salience_episodes: 0,
            blend_count: 0,
            hnsw_query_us_sum: 0,
            hnsw_query_n: 0,
            wake_reconstruct_throttle_passed: 0,
            episodes_speech: 0,
            episodes_env: 0,
            episodes_visual: 0,
            episodes_language: 0,
            episodes_mixed: 0,
            settle_trigger_count: 0,
            settle_iters_sum: 0,
            settle_delta_sum: 0.0,
            speech_encode_total_ms: 0,
            env_encode_total_ms: 0,
            video_encode_total_ms: 0,
            perception_total_ms: 0,
            cosine_ewma: None,
            cosine_baseline: None,
            cosine_baseline_accum: 0.0,
            cosine_baseline_n: 0,
            perf_sleep_flag: false,
            wakefulness_pressure: 0.0,
            s4_ring: [None; 3],
            s4_ring_pos: 0,
            s4_beta_active: false,
            sleep_dur_secs_history: VecDeque::new(),
            s7_mse_ewma: 0.0,
            s7_ewma_n: 0,
            s7_stagnant: 0,
            effective_wake_interval: 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn with_cosine(st: &mut WakeState, c: f64) {
        st.prior_cosine_sums = [c; 5];
        st.prior_cosine_counts = [1; 5];
    }

    #[test]
    fn reset_cycle_zeroes_accumulators_but_keeps_adaptive_state() {
        let mut st = WakeState::for_test();
        st.wake_episodes = 5;
        st.salience_n = 3;
        st.salience_min = 0.2;
        st.episodes_speech = 4;
        st.speech_encode_total_ms = 99;
        st.prior_cosine_sums = [1.0; 5];
        st.perf_sleep_flag = true;
        st.cosine_ewma = Some(0.5);
        // Adaptive state that must survive the reset.
        st.sal_mu = 0.9;
        st.sal_n = 600;
        st.settle_mu = 0.7;
        st.wake_ewma_n = 4;
        st.cooldown_remaining = 2;
        st.s7_mse_ewma = 0.3;
        st.s4_beta_active = true;
        st.wakefulness_pressure = 0.4;

        st.reset_cycle();

        assert_eq!(st.wake_episodes, 0);
        assert_eq!(st.salience_n, 0);
        assert_eq!(st.salience_min, f32::MAX);
        assert_eq!(st.episodes_speech, 0);
        assert_eq!(st.speech_encode_total_ms, 0);
        assert_eq!(st.prior_cosine_sums, [0.0; 5]);
        assert!(!st.perf_sleep_flag);
        assert!(st.cosine_ewma.is_none());
        assert_eq!((st.sal_mu, st.sal_n), (0.9, 600));
        assert_eq!(st.settle_mu, 0.7);
        assert_eq!(st.wake_ewma_n, 4);
        assert_eq!(st.cooldown_remaining, 2);
        assert_eq!(st.s7_mse_ewma, 0.3);
        assert!(st.s4_beta_active);
        assert_eq!(st.wakefulness_pressure, 0.4);
    }

    #[test]
    fn s4_scales_up_after_three_high_cycles_and_drops_on_a_low_one() {
        let mut st = WakeState::for_test();
        for cycle in 0..3 {
            with_cosine(&mut st, 0.8);
            let r = st.s4_step(0.01);
            assert_eq!(
                r,
                if cycle == 2 { Some(0.05) } else { None },
                "cycle {cycle}"
            );
        }
        assert!(st.s4_beta_active);
        // Still high: no further change.
        with_cosine(&mut st, 0.8);
        assert_eq!(st.s4_step(0.01), None);
        // One bad cycle drops back to the base beta.
        with_cosine(&mut st, 0.3);
        assert_eq!(st.s4_step(0.01), Some(0.01));
        assert!(!st.s4_beta_active);
    }

    #[test]
    fn s4_does_not_scale_up_when_a_cycle_in_the_window_is_not_high() {
        let mut st = WakeState::for_test();
        for c in [0.8, 0.5, 0.8] {
            with_cosine(&mut st, c);
            assert_eq!(st.s4_step(0.01), None);
        }
        assert!(!st.s4_beta_active);
        // A cycle with no prior observations records None and never triggers.
        st.prior_cosine_counts = [0; 5];
        assert_eq!(st.s4_step(0.01), None);
    }

    #[test]
    fn s7_shrinks_the_interval_after_three_stagnant_cycles_then_restores() {
        let mut st = WakeState::for_test();
        st.effective_wake_interval = 4;
        st.s7_step(Some(1.0), 4); // first cycle seeds the EWMA
        assert_eq!(st.s7_ewma_n, 1);
        for _ in 0..2 {
            st.s7_step(Some(1.0), 4); // improvement 0 < 1 %
            assert_eq!(st.effective_wake_interval, 4);
        }
        st.s7_step(Some(1.0), 4); // third stagnant cycle
        assert_eq!(st.effective_wake_interval, 3);
        assert_eq!(st.s7_stagnant, 0);
        // A clear improvement restores one step toward the base interval.
        st.s7_step(Some(0.5), 4);
        assert_eq!(st.effective_wake_interval, 4);
    }

    #[test]
    fn s7_ignores_cycles_without_wake_reconstruction_and_never_goes_below_one() {
        let mut st = WakeState::for_test();
        st.effective_wake_interval = 1;
        st.s7_step(None, 4);
        assert_eq!((st.s7_ewma_n, st.effective_wake_interval), (0, 1));
        st.s7_step(Some(1.0), 4);
        for _ in 0..10 {
            st.s7_step(Some(1.0), 4);
        }
        assert_eq!(st.effective_wake_interval, 1);
    }
}

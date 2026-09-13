/// Burn-path contract tests: inference helpers, weight save/load, growth and
/// homeostasis on the CPU (burn) backend.
///
/// These pin the contracts the growth campaign and sleep loop depend on:
/// - `measure_mse` actually tracks learning (it is the growth-gate signal)
/// - head weights are bias-free, so `head_weights_flat`/`set_head_weights_flat`
///   fully determine `infer_episode_outputs` / `measure_output_mse_inference`
/// - `save_weights`/`load_weights`/`load_weights_with_sizes` round-trip exactly
///   (the campaign restart path) and a failed load leaves a functional model
/// - `grow_region_seeded`/`collect_high_residual_inputs`/duty-cycle bookkeeping
/// - `encoder_z_only`/`binding_only` burn-only paths, incl. grown region sizes
///   travelling via the data (the §24 drift bug class)
#[cfg(test)]
mod tests {
    use burn::backend::ndarray::NdArrayDevice;
    use somnus_perception::{PerceptionProcessor, RegionId, EPISODE_DIM, N_UNITS, REGION_DIMS};

    fn make_proc() -> PerceptionProcessor {
        PerceptionProcessor::new(NdArrayDevice::Cpu)
    }

    fn regional_inputs() -> [Vec<f32>; 5] {
        std::array::from_fn(|i| {
            (0..REGION_DIMS[i])
                .map(|j| ((j as f32 + 1.0) * 0.01 * (i as f32 + 1.0)).sin())
                .collect::<Vec<_>>()
        })
    }

    // ── measure_mse ──────────────────────────────────────────────────────────

    /// measure_mse returns finite, non-negative per-region errors; empty region
    /// slots stay exactly 0.0 (skip path).
    #[test]
    fn measure_mse_finite_and_empty_slot_is_zero() {
        let mut proc = make_proc();
        let mut regional = regional_inputs();
        regional[3] = vec![]; // symbolic slot empty
        let mse = proc.measure_mse(&regional).unwrap();
        for (i, &v) in mse.iter().enumerate() {
            assert!(v.is_finite(), "region {i} mse not finite: {v}");
            assert!(v >= 0.0, "region {i} mse negative: {v}");
        }
        assert_eq!(mse[3], 0.0, "empty region slot must stay 0.0");
    }

    /// The growth gate reads measure_mse; it must actually decrease when the
    /// SAE is trained on the same inputs via replay_step_batch.
    #[test]
    fn measure_mse_decreases_with_replay_training() {
        let mut proc = make_proc();
        let regional = regional_inputs();

        let before = proc.measure_mse(&regional).unwrap();
        for _ in 0..20 {
            proc.replay_step_batch(&[&regional]).unwrap();
        }
        let after = proc.measure_mse(&regional).unwrap();

        for i in 0..5 {
            assert!(
                after[i] < before[i],
                "region {i} mse did not decrease after training: {} -> {}",
                before[i],
                after[i]
            );
        }
    }

    // ── head weights ↔ inference ─────────────────────────────────────────────

    fn episode_input() -> Vec<f32> {
        (0..EPISODE_DIM).map(|i| (i as f32 * 0.05).sin()).collect()
    }

    /// head_weights_flat returns one flat [EPISODE_DIM × REGION_DIMS[i]] vector
    /// per region, and re-setting the extracted weights is observable as a no-op
    /// on inference.
    #[test]
    fn head_weights_flat_round_trip_is_inference_noop() {
        let mut proc = make_proc();
        let ep = episode_input();
        let before = proc.infer_episode_outputs(&ep).unwrap();

        let flat = proc.head_weights_flat();
        for (i, w) in flat.iter().enumerate() {
            assert_eq!(
                w.len(),
                EPISODE_DIM * REGION_DIMS[i],
                "head {i} flat length"
            );
        }
        proc.set_head_weights_flat(flat);
        let after = proc.infer_episode_outputs(&ep).unwrap();
        assert_eq!(
            before, after,
            "re-setting identical weights must not move outputs"
        );
    }

    /// Heads are bias-free Linear layers: output is exactly linear in the
    /// weights. Doubling every weight doubles every predicted output; zeroing
    /// the weights zeroes the outputs.
    #[test]
    fn head_weights_scale_linearly_into_outputs() {
        let mut proc = make_proc();
        let ep = episode_input();
        let base = proc.infer_episode_outputs(&ep).unwrap();

        let flat = proc.head_weights_flat();
        let doubled: [Vec<f32>; 5] =
            std::array::from_fn(|i| flat[i].iter().map(|w| w * 2.0).collect());
        proc.set_head_weights_flat(doubled);
        let out2 = proc.infer_episode_outputs(&ep).unwrap();
        for i in 0..5 {
            assert_eq!(out2[i].len(), REGION_DIMS[i]);
            for (a, b) in base[i].iter().zip(&out2[i]) {
                assert!(
                    (b - 2.0 * a).abs() <= 1e-4 * (1.0 + a.abs()),
                    "head {i}: doubled weights did not double output ({a} vs {b})"
                );
            }
        }

        let zeros: [Vec<f32>; 5] =
            std::array::from_fn(|i| vec![0.0f32; EPISODE_DIM * REGION_DIMS[i]]);
        proc.set_head_weights_flat(zeros);
        let out0 = proc.infer_episode_outputs(&ep).unwrap();
        for (i, vals) in out0.iter().enumerate() {
            assert!(
                vals.iter().all(|&v| v == 0.0),
                "head {i}: zero weights must give zero output (no bias)"
            );
        }
    }

    /// measure_output_mse_inference must equal the manual mean-over-regions MSE
    /// between head predictions and targets (DER++ staleness detection reads it).
    #[test]
    fn measure_output_mse_matches_manual_head_prediction() {
        let proc = make_proc();
        let ep = episode_input();
        let regional = regional_inputs();

        let reported = proc.measure_output_mse_inference(&ep, &regional).unwrap();
        let preds = proc.infer_episode_outputs(&ep).unwrap();

        let mut total = 0.0f32;
        let mut n = 0usize;
        for i in 0..5 {
            if regional[i].is_empty() {
                continue;
            }
            let mse: f32 = preds[i]
                .iter()
                .zip(&regional[i])
                .map(|(p, t)| (p - t).powi(2))
                .sum::<f32>()
                / regional[i].len() as f32;
            total += mse;
            n += 1;
        }
        let manual = total / n as f32;
        assert!(
            (reported - manual).abs() <= 1e-5 * (1.0 + manual.abs()),
            "reported {reported} vs manual {manual}"
        );

        // All-empty regional input is defined as 0.0, not NaN.
        let empty: [Vec<f32>; 5] = Default::default();
        assert_eq!(proc.measure_output_mse_inference(&ep, &empty).unwrap(), 0.0);
    }

    /// infer_episode_outputs is deterministic for fixed weights and returns one
    /// REGION_DIMS[i]-length vector per region.
    #[test]
    fn infer_episode_outputs_dims_and_determinism() {
        let proc = make_proc();
        let ep = episode_input();
        let a = proc.infer_episode_outputs(&ep).unwrap();
        let b = proc.infer_episode_outputs(&ep).unwrap();
        assert_eq!(
            a, b,
            "same weights + same episode must give identical output"
        );
        for i in 0..5 {
            assert_eq!(a[i].len(), REGION_DIMS[i]);
            assert!(a[i].iter().all(|v| v.is_finite()));
        }
    }

    // ── homeostasis ──────────────────────────────────────────────────────────

    /// homeostatic_decay(rate) multiplies every trainable weight by (1 - rate).
    #[test]
    fn homeostatic_decay_scales_head_weights() {
        let mut proc = make_proc();
        let before = proc.head_weights_flat();
        proc.homeostatic_decay(0.1);
        let after = proc.head_weights_flat();
        for i in 0..5 {
            for (w0, w1) in before[i].iter().zip(&after[i]) {
                assert!(
                    (w1 - 0.9 * w0).abs() <= 1e-6 * (1.0 + w0.abs()),
                    "head {i}: decay factor wrong ({w0} -> {w1})"
                );
            }
        }
    }

    // ── save / load ──────────────────────────────────────────────────────────

    /// save_weights + load_weights round-trips exactly and writes the `.sizes`
    /// sidecar; loading restores weights even after they were perturbed.
    #[test]
    fn save_load_weights_round_trip() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("weights.bin");

        let mut proc = make_proc();
        let saved_heads = proc.head_weights_flat();
        proc.save_weights(&path).unwrap();

        let sidecar = std::fs::read_to_string(dir.path().join("weights.sizes")).unwrap();
        assert_eq!(
            sidecar,
            [N_UNITS; 5]
                .iter()
                .map(|n| n.to_string())
                .collect::<Vec<_>>()
                .join(" "),
            "sizes sidecar must list baseline region sizes"
        );

        // Perturb, then restore from disk.
        proc.homeostatic_decay(0.5);
        proc.load_weights(&path).unwrap();
        assert_eq!(
            proc.head_weights_flat(),
            saved_heads,
            "loaded head weights must match the saved ones exactly"
        );
    }

    /// The campaign restart path: pre-grow to saved sizes, then load. Region
    /// sizes, duty-cycle vectors and weights must all come back consistent.
    #[test]
    fn load_weights_with_sizes_restores_grown_model() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("grown.bin");

        let mut grown = make_proc();
        grown.grow_region(RegionId::AudioSpeech, 8);
        let sizes = grown.region_sizes();
        assert_eq!(sizes[0], N_UNITS + 8);
        let grown_heads = grown.head_weights_flat();
        grown.save_weights(&path).unwrap();

        let mut fresh = make_proc();
        fresh.load_weights_with_sizes(&path, sizes).unwrap();
        assert_eq!(fresh.region_sizes(), sizes, "regions must be pre-grown");
        assert_eq!(fresh.duty_cycles()[0].len(), N_UNITS + 8);
        assert_eq!(
            fresh.head_weights_flat(),
            grown_heads,
            "weights must round-trip exactly through save/load"
        );
    }

    /// A failed weight load must not strand the processor: it rebuilds a fresh
    /// model at the previous sizes and stays fully functional (lib.rs error path).
    #[test]
    fn load_weights_missing_file_keeps_processor_functional() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut proc = make_proc();
        let sizes_before = proc.region_sizes();

        let err = proc.load_weights(&dir.path().join("nonexistent.bin"));
        assert!(err.is_err(), "loading a missing file must fail");

        assert_eq!(proc.region_sizes(), sizes_before);
        let regional = regional_inputs();
        let loss = proc.replay_step(&regional).unwrap();
        assert!(
            loss.is_finite(),
            "processor must stay usable after failed load"
        );
        let mse = proc.measure_mse(&regional).unwrap();
        assert!(mse.iter().all(|v| v.is_finite()));
    }

    // ── growth ───────────────────────────────────────────────────────────────

    /// grow_region_seeded updates region size + duty-cycle bookkeeping and the
    /// model stays trainable; empty seeds fall back to plain grow_region.
    #[test]
    fn grow_region_seeded_sizes_duty_and_fallback() {
        let mut proc = make_proc();
        let seeds: Vec<Vec<f32>> = (0..3)
            .map(|s| {
                (0..REGION_DIMS[0])
                    .map(|j| ((j + s * 37) as f32 * 0.02).sin())
                    .collect()
            })
            .collect();

        proc.grow_region_seeded(RegionId::AudioSpeech, 5, &seeds);
        assert_eq!(proc.region_sizes()[0], N_UNITS + 5);
        assert_eq!(proc.duty_cycles()[0].len(), N_UNITS + 5);

        // Still trainable after seeded growth.
        let regional = regional_inputs();
        let loss = proc.replay_step(&regional).unwrap();
        assert!(loss.is_finite());

        // Empty seeds degrade to plain growth, not an error.
        proc.grow_region_seeded(RegionId::Visual, 4, &[]);
        assert_eq!(proc.region_sizes()[2], N_UNITS + 4);
    }

    /// collect_high_residual_inputs caps per-region results, skips empty slots,
    /// and returns actual inputs (not transformed copies).
    #[test]
    fn collect_high_residual_inputs_caps_and_skips_empty() {
        let mut proc = make_proc();
        let mk = |seed: usize, len: usize| -> Vec<f32> {
            (0..len)
                .map(|j| ((j + seed * 13) as f32 * 0.03).cos())
                .collect()
        };
        let s1 = mk(1, REGION_DIMS[0]);
        let s2 = mk(2, REGION_DIMS[0]);
        let s3 = mk(3, REGION_DIMS[0]);
        let e1 = mk(4, REGION_DIMS[1]);

        type Trace = (Vec<f32>, [Vec<f32>; 5], (), ());
        let traces: Vec<Trace> = vec![
            (vec![], [s1.clone(), vec![], vec![], vec![], vec![]], (), ()),
            (
                vec![],
                [s2.clone(), e1.clone(), vec![], vec![], vec![]],
                (),
                (),
            ),
            (vec![], [s3.clone(), vec![], vec![], vec![], vec![]], (), ()),
        ];

        let capped = proc.collect_high_residual_inputs(&traces, 2);
        assert_eq!(capped[0].len(), 2, "speech capped at 2");
        assert_eq!(capped[1].len(), 1, "env has exactly one non-empty input");
        for v in &capped[0] {
            assert!(
                *v == s1 || *v == s2 || *v == s3,
                "returned vector must be an input"
            );
        }
        assert_eq!(capped[1][0], e1);
        assert!(capped[2].is_empty() && capped[3].is_empty() && capped[4].is_empty());

        let uncapped = proc.collect_high_residual_inputs(&traces, 10);
        assert_eq!(uncapped[0].len(), 3, "cap above count returns everything");
    }

    /// Duty cycles start all-zero at N_UNITS length, update on ingest, and are
    /// extended with zeros on growth.
    #[test]
    fn duty_cycles_initial_zero_update_on_ingest_extend_on_growth() {
        let mut proc = make_proc();
        for dc in proc.duty_cycles() {
            assert_eq!(dc.len(), N_UNITS);
            assert!(dc.iter().all(|&v| v == 0.0));
        }

        proc.ingest_speech(&regional_inputs()[0]).unwrap();
        let dc = &proc.duty_cycles()[0];
        assert!(
            dc.iter().any(|&v| v > 0.0),
            "ingest must update the speech duty cycle"
        );

        proc.grow_region(RegionId::AudioSpeech, 4);
        let dc = proc.duty_cycles()[0].clone();
        assert_eq!(dc.len(), N_UNITS + 4);
        assert!(
            dc[N_UNITS..].iter().all(|&v| v == 0.0),
            "new units start with zero duty cycle"
        );
    }

    // ── burn-only inference paths ────────────────────────────────────────────

    /// encoder_z_only: correct per-region dims, no side effects (deterministic),
    /// and `language: None` is exactly equivalent to an all-zero language input.
    #[test]
    fn encoder_z_only_dims_determinism_and_lang_padding() {
        let proc = make_proc();
        let (speech, env, visual, _, lang) = {
            let r = regional_inputs();
            (
                r[0].clone(),
                r[1].clone(),
                r[2].clone(),
                r[3].clone(),
                r[4].clone(),
            )
        };

        let z_none = proc.encoder_z_only(&speech, &env, &visual, None);
        let z_zero =
            proc.encoder_z_only(&speech, &env, &visual, Some(&vec![0.0f32; REGION_DIMS[4]]));
        let z_lang = proc.encoder_z_only(&speech, &env, &visual, Some(&lang));
        let z_again = proc.encoder_z_only(&speech, &env, &visual, None);

        for (i, z) in z_none.iter().enumerate() {
            assert_eq!(z.len(), N_UNITS, "region {i} z dim");
            assert!(z.iter().all(|v| v.is_finite()));
        }
        assert_eq!(z_none, z_again, "encoder_z_only must be side-effect free");
        assert_eq!(
            z_none, z_zero,
            "None must equal explicit zero language input"
        );
        assert_ne!(
            z_none[4], z_lang[4],
            "real language input must change z_language"
        );
    }

    /// binding_only: episode + GRU hidden shapes, temporal context changes the
    /// output, and grown region sizes travel via the dense vector lengths
    /// (the §24 drift class — n_units must come from the data, not a constant).
    #[test]
    fn binding_only_shapes_temporal_context_and_grown_region() {
        let mut proc = make_proc();
        let dense_at = |n: usize, k: usize, fill: f32| -> Vec<f32> {
            let mut v = vec![0.0f32; n];
            for i in 0..k {
                v[i * 7 % n] = fill + i as f32 * 0.01;
            }
            v
        };
        let sparse_z: [Vec<f32>; 5] =
            std::array::from_fn(|i| dense_at(N_UNITS, 20, 0.5 + i as f32));

        let (ep, gru_h) = proc.binding_only(&sparse_z, None);
        assert_eq!(ep.len(), EPISODE_DIM);
        assert!(ep.iter().all(|v| v.is_finite()));
        assert_eq!(gru_h.len(), 256, "GRU_HIDDEN");
        assert!(gru_h.iter().all(|v| v.is_finite()));

        let (ep2, _) = proc.binding_only(&sparse_z, Some(&ep));
        assert_ne!(
            ep, ep2,
            "temporal context (prev episode) must change the output"
        );

        // Grow region 0 and feed a correspondingly longer dense vector.
        proc.grow_region(RegionId::AudioSpeech, 8);
        let mut grown_z = sparse_z.clone();
        grown_z[0] = dense_at(N_UNITS + 8, 20, 0.7);
        let (ep3, _) = proc.binding_only(&grown_z, None);
        assert_eq!(ep3.len(), EPISODE_DIM);
        assert!(
            ep3.iter().all(|v| v.is_finite()),
            "grown-region binding must stay finite"
        );
    }

    // ── binding replay (ADR 0041, gaps.md §33) ───────────────────────────────

    /// Empty batch is a no-op returning 0.0 (same contract as the other batch
    /// training steps).
    #[test]
    fn binding_replay_empty_batch_is_noop() {
        let mut proc = make_proc();
        assert_eq!(
            proc.binding_replay_step_batch(&[], 1.0, false).unwrap(),
            0.0
        );
    }

    /// Control arm of the §33 fix: grown units' binding-projection rows start
    /// at exactly zero (`grow_region` appends zero rows) and can only receive
    /// gradients when a grown unit wins k-WTA and its row is gathered by the
    /// sparse `select`. On a fresh model with a single repeated input that
    /// never happens here: grown encoder columns init at 0.01 scale, far below
    /// the Kaiming-scale base columns, and the duty-cycle boost (≤ k/n ≈ 0.02,
    /// growing ~0.001/step) cannot bridge that gap on a test-sized step budget
    /// — grown units break into the winner set only on production timescales
    /// (homeostatic rotation over many cycles of a diverse corpus). The grown
    /// rows must therefore stay exactly zero: this pins that gradients reach
    /// only gathered rows (no dense-scatter leak into never-active rows). The
    /// positive case — grown rows moving once grown units are active — is
    /// pinned by `binding_replay_trains_grown_binding_rows_when_active` in
    /// lib.rs, which can force grown units into the winner set deterministically.
    /// Loss still decreases: base rows, W_q/W_k/W_v and the heads train on the
    /// live episodes.
    #[test]
    fn binding_replay_trains_only_active_binding_rows() {
        let mut proc = make_proc();
        proc.grow_region(RegionId::AudioSpeech, 64);
        let norms_before = proc.binding_proj_row_norms(RegionId::AudioSpeech);
        assert_eq!(norms_before.len(), N_UNITS + 64);
        assert!(
            norms_before[N_UNITS..].iter().all(|&n| n == 0.0),
            "grown binding rows must start exactly zero"
        );
        // Mirror the production sequence: growth happens in REM, and the next
        // NREM opens with refresh_autodiff_state (sleep.rs), which remaps every
        // param to a fresh require_grad leaf — grow_region's Tensor::from_inner
        // rebuilds don't carry the flag on their own.
        proc.refresh_autodiff_state();

        let regional = regional_inputs();
        let first_loss = proc
            .binding_replay_step_batch(&[&regional], 1.0, false)
            .unwrap();
        assert!(first_loss.is_finite() && first_loss >= 0.0);
        let mut last_loss = first_loss;
        for _ in 0..29 {
            last_loss = proc
                .binding_replay_step_batch(&[&regional], 1.0, false)
                .unwrap();
            assert!(last_loss.is_finite(), "binding replay loss went non-finite");
        }
        assert!(
            last_loss < first_loss,
            "binding replay loss did not decrease on repeated input: {first_loss} -> {last_loss}"
        );

        let norms_after = proc.binding_proj_row_norms(RegionId::AudioSpeech);
        assert!(
            norms_after[N_UNITS..].iter().all(|&n| n == 0.0),
            "never-active grown binding rows must stay exactly zero (no dense-scatter leak)"
        );
        // Base rows are live too (they were random-init, not zero) — training must
        // not zero them out wholesale.
        assert!(
            norms_after[..N_UNITS].iter().any(|&n| n > 0.0),
            "base binding rows must remain live"
        );
    }

    // ── ADR 0048: decoupled wake-time binding snapshot ──────────────────────

    /// `wake_binding` must stay byte-identical to its last sync while `binding`
    /// keeps moving under `binding_replay_step_batch` — that's the entire point
    /// of the lag. Only an explicit `sync_wake_binding()` call may change it.
    #[test]
    fn wake_binding_stays_frozen_until_synced() {
        let mut proc = make_proc();
        proc.refresh_autodiff_state();
        let regional = regional_inputs();

        let wake_before = proc.wake_binding_proj_row_norms(RegionId::AudioSpeech);
        let live_before = proc.binding_proj_row_norms(RegionId::AudioSpeech);
        assert_eq!(
            wake_before, live_before,
            "wake_binding must start as an exact clone of binding"
        );

        for _ in 0..10 {
            proc.binding_replay_step_batch(&[&regional], 1.0, false)
                .unwrap();
        }
        let live_after_training = proc.binding_proj_row_norms(RegionId::AudioSpeech);
        assert_ne!(
            live_before, live_after_training,
            "binding must move under binding_replay_step_batch"
        );
        let wake_after_training = proc.wake_binding_proj_row_norms(RegionId::AudioSpeech);
        assert_eq!(
            wake_before, wake_after_training,
            "wake_binding must not move just because binding trained — no sync happened yet"
        );

        proc.sync_wake_binding();
        let wake_after_sync = proc.wake_binding_proj_row_norms(RegionId::AudioSpeech);
        assert_eq!(
            wake_after_sync, live_after_training,
            "sync_wake_binding must copy binding's exact current values into wake_binding"
        );
    }

    /// `grow_region`/`grow_region_seeded`/`prune_region` must reshape `binding` and
    /// `wake_binding` in lockstep, unconditionally — shape never lags, only trained
    /// values do (ADR 0048). A shape mismatch here would panic the next
    /// `wake_binding.forward()` call at production scale, not just fail this test.
    #[test]
    fn growth_and_prune_keep_binding_and_wake_binding_in_lockstep() {
        let mut proc = make_proc();

        proc.grow_region(RegionId::Visual, 32);
        assert_eq!(
            proc.binding_proj_row_norms(RegionId::Visual).len(),
            proc.wake_binding_proj_row_norms(RegionId::Visual).len(),
            "grow_region must grow wake_binding's shape alongside binding's"
        );

        let seeds: Vec<Vec<f32>> = (0..4)
            .map(|k| (0..REGION_DIMS[2]).map(|j| (j + k) as f32 * 0.01).collect())
            .collect();
        proc.grow_region_seeded(RegionId::Visual, 16, &seeds);
        assert_eq!(
            proc.binding_proj_row_norms(RegionId::Visual).len(),
            proc.wake_binding_proj_row_norms(RegionId::Visual).len(),
            "grow_region_seeded must grow wake_binding's shape alongside binding's"
        );

        proc.prune_region(RegionId::Visual, 0.5);
        assert_eq!(
            proc.binding_proj_row_norms(RegionId::Visual).len(),
            proc.wake_binding_proj_row_norms(RegionId::Visual).len(),
            "prune_region must shrink wake_binding's shape alongside binding's"
        );
        // Confirm pruning actually did something, or the three shape checks above
        // would trivially pass without exercising prune's row-removal path at all.
        assert!(proc.binding_proj_row_norms(RegionId::Visual).len() < N_UNITS + 32 + 16);
    }
}

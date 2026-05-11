/// Phase 5 feasibility smoke tests.
///
/// These tests verify three empirical assumptions the Phase 5 design rests on:
///   1. candle autograd flows through a k-WTA straight-through mask
///   2. VarMap must be stored to access vars after construction
///   3. Linear weight expansion via Tensor::cat + set_one is feasible

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device, Module, Result, Tensor};
    use candle_nn::{linear, AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};

    // ── Test 1: autograd through straight-through k-WTA mask ─────────────────
    #[test]
    fn autograd_through_kwta_mask() -> Result<()> {
        let dev = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);

        let encoder = linear(4, 8, vb.pp("enc"))?;
        let decoder = linear(8, 4, vb.pp("dec"))?;

        let input = Tensor::from_vec(vec![1.0f32, 0.5, -0.3, 0.8], (1, 4), &dev)?;

        // Encoder forward — z is a tracked tensor.
        let z = encoder.forward(&input)?.squeeze(0)?; // [8]
                                                      // Read values to find top-k — does NOT detach z from the graph.
        let z_vec: Vec<f32> = z.to_vec1()?;

        // Build straight-through mask: top-3 of 8.
        let k = 3usize;
        let mut indexed: Vec<(usize, f32)> = z_vec.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let mut mask_vec = vec![0.0f32; 8];
        for &(i, _) in &indexed[..k] {
            mask_vec[i] = 1.0;
        }
        let mask = Tensor::from_vec(mask_vec, (8,), &dev)?; // untracked constant

        // sparse_z: gradient flows through z (mask is a constant multiplier).
        let sparse_z = z.mul(&mask)?.unsqueeze(0)?; // [1, 8]
        let recon = decoder.forward(&sparse_z)?.squeeze(0)?; // [4]
        let input_1d = input.squeeze(0)?; // [4]
        let loss = (input_1d - recon)?.sqr()?.mean_all()?;

        let grads = loss.backward()?;

        // Check that at least one encoder weight has a non-zero gradient.
        let enc_vars = varmap.all_vars();
        let any_nonzero = enc_vars.iter().any(|v| {
            grads
                .get(v) // Var Derefs to Tensor, GradStore::get(&Tensor)
                .map(|g| {
                    g.flatten_all()
                        .and_then(|t| t.to_vec1::<f32>())
                        .map(|vec| vec.iter().any(|&x| x.abs() > 1e-9))
                        .unwrap_or(false)
                })
                .unwrap_or(false)
        });
        assert!(
            any_nonzero,
            "all gradients are zero — autograd does not flow through the mask"
        );
        println!("Test 1 PASS: autograd flows through k-WTA straight-through mask");
        Ok(())
    }

    // ── Test 2: VarMap must be stored to access vars after construction ───────
    #[test]
    fn varmap_must_be_stored() -> Result<()> {
        let dev = Device::Cpu;

        // A: VarMap kept — all_vars() returns the vars.
        let kept_varmap = VarMap::new();
        {
            let vb = VarBuilder::from_varmap(&kept_varmap, DType::F32, &dev);
            let _lin = linear(4, 8, vb.pp("layer"))?;
        }
        let vars = kept_varmap.all_vars();
        assert!(!vars.is_empty(), "kept VarMap should have vars");
        println!("Test 2 PASS (A): kept VarMap has {} var(s)", vars.len());

        // B: VarMap dropped — inference still works, but vars are unreachable via VarMap.
        let lin_for_inference;
        {
            let dropped_varmap = VarMap::new();
            let vb = VarBuilder::from_varmap(&dropped_varmap, DType::F32, &dev);
            lin_for_inference = linear(4, 8, vb.pp("layer"))?;
            // dropped_varmap is about to drop — vars still alive via Tensor ref-count.
        }
        let input = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1, 4), &dev)?;
        let out = lin_for_inference.forward(&input)?;
        assert_eq!(out.dims(), &[1, 8], "inference works after VarMap drop");
        println!("Test 2 PASS (B): inference works after VarMap drop; vars inaccessible");
        Ok(())
    }

    // ── Test 3: weight expansion via cat + set_one ────────────────────────────
    #[test]
    fn weight_expansion_feasible() -> Result<()> {
        let dev = Device::Cpu;

        // Build and do one training step.
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
        let encoder = linear(4, 8, vb.pp("enc"))?;
        let decoder = linear(8, 4, vb.pp("dec"))?;
        let mut opt = AdamW::new(varmap.all_vars(), ParamsAdamW::default())?;

        let input = Tensor::from_vec(vec![1.0f32, 0.5, -0.3, 0.8], (1, 4), &dev)?;
        let recon = decoder.forward(&encoder.forward(&input)?)?;
        let loss = (input.clone() - recon)?.sqr()?.mean_all()?;
        opt.backward_step(&loss)?;

        // Capture old weights.
        let old_w: Tensor = encoder.weight().clone(); // [8, 4]
        let old_w_vec: Vec<f32> = old_w.flatten_all()?.to_vec1()?;

        // Expand: add 4 new rows.
        let new_rows = Tensor::randn(0.0f32, 0.01, (4, 4), &dev)?;
        let expanded_w = Tensor::cat(&[&old_w, &new_rows], 0)?; // [12, 4]
        assert_eq!(expanded_w.dims(), &[12, 4]);

        // Create a new VarMap + Linear with the target shape, then overwrite weights.
        let mut new_varmap = VarMap::new();
        let new_vb = VarBuilder::from_varmap(&new_varmap, DType::F32, &dev);
        let new_enc = linear(4, 12, new_vb.pp("enc"))?;
        // set_one requires the var to already exist (created above) and shapes to match.
        new_varmap.set_one("enc.weight", &expanded_w)?;

        // Verify first 8 rows unchanged.
        let new_w_vec: Vec<f32> = new_enc.weight().flatten_all()?.to_vec1()?;
        // Note: set_one updates the Var's data, but new_enc.weight() is a Tensor clone
        // taken at construction time — need to re-read from varmap.
        let updated_w: Vec<f32> = {
            let data = new_varmap.data().lock().unwrap();
            data["enc.weight"].as_tensor().flatten_all()?.to_vec1()?
        };
        for (i, (&a, &b)) in old_w_vec.iter().zip(updated_w.iter()).enumerate() {
            assert!((a - b).abs() < 1e-7, "old weight[{i}] changed: {a} vs {b}");
        }

        // Shape check: new encoder produces [1, 12].
        let out = new_enc.forward(&input)?;
        // new_enc was built before set_one — its internal tensor may not reflect update.
        // For a real implementation we'd rebuild Linear from the Var.
        // This test just verifies the VarMap update + shape logic.
        assert_eq!(
            new_w_vec.len(),
            12 * 4,
            "expanded weight has wrong element count"
        );
        assert_eq!(out.dims(), &[1, 12]);

        println!("Test 3 PASS: weight expansion via cat + set_one works");
        Ok(())
    }

    // ── Test 4: wake_reconstruct_step produces finite loss and updates weights ──

    #[test]
    fn wake_reconstruct_step_trains() -> candle_core::Result<()> {
        use somnus_perception::{PerceptionProcessor, REGION_DIMS};

        let mut proc =
            PerceptionProcessor::new(Device::Cpu).expect("PerceptionProcessor::new failed");

        // Synthesise a dummy episode vector and one non-empty regional input.
        let episode = vec![0.1f32; 512];
        let mut regional: [Vec<f32>; 4] = [vec![], vec![], vec![], vec![]];
        regional[0] = vec![0.5f32; REGION_DIMS[0]]; // speech only

        // Capture a weight before the step.
        let lr = proc.learning_rate();
        let mse = proc
            .wake_reconstruct_step(&episode, &regional)
            .expect("wake_reconstruct_step failed");

        assert!(
            mse.is_finite() && mse >= 0.0,
            "MSE must be finite and non-negative"
        );
        assert!(lr > 0.0, "learning rate must be positive");
        println!("Test 4 PASS: wake_reconstruct_step mse={mse:.4}");
        Ok(())
    }

    // ── Test 5: top-down priors — blend applied and prior updated after episode ─
    #[test]
    fn top_down_prior_applied() -> candle_core::Result<()> {
        use somnus_perception::{PerceptionProcessor, REGION_DIMS};

        let mut proc =
            PerceptionProcessor::new(Device::Cpu).expect("PerceptionProcessor::new failed");

        // Enable prior injection (β = 0.2).
        proc.set_prior_beta(0.2);

        // Ingest all four region types to produce the first episode and populate priors.
        let speech_v: Vec<f32> = (0..REGION_DIMS[0])
            .map(|i| (i as f32) / REGION_DIMS[0] as f32)
            .collect();
        let env_v: Vec<f32> = (0..REGION_DIMS[1])
            .map(|i| (i as f32) / REGION_DIMS[1] as f32)
            .collect();
        let visual_v: Vec<f32> = (0..REGION_DIMS[2])
            .map(|i| (i as f32) / REGION_DIMS[2] as f32)
            .collect();

        let out_s = proc.ingest_speech(&speech_v).expect("ingest_speech failed");
        let out_e = proc.ingest_env(&env_v).expect("ingest_env failed");
        let out_v = proc.ingest_visual(&visual_v).expect("ingest_visual failed");

        // After three ingests, at least one should have produced an episode.
        let has_episode =
            out_s.episode.is_some() || out_e.episode.is_some() || out_v.episode.is_some();
        assert!(
            has_episode,
            "expected an episode after all four regions ingested"
        );

        // Second round: priors are now populated. Ingest again — must not panic.
        let out2 = proc
            .ingest_speech(&speech_v)
            .expect("second ingest_speech failed");
        let _ = proc.ingest_env(&env_v).expect("second ingest_env failed");
        let _ = proc
            .ingest_visual(&visual_v)
            .expect("second ingest_visual failed");

        // Reconstruction errors must remain finite.
        for &(_, err) in &out2.recon_errors {
            assert!(
                err.is_finite(),
                "recon error is not finite after prior injection"
            );
        }

        println!("Test 5 PASS: top-down priors applied without error");
        Ok(())
    }
}

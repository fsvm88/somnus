//! Handlers for the events the wake loop in `run()` reacts to. Each used to be pasted inline in
//! two places (the pre-sleep drain and the `select!` arm) and the copies had drifted apart.

use std::{
    ops::ControlFlow,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc,
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use somnus_memory::{
    save_sal_ewma, save_settle_ewma, save_wake_ewma, EpisodicBuffer, RecollectionLayer,
};
use somnus_perception::{
    IngestPerfBreakdown, PerceptionOutput, PerceptionProcessor, PerceptionResult, WakeReconWorker,
};

use crate::compute_effective_sleep_max_secs;
use crate::current_process_rss_bytes;
use crate::dissipate_wakefulness_pressure;
use crate::memory_dir_disk_free_bytes;
use crate::recompute_max_wake_secs;
use crate::should_use_ort_inference;
use crate::sink_flush;
use crate::sleep;
use crate::update_wake_ewma;
use crate::{
    accumulate_cosines, accumulate_salience, accumulate_wakefulness_pressure,
    action::{ActionDecoder, GainWorker},
    build_query_with_history,
    cli::Args,
    episode_cosine_mean, is_introspective_query, maybe_startup_query, query_recalled_episodes,
    record_episode, sal_gate, settle_gate_val,
    sleep::SleepStats,
    update_perf_ewma, update_sal_ewma, update_settle_ewma,
    wake_state::WakeState,
    write_training_pair, LangIngestResult,
};
use tracing::info;

/// Answer a stdin query with the current episode as context (ADR 0012, R1): the query text is
/// also stored as the latest episode's `text_target` so NREM can train on it. `stage` only labels
/// the log lines ("pre-sleep" when drained ahead of a sleep cycle, so queries are never blocked
/// behind it; "live" from the `select!` arm).
pub(crate) fn answer_query(
    query: &str,
    stage: &str,
    episodic: &mut EpisodicBuffer,
    decoder: &mut Option<ActionDecoder>,
    recall: &RecollectionLayer,
    args: &Args,
) {
    tracing::info!(query = %query, stage, "query received");
    // Collect recent text history before borrowing the decoder (R3 Phase 1).
    let raw_targets = if is_introspective_query(query) {
        episodic.recent_text_targets(8)
    } else {
        vec![]
    };
    let ep_info = episodic
        .iter_traces()
        .last()
        .map(|t| (t.record.id, t.record.values.clone()));
    let Some((last_id, ep)) = ep_info else {
        println!(
            "{}",
            serde_json::json!({"type":"action_response","error":"no episode yet"})
        );
        return;
    };
    let Some(dec) = decoder.as_mut() else {
        return;
    };
    // Store the query as text_target before generation so it is available for NREM.
    if let Ok(ids) = dec.tokenize(query) {
        episodic.set_last_text_target(ids);
    }
    let effective_query = build_query_with_history(query, raw_targets, dec);
    let recalled = query_recalled_episodes(recall, &ep, Some(last_id), args.recall_k);
    tracing::info!(
        recall_n = recalled.len(),
        introspective = effective_query.len() != query.len(),
        stage,
        "starting generation"
    );
    // ActionDecoder is not Send: generate synchronously, blocking this async task.
    let max_tokens = args.action_max_tokens;
    let result =
        tokio::task::block_in_place(|| dec.generate(&ep, &recalled, &effective_query, max_tokens));
    match result {
        Ok(response) => {
            let ts_ms = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_millis())
                .unwrap_or(0);
            tracing::info!(
                tokens = response.split_whitespace().count(),
                "generation complete"
            );
            println!(
                "{}",
                serde_json::json!({
                    "type": "action_response",
                    "ts": ts_ms,
                    "query": query,
                    "response": response,
                })
            );
        }
        Err(e) => {
            tracing::warn!(?e, "generation failed");
            somnus_metrics::counted_core!(somnus_metrics::Metric::ActionGenerationError, 1);
        }
    }
}

/// Fold one ingested stdin/ASR language line into the wake-cycle statistics, the adaptive EWMAs,
/// the settle gate, the gain worker and the training-data export.
///
/// Two inline copies of this lived in `run()` (pre-sleep drain and `select!` arm) and had
/// drifted: the drain copy skipped `perception_total_ms`, the gain-worker request and the HNSW
/// query stats; the arm copy skipped the settle-trigger stats. This is their union.
pub(crate) fn record_language_result(
    st: &mut WakeState,
    perception: &mut PerceptionProcessor,
    gain_worker: Option<&GainWorker>,
    args: &Args,
    r: &LangIngestResult,
    text: &str,
) {
    st.lang_ingest_count += 1;
    st.perception_total_ms += r.perception_ms;
    if r.episode_produced {
        st.wake_episodes += 1;
        if let Some(m) = episode_cosine_mean(&r.prior_cosines) {
            update_perf_ewma(
                m,
                &mut st.cosine_ewma,
                &mut st.cosine_baseline,
                &mut st.cosine_baseline_accum,
                &mut st.cosine_baseline_n,
                &mut st.perf_sleep_flag,
                args.perf_sleep_threshold,
            );
            update_settle_ewma(m, &mut st.settle_mu, &mut st.settle_n);
            perception.set_settle_gate(settle_gate_val(
                st.settle_mu,
                st.settle_n,
                args.settle_gate,
            ));
        }
        if let (Some(gw), Some(ep)) = (gain_worker, r.last_episode.as_ref()) {
            st.gain_episode_counter += 1;
            if st.gain_episode_counter.is_multiple_of(args.gain_interval) {
                gw.request_gain(ep.clone());
            }
        }
    }
    if let Some(m) = r.reconstruct_mse {
        st.wake_output_mse_sum += m as f64;
        st.wake_output_mse_n += 1;
    }
    if r.throttle_passed {
        st.wake_reconstruct_throttle_passed += 1;
    }
    if r.settle_iters > 0 {
        st.settle_trigger_count += 1;
        st.settle_iters_sum += r.settle_iters;
        st.settle_delta_sum += r.settle_delta as f64;
    }
    if let Some(es) = r.ep_stats.as_ref() {
        let sal_thresh = sal_gate(st.sal_mu, st.sal_sigma, st.sal_n);
        accumulate_salience(
            es,
            sal_thresh,
            &mut st.salience_sum,
            &mut st.salience_n,
            &mut st.salience_min,
            &mut st.high_salience_episodes,
            &mut st.blend_count,
            &mut st.episodes_speech,
            &mut st.episodes_env,
            &mut st.episodes_visual,
            &mut st.episodes_language,
            &mut st.episodes_mixed,
        );
        update_sal_ewma(
            es.salience,
            &mut st.sal_mu,
            &mut st.sal_sigma,
            &mut st.sal_n,
        );
        accumulate_wakefulness_pressure(es.salience, sal_thresh, &mut st.wakefulness_pressure);
        st.hnsw_query_us_sum += es.hnsw_query_us;
        st.hnsw_query_n += 1;
    }
    accumulate_cosines(
        &r.prior_cosines,
        &mut st.prior_cosine_sums,
        &mut st.prior_cosine_counts,
    );
    // Inline training data export (ADR 0033): write the (episode, text) pair at ingest time,
    // capturing every ASR/text event.
    if let (Some(export_path), Some(ep)) = (&args.export_training_data, &r.last_episode) {
        write_training_pair(export_path, ep, text);
    }
}

/// Everything the per-cycle metrics line needs that is not already in `WakeState`/`SleepStats`/`Args`.
pub(crate) struct CycleContext {
    pub(crate) sleep_trigger: &'static str,
    pub(crate) cycle_asr: usize,
    pub(crate) region_sizes: [usize; 5],
    pub(crate) warm_len: usize,
    pub(crate) cold_len: usize,
    pub(crate) prior_betas: [f32; 5],
    pub(crate) perf_breakdown: IngestPerfBreakdown,
    pub(crate) fine_metrics: serde_json::Value,
    pub(crate) effective_max_cold: usize,
    pub(crate) effective_sleep_max_secs: u64,
    /// Mean wake-phase output MSE of the cycle just finished (None when no wake reconstruction ran).
    pub(crate) mean_mse_wake: Option<f64>,
}

/// The NDJSON metrics line written after every sleep cycle (`--metrics-file`). The key set is
/// consumed by `scripts/campaign_status.py` / `eval_metrics.py`; `metrics_line_keys_are_stable`
/// pins it.
pub(crate) fn cycle_metrics_json(
    st: &WakeState,
    stats: &SleepStats,
    args: &Args,
    c: CycleContext,
) -> serde_json::Value {
    let CycleContext {
        sleep_trigger,
        cycle_asr,
        region_sizes,
        warm_len,
        cold_len,
        prior_betas,
        perf_breakdown,
        fine_metrics,
        effective_max_cold,
        effective_sleep_max_secs,
        mean_mse_wake,
    } = c;
    let grew = !stats.regions_grown.is_empty();
    let prior_cosine_mean: Vec<Option<f64>> = (0..5)
        .map(|i| {
            if st.prior_cosine_counts[i] > 0 {
                Some(st.prior_cosine_sums[i] / st.prior_cosine_counts[i] as f64)
            } else {
                None
            }
        })
        .collect();
    // ADR 0022: salience stats (None when warm was empty all cycle).
    let mean_salience_wake = if st.salience_n > 0 {
        Some(st.salience_sum / st.salience_n as f64)
    } else {
        None
    };
    let min_salience_wake = if st.salience_n > 0 {
        Some(st.salience_min)
    } else {
        None
    };
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0);
    serde_json::json!({
        "cycle":             st.sleep_cycle_count,
        "session":           if args.session > 0 { Some(args.session) } else { None },
        "ts":                ts,
        "sleep_trigger":     sleep_trigger,
        "traces_replayed":   stats.traces_replayed,
        "nrem_budget_traces":        stats.nrem_budget_traces,
        "nrem_unique_traces":        stats.nrem_unique_traces,
        "mean_loss":         stats.mean_train_loss,
        "mean_output_mse":   stats.mean_output_mse,
        "per_region_mse":    stats.per_region_mse,
        "warm_recall_l2":    stats.warm_recall_l2,
        "gist_recall_rate":        stats.gist_recall_rate,
        "within_session_gist_rate": stats.within_session_gist_rate,
        "cross_modal_gist_entropy": stats.cross_modal_gist_entropy,
        "was_interrupted":   stats.was_interrupted,
        "nrem_exit_reason":  stats.nrem_exit_reason,
        "region_sizes":      region_sizes,
        "warm_len":          warm_len,
        "cold_len":          cold_len,
        "grew":              grew,
        "regions_grown":     stats.regions_grown.iter().map(|r| r.as_index()).collect::<Vec<_>>(),
        "growth_cooldown_remaining": stats.growth_cooldown_remaining,
        "wake_episodes":     st.wake_episodes,
        "episodes_speech":   st.episodes_speech,
        "episodes_env":      st.episodes_env,
        "episodes_visual":   st.episodes_visual,
        "episodes_language": st.episodes_language,
        "episodes_mixed":    st.episodes_mixed,
        "asr_count":         cycle_asr,
        "lang_ingest_count": st.lang_ingest_count,
        "mean_salience_wake":       mean_salience_wake,
        "min_salience_wake":        min_salience_wake,
        "high_salience_episodes":   st.high_salience_episodes,
        "blend_count":              st.blend_count,
        "wake_reconstruct_attempts":       st.wake_reconstruct_counter,
        "wake_reconstruct_throttle_passed": st.wake_reconstruct_throttle_passed,
        "wake_reconstruct_fired":          st.wake_output_mse_n,
        "wake_recon_dropped":              st.wake_recon_dropped,
        "settle_trigger_count":  st.settle_trigger_count,
        "settle_iters_sum":      st.settle_iters_sum,
        "settle_delta_mean": if st.settle_trigger_count > 0 { Some(st.settle_delta_sum / st.settle_trigger_count as f64) } else { None },
        "mean_output_mse_wake": mean_mse_wake,
        "prior_cosine_mean": prior_cosine_mean,
        "mean_proj_loss":    stats.mean_proj_loss,
        "proj_loss_max":     stats.proj_loss_max,
        "decoder_trainable": stats.decoder_trainable_records,
        "synthetics_produced":        stats.synthetics_produced,
        "mean_real_inference_mse":    stats.mean_real_inference_mse,
        "mean_synthetic_output_mse":  stats.mean_synthetic_output_mse,
        "regions_pruned":             stats.regions_pruned.len(),
        "total_units_pruned":         stats.total_units_pruned,
        "nrem_wall_ms":               stats.nrem_wall_ms,
        "distill_precompute_ms":      stats.distill_precompute_ms,
        "rem_wall_ms":                stats.rem_wall_ms,
        "spill_wall_ms":              stats.spill_wall_ms,
        "snapshot_wall_ms":           stats.snapshot_wall_ms,
        "lm_hidden_computed":         stats.lm_hidden_computed,
        "lm_hidden_unique":           stats.lm_hidden_unique,
        "speech_encode_total_ms":     st.speech_encode_total_ms,
        "env_encode_total_ms":        st.env_encode_total_ms,
        "video_encode_total_ms":      st.video_encode_total_ms,
        "gain_update_total_ms":       0u64,
        "perception_total_ms":        st.perception_total_ms,
        "perc_kwta_us":         perf_breakdown.kwta_us,
        "perc_symbolic_us":     perf_breakdown.symbolic_us,
        "perc_bind_us":         perf_breakdown.bind_us,
        "perc_prior_us":        perf_breakdown.prior_us,
        "perc_settle_us":       perf_breakdown.settle_us,
        "perc_prior_update_us": perf_breakdown.prior_update_us,
        "warm_query_us_mean": st.hnsw_query_us_sum.checked_div(st.hnsw_query_n),
        "sal_ewma_mean":  st.sal_mu as f64,
        "sal_ewma_std":   st.sal_sigma as f64,
        "sal_threshold":  sal_gate(st.sal_mu, st.sal_sigma, st.sal_n) as f64,
        "settle_ewma_mean":    st.settle_mu as f64,
        "settle_gate_dynamic": settle_gate_val(st.settle_mu, st.settle_n, args.settle_gate) as f64,
        "wakefulness_pressure": st.wakefulness_pressure as f64,
        "prior_beta_effective": if st.s4_beta_active { 0.05f64 } else { args.prior_beta as f64 },
        "prior_betas_effective": prior_betas.map(|b| b as f64),
        "wake_ewma_mean":       st.wake_ewma_mu as f64,
        "max_wake_secs_effective": st.max_wake.as_secs(),
        "max_cold_records_effective": effective_max_cold,
        "sleep_max_secs_effective": effective_sleep_max_secs,
        "wake_interval_effective": st.effective_wake_interval,
        "s7_mse_ewma":           if st.s7_ewma_n > 0 { Some(st.s7_mse_ewma) } else { None },
        "fine_metrics":          fine_metrics,
    })
}

/// The mutable resources the per-embedding handlers need, borrowed from `run()`'s locals for the
/// duration of one call (the owning variables stay locals because `select!` also borrows the
/// channel receivers).
pub(crate) struct WakeResources<'a> {
    pub(crate) perception: &'a mut PerceptionProcessor,
    pub(crate) sink: &'a mut Option<std::io::BufWriter<std::fs::File>>,
    pub(crate) episodic: &'a mut EpisodicBuffer,
    pub(crate) recall: &'a RecollectionLayer,
    pub(crate) decoder: &'a mut Option<ActionDecoder>,
    pub(crate) gain_worker: Option<&'a GainWorker>,
    pub(crate) wake_recon_worker: Option<&'a WakeReconWorker>,
    pub(crate) args: &'a Args,
}

/// Handle the perception result for one audio or video embedding: salience/cosine/settle
/// accounting, episode recording, throttled wake reconstruction (async worker or inline), the
/// adaptive EWMAs, the one-shot startup query and the gain-worker request. `modality` only labels
/// the error log ("audio" / "video"). The audio and video `select!` arms used to carry two
/// identical copies of this body.
pub(crate) fn process_media_output(
    st: &mut WakeState,
    res: &mut WakeResources<'_>,
    result: PerceptionResult<PerceptionOutput>,
    timestamp: &somnus_sensory::Timestamp,
    modality: &str,
) {
    match result {
        Ok(out) => {
            accumulate_cosines(
                &out.prior_cosines,
                &mut st.prior_cosine_sums,
                &mut st.prior_cosine_counts,
            );
            if out.settle_iters > 0 {
                st.settle_trigger_count += 1;
                st.settle_iters_sum += out.settle_iters;
                st.settle_delta_sum += out.settle_delta as f64;
            }
            let rec = record_episode(
                &out,
                timestamp,
                res.sink,
                res.episodic,
                res.args.top_k,
                res.recall,
            );
            if let Some(ref rec) = rec {
                let sal_thresh = sal_gate(st.sal_mu, st.sal_sigma, st.sal_n);
                accumulate_salience(
                    rec,
                    sal_thresh,
                    &mut st.salience_sum,
                    &mut st.salience_n,
                    &mut st.salience_min,
                    &mut st.high_salience_episodes,
                    &mut st.blend_count,
                    &mut st.episodes_speech,
                    &mut st.episodes_env,
                    &mut st.episodes_visual,
                    &mut st.episodes_language,
                    &mut st.episodes_mixed,
                );
                update_sal_ewma(
                    rec.salience,
                    &mut st.sal_mu,
                    &mut st.sal_sigma,
                    &mut st.sal_n,
                );
                accumulate_wakefulness_pressure(
                    rec.salience,
                    sal_thresh,
                    &mut st.wakefulness_pressure,
                );
                st.hnsw_query_us_sum += rec.hnsw_query_us;
                st.hnsw_query_n += 1;
            }
            if let (Some(ep), Some(regional)) = (&out.episode, &out.regional_inputs) {
                st.wake_reconstruct_counter += 1;
                if st
                    .wake_reconstruct_counter
                    .is_multiple_of(st.effective_wake_interval)
                {
                    st.wake_reconstruct_throttle_passed += 1;
                    if rec
                        .as_ref()
                        .is_some_and(|r| r.salience > sal_gate(st.sal_mu, st.sal_sigma, st.sal_n))
                    {
                        if let Some(w) = res.wake_recon_worker {
                            if !w.request_train(
                                ep.0.clone(),
                                regional.clone(),
                                res.perception.learning_rate(),
                            ) {
                                st.wake_recon_dropped += 1;
                            }
                        } else {
                            match res.perception.wake_reconstruct_step(&ep.0, regional) {
                                Ok(mse) => {
                                    st.wake_output_mse_sum += mse as f64;
                                    st.wake_output_mse_n += 1;
                                    tracing::info!(output_mse = mse, "wake reconstruction");
                                }
                                Err(e) => {
                                    tracing::warn!(?e, "wake reconstruct error");
                                    somnus_metrics::counted_core!(
                                        somnus_metrics::Metric::WakeReconstructError,
                                        1
                                    );
                                }
                            }
                        }
                    }
                    // Refresh autodiff graph every N episodes to prevent
                    // node accumulation from forward passes slowing into_data().
                    res.perception.refresh_autodiff_state();
                }
                st.wake_episodes += 1;
                if let Some(m) = episode_cosine_mean(&out.prior_cosines) {
                    update_perf_ewma(
                        m,
                        &mut st.cosine_ewma,
                        &mut st.cosine_baseline,
                        &mut st.cosine_baseline_accum,
                        &mut st.cosine_baseline_n,
                        &mut st.perf_sleep_flag,
                        res.args.perf_sleep_threshold,
                    );
                    update_settle_ewma(m, &mut st.settle_mu, &mut st.settle_n);
                    res.perception.set_settle_gate(settle_gate_val(
                        st.settle_mu,
                        st.settle_n,
                        res.args.settle_gate,
                    ));
                }
            }
            if !st.startup_query_fired && out.episode.is_some() {
                st.startup_query_fired = maybe_startup_query(
                    &res.args.query,
                    res.decoder,
                    &*res.episodic,
                    res.args.action_max_tokens,
                );
            }
            if let Some(gw) = res.gain_worker {
                if let Some(ep) = &out.episode {
                    st.gain_episode_counter += 1;
                    if st
                        .gain_episode_counter
                        .is_multiple_of(res.args.gain_interval)
                    {
                        gw.request_gain(ep.0.clone());
                    }
                }
            }
        }
        Err(e) => {
            tracing::error!(?e, modality, "perception error");
            somnus_metrics::counted_core!(somnus_metrics::Metric::PerceptionIngestError, 1);
        }
    }
}

/// The long-lived resources a sleep cycle touches, borrowed from `run()`'s locals for one call.
pub(crate) struct SleepResources<'a> {
    pub(crate) args: &'a Args,
    pub(crate) interrupt: &'a Arc<AtomicBool>,
    pub(crate) interrupt_sleep: &'a Arc<AtomicBool>,
    pub(crate) pending_embs: &'a Arc<AtomicUsize>,
    pub(crate) asr_count: &'a Arc<AtomicUsize>,
    pub(crate) perception: &'a mut PerceptionProcessor,
    pub(crate) episodic: &'a mut EpisodicBuffer,
    pub(crate) recall: &'a mut RecollectionLayer,
    pub(crate) decoder: &'a mut Option<ActionDecoder>,
    pub(crate) gain_worker: Option<&'a GainWorker>,
    pub(crate) wake_recon_worker: Option<&'a WakeReconWorker>,
    pub(crate) sink: &'a mut Option<std::io::BufWriter<std::fs::File>>,
    pub(crate) metrics_sink: &'a mut Option<std::fs::File>,
    pub(crate) onnx_dir: Option<&'a std::path::Path>,
    pub(crate) onnx_script_path: Option<&'a std::path::Path>,
    pub(crate) effective_max_cold: usize,
    pub(crate) session_start_secs: f64,
    pub(crate) prior_beta_regions_static: Option<[f32; 5]>,
    pub(crate) wake_ewma_floor: u64,
}

/// One full sleep episode of the wake loop: quiesce wake-time workers, run NREM/REM
/// (`sleep::run_sleep_cycle`), persist the adaptive state, run the S4/S7 controllers, write the
/// cycle's metrics line, reset the per-cycle counters, restart the wake clock, sync/rebuild the
/// ORT binding session and honour `--max-sleep-cycles` (returns `Break` when the run should end).
pub(crate) fn sleep_cycle(
    st: &mut WakeState,
    res: &mut SleepResources<'_>,
    sleep_trigger: &'static str,
) -> ControlFlow<()> {
    let SleepResources {
        args,
        interrupt,
        interrupt_sleep,
        pending_embs,
        asr_count,
        perception,
        episodic,
        recall,
        decoder,
        gain_worker,
        wake_recon_worker,
        sink,
        metrics_sink,
        onnx_dir,
        onnx_script_path,
        effective_max_cold,
        session_start_secs,
        prior_beta_regions_static,
        wake_ewma_floor,
    } = res;
    let (effective_max_cold, session_start_secs, prior_beta_regions_static, wake_ewma_floor) = (
        *effective_max_cold,
        *session_start_secs,
        *prior_beta_regions_static,
        *wake_ewma_floor,
    );
    let this_wake_secs = st.wake_start.elapsed().as_secs_f32();
    info!(
        trigger = sleep_trigger,
        wakefulness_pressure = st.wakefulness_pressure,
        buffer_len = episodic.len(),
        buffer_cap = args.episodic_capacity,
        wake_secs = this_wake_secs as u64,
        "entering sleep"
    );
    interrupt.store(false, Ordering::Relaxed);
    pending_embs.store(0, Ordering::Relaxed);
    perception.reset_region_gain(); // stale gain must not carry across sleep boundary
    perception.clear_working_memory(); // working memory is wake-local
                                       // Merge worker-trained head weights into the model so NREM training
                                       // continues from them (drains the worker's queue; applies any
                                       // in-flight EMA updates so they aren't lost).
    if let Some(w) = wake_recon_worker {
        let mut pending_ema: Option<[Vec<f32>; 5]> = None;
        match w.fetch_weights(|flat, loss| {
            st.wake_output_mse_sum += loss as f64;
            st.wake_output_mse_n += 1;
            pending_ema = Some(flat);
        }) {
            Some(weights) => {
                perception.set_head_weights_flat(weights);
                if let Some(flat) = pending_ema {
                    perception.apply_ema_flat(flat);
                }
            }
            None => {
                tracing::warn!("wake-recon worker died — heads not merged");
                somnus_metrics::counted_core!(somnus_metrics::Metric::WakeReconWorkerDied, 1);
            }
        }
    }
    // P1b: auto --sleep-max-secs: cap = max(300, 3 × mean_sleep_secs) after first cycle.
    let effective_sleep_max_secs: u64 =
        compute_effective_sleep_max_secs(args.sleep_max_secs, &st.sleep_dur_secs_history);
    // Clear conversation history at sleep entry (R2 — wake-local by design).
    if let Some(ref mut dec) = decoder {
        dec.clear_conversation();
    }
    let sleep_params = sleep::SleepParams {
        sleep_loss_target: args.sleep_loss_target,
        sleep_max_secs: effective_sleep_max_secs,
        prune_fraction: args.prune_fraction,
        prune_min_grown: args.prune_min_grown,
        replay_batch_size: args.replay_batch_size,
        nrem_min_traces: args.nrem_min_traces,
        nrem_salience_threshold: args.nrem_salience_threshold,
        lang_oversample: args.lang_oversample,
        replay_epochs: args.nrem_replay_epochs,
        binding_replay: args.binding_replay,
        binding_replay_steps: args.binding_replay_steps,
        binding_replay_lr_scale: args.binding_replay_lr_scale,
        binding_replay_freeze_heads: args.binding_replay_freeze_heads,
        max_warm_records: args.max_warm_records,
        max_cold_records: effective_max_cold,
        growth_cooldown: args.growth_cooldown,
        cooldown_remaining: st.cooldown_remaining,
        session_start_secs,
        training_data_path: args.export_training_data.clone(),
    };
    let sleep_signals = sleep::SleepSignals {
        interrupt: Arc::clone(interrupt_sleep),
        pending_embs: Arc::clone(pending_embs),
    };
    match sleep::run_sleep_cycle(
        perception,
        episodic,
        recall,
        &args.memory_dir,
        decoder,
        sleep_signals,
        &sleep_params,
    ) {
        Ok(stats) => {
            st.cooldown_remaining = stats.growth_cooldown_remaining;
            if let Err(e) = save_sal_ewma(st.sal_mu, st.sal_sigma, st.sal_n, &args.memory_dir) {
                tracing::warn!(?e, "sal_ewma save failed");
                somnus_metrics::counted_core!(somnus_metrics::Metric::MiscWarning, 1);
            }
            if let Err(e) = save_settle_ewma(st.settle_mu, st.settle_n, &args.memory_dir) {
                tracing::warn!(?e, "settle_ewma save failed");
                somnus_metrics::counted_core!(somnus_metrics::Metric::MiscWarning, 1);
            }
            // S6: update wake-duration EWMA (per-cycle α=0.3; ADR 0024).
            update_wake_ewma(this_wake_secs, &mut st.wake_ewma_mu, &mut st.wake_ewma_n);
            if let Err(e) = save_wake_ewma(st.wake_ewma_mu, st.wake_ewma_n, &args.memory_dir) {
                tracing::warn!(?e, "wake_ewma save failed");
                somnus_metrics::counted_core!(somnus_metrics::Metric::MiscWarning, 1);
            }
            // Recompute max_wake from updated EWMA (floor until n≥3).
            st.max_wake = Duration::from_secs(recompute_max_wake_secs(
                st.wake_ewma_mu,
                st.wake_ewma_n,
                wake_ewma_floor,
            ));
            // Dissipate wakefulness pressure by NREM consolidation quality (ADR 0024 S3).
            // Prefer within-session gist (clean signal); fall back to cross-session at c1.
            st.wakefulness_pressure = dissipate_wakefulness_pressure(
                st.wakefulness_pressure,
                stats.within_session_gist_rate,
                stats.gist_recall_rate,
            );
            st.sleep_cycle_count += 1;
            somnus_metrics::counted_core!(
                somnus_metrics::Metric::ProcessRssBytes,
                current_process_rss_bytes()
            );
            for r in perception.duty_cycle_report() {
                info!(
                    region = r.region,
                    n_units = r.n_units,
                    k_active = r.k_active,
                    sum_dc_base = r.sum_dc_base,
                    sum_dc_grown = r.sum_dc_grown,
                    base_units_over_half = r.base_units_over_half,
                    base_units_silent = r.base_units_silent,
                    grown_units_ever_fired = r.grown_units_ever_fired,
                    mean_abs_active_base = r.mean_abs_active_base,
                    mean_abs_active_grown = r.mean_abs_active_grown,
                    grown_slot_share = r.grown_slot_share,
                    top5 = ?r.top5,
                    "kwta duty-cycle report"
                );
            }
            let (duty_dev, duty_max) = perception.duty_cycle_summary();
            somnus_metrics::counted_core!(
                somnus_metrics::Metric::KwtaDutyDevMean,
                (duty_dev * 1e6) as u64
            );
            somnus_metrics::counted_core!(somnus_metrics::Metric::KwtaDutyMax, (duty_max * 1e6) as u64);
            somnus_metrics::counted_core!(
                somnus_metrics::Metric::MemoryDirDiskFreeBytes,
                memory_dir_disk_free_bytes(&args.memory_dir)
            );
            let cycle_asr = asr_count.swap(0, Ordering::Relaxed);
            info!(
                cycle         = st.sleep_cycle_count,
                traces_replayed = stats.traces_replayed,
                mean_loss     = stats.mean_train_loss,
                mean_output_mse = stats.mean_output_mse,
                warm_recall_l2 = ?stats.warm_recall_l2,
                mean_proj_loss = ?stats.mean_proj_loss,
                cross_modal_gist_entropy = ?stats.cross_modal_gist_entropy,
                decoder_trainable = stats.decoder_trainable_records,
                interrupted   = stats.was_interrupted,
                nrem_exit     = stats.nrem_exit_reason,
                wake_episodes = st.wake_episodes,
                asr_count     = cycle_asr,
                lang_ingest   = st.lang_ingest_count,
                "sleep cycle complete"
            );
            // Save decoder projection weights alongside perception weights.
            if let Some(ref dec) = decoder {
                let dec_w = args
                    .memory_dir
                    .join("snapshot")
                    .join("decoder_weights.safetensors");
                if let Err(e) = dec.save_weights(&dec_w) {
                    tracing::warn!(?e, "decoder weights snapshot failed");
                    somnus_metrics::counted_core!(somnus_metrics::Metric::MiscWarning, 1);
                }
                // Push updated projection weights to gain worker (non-blocking).
                if let Some(gw) = gain_worker {
                    gw.update_weights(dec.proj_weights_flat());
                }
            }
            // S4: prior-beta scale-up controller (ADR 0024); disabled in static
            // per-region mode (--prior-beta-regions).
            if prior_beta_regions_static.is_none() && args.prior_beta < 0.05 {
                if let Some(beta) = st.s4_step(args.prior_beta) {
                    perception.set_prior_beta(beta);
                }
            }
            // P1b: record sleep duration for auto sleep-max-secs cap.
            let total_sleep_secs = (stats.nrem_wall_ms
                + stats.rem_wall_ms
                + stats.spill_wall_ms
                + stats.snapshot_wall_ms) as f64
                / 1000.0;
            if st.sleep_dur_secs_history.len() >= 5 {
                st.sleep_dur_secs_history.pop_front();
            }
            st.sleep_dur_secs_history.push_back(total_sleep_secs);

            // P2a / S7: adapt effective_wake_interval from the cycle-level MSE trend.
            let mean_mse_wake = if st.wake_output_mse_n > 0 {
                Some(st.wake_output_mse_sum / st.wake_output_mse_n as f64)
            } else {
                None
            };
            st.s7_step(mean_mse_wake, args.wake_reconstruct_interval);

            // Write metrics NDJSON line.
            if let Some(ref mut mf) = metrics_sink {
                let line = cycle_metrics_json(
                    st,
                    &stats,
                    args,
                    CycleContext {
                        sleep_trigger,
                        cycle_asr,
                        region_sizes: perception.region_sizes(),
                        warm_len: recall.warm.len(),
                        cold_len: recall.cold.len(),
                        prior_betas: perception.prior_betas(),
                        perf_breakdown: perception.take_perf_breakdown(),
                        fine_metrics: somnus_metrics::METRICS.snapshot_and_reset(),
                        effective_max_cold,
                        effective_sleep_max_secs,
                        mean_mse_wake,
                    },
                )
                .to_string();
                use std::io::Write as _;
                if let Err(e) = writeln!(mf, "{line}") {
                    tracing::warn!(?e, "metrics write failed");
                    somnus_metrics::counted_core!(somnus_metrics::Metric::MiscWarning, 1);
                }
            }
            st.reset_cycle();
        }
        Err(e) => {
            tracing::warn!(?e, "sleep cycle error");
            somnus_metrics::counted_core!(somnus_metrics::Metric::MiscWarning, 1);
        }
    }
    // Push NREM-trained head weights back to the worker for the next
    // wake phase (also after a failed cycle — partial training counts).
    if let Some(w) = wake_recon_worker {
        w.set_weights(perception.head_weights_flat());
    }
    st.wake_start = std::time::Instant::now();

    // ADR 0048 (gaps.md §33 candidate (e)): hard-copy the live binding
    // training copy into the wake-time snapshot only every
    // `--binding-wake-lag-cycles` cycles, not every cycle — decouples "how
    // often the binder trains" from "how often wake sees a new binder
    // generation". Must run BEFORE the ORT rebuild below so a sync cycle's
    // ORT export reflects the just-synced snapshot, not a stale one.
    //
    // NOT gated on `args.binding_replay`: `homeostatic_decay` (sleep.rs)
    // mutates `binding`'s trained tensors every cycle unconditionally, so
    // even with `--binding-replay` off, `binding` still moves and
    // `wake_binding` must still be kept in sync at the configured cadence
    // — gating this on `binding_replay` would leave `wake_binding` frozen
    // at its initial values forever whenever binding-replay is off (the
    // default/production path), which is not this ADR's intent. lag=1
    // (the default) reproduces the pre-ADR-0048 always-immediate behavior
    // exactly in both configurations.
    if st
        .sleep_cycle_count
        .is_multiple_of(args.binding_wake_lag_cycles.max(1))
    {
        perception.sync_wake_binding();
    }

    // Rebuild ORT binding session with updated weights after sleep training —
    // or, if not active yet, check whether growth just crossed the ORT
    // crossover threshold (ADR 0037) and build it for the first time.
    if let (Some(onnx_dir), Some(script)) = (&onnx_dir, &onnx_script_path) {
        let already_active = perception.ort_active();
        let now_wants_ort = already_active
            || (!st.onnx_failed
                && should_use_ort_inference(&perception.region_sizes(), args.onnx_crossover_units));
        if now_wants_ort {
            match perception.rebuild_ort_session(script, onnx_dir, args.onnx_migraphx) {
                Ok(()) => {
                    if already_active {
                        tracing::debug!("ORT binding session rebuilt");
                    } else {
                        info!(
                            region_sizes = ?perception.region_sizes(),
                            threshold = args.onnx_crossover_units,
                            "region size crossed ORT threshold mid-run — ORT binding session now active"
                        );
                    }
                }
                Err(e) if already_active => {
                    tracing::warn!(
                        ?e,
                        "ORT session rebuild failed — continuing with existing session"
                    );
                    somnus_metrics::counted_core!(somnus_metrics::Metric::OrtSessionRebuildFailed, 1);
                }
                Err(e) => {
                    tracing::warn!(?e, "ORT session init failed — running with burn inference");
                    somnus_metrics::counted_core!(somnus_metrics::Metric::OrtSessionInitFailed, 1);
                    st.onnx_failed = true;
                }
            }
        }
    }

    // Exit after N sleep cycles if requested.
    if args.max_sleep_cycles > 0 && st.sleep_cycle_count >= args.max_sleep_cycles {
        info!(
            cycles = st.sleep_cycle_count,
            "max-sleep-cycles reached — exiting"
        );
        sink_flush(sink);
        return ControlFlow::Break(());
    }
    ControlFlow::Continue(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    /// The per-cycle metrics line is a file format: `scripts/campaign_status.py` and
    /// `scripts/eval_metrics.py` read it by key. This is the key set a real run (2026-10-17 pilot)
    /// produced; a rename or accidental prefix (as an earlier mechanical refactor caused) fails here.
    const EXPECTED_KEYS: &[&str] = &[
        "asr_count",
        "blend_count",
        "cold_len",
        "cross_modal_gist_entropy",
        "cycle",
        "decoder_trainable",
        "distill_precompute_ms",
        "env_encode_total_ms",
        "episodes_env",
        "episodes_language",
        "episodes_mixed",
        "episodes_speech",
        "episodes_visual",
        "fine_metrics",
        "gain_update_total_ms",
        "gist_recall_rate",
        "grew",
        "growth_cooldown_remaining",
        "high_salience_episodes",
        "lang_ingest_count",
        "lm_hidden_computed",
        "lm_hidden_unique",
        "max_cold_records_effective",
        "max_wake_secs_effective",
        "mean_loss",
        "mean_output_mse",
        "mean_output_mse_wake",
        "mean_proj_loss",
        "mean_real_inference_mse",
        "mean_salience_wake",
        "mean_synthetic_output_mse",
        "min_salience_wake",
        "nrem_budget_traces",
        "nrem_exit_reason",
        "nrem_unique_traces",
        "nrem_wall_ms",
        "per_region_mse",
        "perc_bind_us",
        "perc_kwta_us",
        "perc_prior_update_us",
        "perc_prior_us",
        "perc_settle_us",
        "perc_symbolic_us",
        "perception_total_ms",
        "prior_beta_effective",
        "prior_betas_effective",
        "prior_cosine_mean",
        "proj_loss_max",
        "region_sizes",
        "regions_grown",
        "regions_pruned",
        "rem_wall_ms",
        "s7_mse_ewma",
        "sal_ewma_mean",
        "sal_ewma_std",
        "sal_threshold",
        "session",
        "settle_delta_mean",
        "settle_ewma_mean",
        "settle_gate_dynamic",
        "settle_iters_sum",
        "settle_trigger_count",
        "sleep_max_secs_effective",
        "sleep_trigger",
        "snapshot_wall_ms",
        "speech_encode_total_ms",
        "spill_wall_ms",
        "synthetics_produced",
        "total_units_pruned",
        "traces_replayed",
        "ts",
        "video_encode_total_ms",
        "wake_episodes",
        "wake_ewma_mean",
        "wake_interval_effective",
        "wake_recon_dropped",
        "wake_reconstruct_attempts",
        "wake_reconstruct_fired",
        "wake_reconstruct_throttle_passed",
        "wakefulness_pressure",
        "warm_len",
        "warm_query_us_mean",
        "warm_recall_l2",
        "was_interrupted",
        "within_session_gist_rate",
    ];

    fn line() -> serde_json::Value {
        let args = Args::try_parse_from(["somnus"]).unwrap();
        cycle_metrics_json(
            &WakeState::for_test(),
            &SleepStats::default(),
            &args,
            CycleContext {
                sleep_trigger: "high_water",
                cycle_asr: 0,
                region_sizes: [1024; 5],
                warm_len: 0,
                cold_len: 0,
                prior_betas: [0.01; 5],
                perf_breakdown: IngestPerfBreakdown::default(),
                fine_metrics: serde_json::json!({}),
                effective_max_cold: 0,
                effective_sleep_max_secs: 0,
                mean_mse_wake: None,
            },
        )
    }

    #[test]
    fn metrics_line_keys_are_stable() {
        let v = line();
        let mut got: Vec<&str> = v.as_object().unwrap().keys().map(|k| k.as_str()).collect();
        let mut want: Vec<&str> = EXPECTED_KEYS.to_vec();
        got.sort_unstable();
        want.sort_unstable();
        let missing: Vec<_> = want.iter().filter(|k| !got.contains(k)).collect();
        let extra: Vec<_> = got.iter().filter(|k| !want.contains(k)).collect();
        assert!(
            missing.is_empty() && extra.is_empty(),
            "missing {missing:?}, extra {extra:?}"
        );
    }

    #[test]
    fn metrics_line_reports_cycle_counters_and_nulls_when_unobserved() {
        let args = Args::try_parse_from(["somnus"]).unwrap();
        let mut st = WakeState::for_test();
        st.wake_episodes = 7;
        st.salience_n = 4;
        st.salience_sum = 2.0;
        st.salience_min = 0.1;
        st.settle_trigger_count = 2;
        st.settle_delta_sum = 1.0;
        let v = cycle_metrics_json(
            &st,
            &SleepStats::default(),
            &args,
            CycleContext {
                sleep_trigger: "schedule",
                cycle_asr: 3,
                region_sizes: [1, 2, 3, 4, 5],
                warm_len: 9,
                cold_len: 8,
                prior_betas: [0.0; 5],
                perf_breakdown: IngestPerfBreakdown::default(),
                fine_metrics: serde_json::json!({}),
                effective_max_cold: 5,
                effective_sleep_max_secs: 6,
                mean_mse_wake: Some(0.25),
            },
        );
        assert_eq!(v["wake_episodes"], 7);
        assert_eq!(v["asr_count"], 3);
        assert_eq!(v["warm_len"], 9);
        assert_eq!(v["cold_len"], 8);
        assert_eq!(v["sleep_trigger"], "schedule");
        assert_eq!(v["region_sizes"], serde_json::json!([1, 2, 3, 4, 5]));
        assert_eq!(v["mean_salience_wake"], 0.5);
        assert_eq!(v["min_salience_wake"].as_f64().unwrap() as f32, 0.1);
        assert_eq!(v["settle_delta_mean"], 0.5);
        assert_eq!(v["mean_output_mse_wake"], 0.25);
        // No salience / settle observations -> null, not 0.
        let empty = line();
        assert!(empty["mean_salience_wake"].is_null());
        assert!(empty["settle_delta_mean"].is_null());
        assert!(empty["mean_output_mse_wake"].is_null());
    }
}

//! Handlers for the events the wake loop in `run()` reacts to. Each used to be pasted inline in
//! two places (the pre-sleep drain and the `select!` arm) and the copies had drifted apart.

use std::time::{SystemTime, UNIX_EPOCH};

use somnus_memory::{EpisodicBuffer, RecollectionLayer};
use somnus_perception::{IngestPerfBreakdown, PerceptionProcessor};

use crate::{
    accumulate_cosines, accumulate_salience, accumulate_wakefulness_pressure,
    action::{ActionDecoder, GainWorker},
    build_query_with_history,
    cli::Args,
    episode_cosine_mean, is_introspective_query, query_recalled_episodes, sal_gate,
    settle_gate_val,
    sleep::SleepStats,
    update_perf_ewma, update_sal_ewma, update_settle_ewma,
    wake_state::WakeState,
    write_training_pair, LangIngestResult,
};

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

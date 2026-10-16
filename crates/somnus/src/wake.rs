//! Handlers for the events the wake loop in `run()` reacts to. Each used to be pasted inline in
//! two places (the pre-sleep drain and the `select!` arm) and the copies had drifted apart.

use std::time::{SystemTime, UNIX_EPOCH};

use somnus_memory::{EpisodicBuffer, RecollectionLayer};
use somnus_perception::PerceptionProcessor;

use crate::{
    accumulate_cosines, accumulate_salience, accumulate_wakefulness_pressure,
    action::{ActionDecoder, GainWorker},
    build_query_with_history,
    cli::Args,
    episode_cosine_mean, is_introspective_query, query_recalled_episodes, sal_gate,
    settle_gate_val, update_perf_ewma, update_sal_ewma, update_settle_ewma,
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

# scripts/

Index of helper scripts. Status: **live** = used by current workflows, **historical** = kept to
reproduce a past result (see the cited doc), **one-shot** = data/model preparation.

## Campaign / run
- `run.sh` — live. Release build then exec `somnus` (avoids cargo's stderr-pipe problem).
- `train_campaign.sh` — live. N sequential sessions accumulating memory. `SEED=<n>` passes `--seed`.
- `resume_growth_campaign.sh` — live. Resume the 10x growth campaign (gaps/engineering-log §16/§20).
- `stop_graceful.sh` — live. SIGTERM wrapper + somnus so the cycle snapshot flushes.
- `status.py`, `campaign_status.py`, `eval_metrics.py` — live. Status and metrics reporting.
- `profile.sh` — live. Short profiled run.
- `run_novel_experiment.sh`, `run_ef_scale_ab.sh`, `gen_qwen3_projweights.sh` — one-shot experiments.

## A/B pilots (`ab_*.sh`)
All export `SEED` (default 42) so both arms share a burn RNG seed.
- Current (fine-metrics on): `ab_binding_replay*.sh` (ADR 0041/0048, gaps §33), `ab_binding_wake_lag.sh` (ADR 0048 4-arm lag sweep, `SEED=`/`PILOT_ROOT=` overridable), `ab_rem_scoring.sh` (gaps §49 R1/R4/R8; control binary from `CONTROL_REV`), `ab_video_change_threshold.sh` (§41).
- Historical (pre-2026-08-15, no fine-metrics): `ab_batch_topk`, `ab_dinov3`, `ab_der_auto_lambda`,
  `ab_mlp_proj`, `ab_qwen3`, `ab_qwen3_longclips`, `ab_replay_ratio`, `ab_efficientat_mn10`.
- `analyze_ab_prior_beta.py` — analysis for the prior-beta A/B (ADR 0027).

## Model export (one-shot; need `scripts/.venv`, see `requirements.txt`)
`export_models.py`, `export_efficientat.py`, `export_perception_graph.py` (also called at runtime),
`export_kv_cache*.py`, `convert_fp16.py`, `quantize_kv.py`, `train_lora_qwen3.py` (never run, gaps §39).

## Data prep (one-shot)
`download_*.sh`, `transcribe_clips.sh`, `extract_wiki.py`, `inject_text.py` (live: feeds text to somnus stdin),
`gen_mel_fixtures.py` (test fixtures).

## Analysis / bench
`svd_rank_check.py`, `bench_binding_projection.py`.

## Git hooks
`setup-hooks.sh` installs `hooks/pre-commit` (mirrors CI: fmt + clippy default and fine-metrics).

`requirements.txt` — pinned Python deps for the export/analysis scripts (install steps in its header; `export_efficientat.py` uses a separate venv).

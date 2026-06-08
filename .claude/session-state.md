# Session state — somnus

> **Last updated:** 2026-06-08
> **Last commit on main:** 9b5e3bf — perf: three wake-path optimizations
> **Branch:** `main` (not yet pushed)

## Status: Three more wake-path perf optimizations committed (9b5e3bf).

---

## Completed this session

| Item | Commit | What changed |
|------|--------|--------------|
| Fix perception_total_ms metric | 9b5e3bf | video ingest now times only ingest_visual; was ~12ms/ep inflated |
| SIMD EMA matmuls | 9b5e3bf | EmaOutputHeads::predict → matrixmultiply::sgemm (AVX2/FMA) |
| Configurable settle rate | 9b5e3bf | --settle-gate (default 0.85) + --settle-every N (default 1) CLI args |

Previous session:
| train_campaign.sh ONNX defaults | 590f7e6 | ONNX_MODELS_DIR / ONNX_SCRIPT env vars |
| Phase 1.5: encoder ORT | 590f7e6 | 5 encoder matmuls → GPU |
| GainWorker | 321cb8e | Qwen gain ORT → dedicated OS thread |
| Instrument prior_us | de26f6c | settle_us + prior_update_us sub-timers in metrics |
| ORT binding in settle | 86ca92e | burn GRU in settle → ORT binding; hidden saved/restored |
| ORT encoder in settle | ea9ed9a | burn kwta in settle → ORT encoder + topk_from_z_no_dc |
| Skip burn kwta in ingest | a0eab2d | skip burn forward in ingest_{speech,env,visual,lang} when ORT enc active |
| Flat matmul for sym | 8419d1f | cross_modal_proj → allocation-free Vec matmul; kwta=0, sym~0.30ms |

---

## Key finding this session

**The 18ms `perc_total` was inflated**. For video frames, `t_perc` was timing
`ingest_visual` + `record_episode` + `wake_reconstruct_step` — all inside the
Ok() block. Audio and language timers correctly wrap only the ingest call.
Fixed in 9b5e3bf. Cycle 3's 4.93ms ≈ components (4.75ms) was already correct.

**True perception pipeline cost (ONNX path):**
- bind_us: ~1.6–2.6 ms/ep (ORT enc + ORT bind)
- settle_us: ~2–3.5 ms/ep (ORT enc + topk + ORT bind, fires ~0.985/ep)
- prior_update_us: ~0.35–0.50 ms/ep (5 EMA head matmuls — now sgemm)
- sym_us: ~0.30 ms/ep (flat Vec matmul)
- kwta_us: 0.000 ms/ep

The `record_episode` + `wake_reconstruct_step` overhead (the ~12ms gap in
cycles 1-2) is outside perception and can only be improved by rate-limiting
wake_reconstruct or async post-processing.

---

## Benchmark progression (warm cycles, CAPACITY=2048)

| Stage | perc_total | Δ |
|---|---|---|
| Burn-only (control) | 18.08 ms/ep | — |
| Phase 1.5 (ORT enc+bind) | 16.83 ms/ep | −1.25 |
| + ORT binding in settle | 14.59 ms/ep | −2.24 |
| + ORT encoder in settle | 14.10 ms/ep | −0.49 |
| + skip burn kwta+sym | ~5ms/ep (cycles 3) | metric was misleading |
| + sgemm + settle knobs | rebenchmark needed | new commit 9b5e3bf |

### Last confirmed benchmark (campaign_20260607_185345, 2s×3c, CAPACITY=2048)

| s | c | perc ms/ep | kwta | sym | bind | settle | prior_upd |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 18.45* | 0.000 | 0.304 | 2.646 | 3.085 | 0.436 |
| 1 | 2 | 17.67* | 0.000 | 0.326 | 1.883 | 2.088 | 0.413 |
| 1 | 3 |  4.93  | 0.000 | 0.329 | 1.762 | 2.161 | 0.495 |

*Cycles 1-2 were inflated by ~12ms due to wake_reconstruct in video timer (fixed).
True perc = bind + settle + prior_update + sym ≈ 6ms/ep for cycle 1.

---

## Performance knobs after 9b5e3bf

| Flag | Default | Effect |
|---|---|---|
| `--settle-steps 0` | N/A | Disable settle entirely; saves ~3ms/ep at 0.985/ep |
| `--settle-every 2` | 1 | Halve settle rate; saves ~1.5ms/ep |
| `--settle-every 4` | 1 | 25% settle rate; saves ~2.25ms/ep |
| `--settle-gate 0.99` | 0.85 | Reduces trigger rate as priors improve |

At `--settle-every 4`: expected total ≈ 2ms bind + 0.75ms settle + 0.35ms prior_upd + 0.30ms sym ≈ 3.4ms/ep.

---

## settle_episode execution paths (ea9ed9a)

Three paths selected at runtime:
- **Path A** (encoder+binding active): ORT encoder → topk_from_z_no_dc → ORT binding
- **Path B** (binding only): burn kwta → ORT binding
- **Path C** (burn only): burn kwta → burn binding

Hidden state saved before loop, restored after each ORT binding call.

---

## MIGraphX constraint (hard)

Never expose ScatterElements or INT64 outputs. Workaround: two-graph split, topk/masking on CPU.

---

## Key facts

- `--replay-batch-size 128` hardcoded in `train_campaign.sh`
- `--working-memory-size 8` hardcoded in `train_campaign.sh`
- `--asr-min-chars 5` for noisy-audio clips
- `--max-warm-records 200000` recommended for long runs
- Wake schedule trigger at 602s/cycle; N_UNITS=1024, K_ACTIVE=20, D_PROJ=128, EPISODE_DIM=640
- GRU_INPUT=640, GRU_HIDDEN=256, CROSS_MODAL_DIM=2176
- burn GRU: reset_after=True → ONNX linear_before_reset=1
- burn Linear weight: [n_in, n_out] (transpose of ONNX/PyTorch)
- DUTY_CYCLE_ALPHA=0.001, BOOST_STRENGTH=1.0, TARGET_DC=20/1024≈0.0195
- settle_beta default 0.5; settle_gate default 0.85; settle_every default 1

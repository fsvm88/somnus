# recall-ef A/B verdict at scale

**Date:** 2026-06-13  
**Commits:** 37fa1eb (warm_query_us_mean metric + campaign script)

## Setup

Two campaigns, each 10 sessions × 3 cycles, CAPACITY=4096, ONNX on.  
Clips: 4k Japan walk (1 hr) + London walk (1 hr).  
A = `--recall-ef 64` (default), B = `--recall-ef 16`.  
Warm index grew from 0 → ~98k records over 30 cycles.

## Results

| warm_len range | A query_us (ef=64) | B query_us (ef=16) | speedup |
|---|---|---|---|
| ~6k–10k | 593–661 | 269–298 | 2.0–2.5× |
| ~20k–50k | 479–879 | 164–443 | 1.9–3.3× |
| ~50k–98k | 502–869 | 179–425 | 2.0–2.9× |

Speedup stays in the **2–3× band** across the full index range. No sign of convergence — consistent across all 30 cycles.

`warm_recall_l2`: ef=16 is occasionally slightly worse (B higher L2), but differences are small (~0.1–0.3) and inconsistent. No clear recall degradation in `gist_recall_rate`.

## Decision

**Use `--recall-ef 16` in all campaigns** (train_campaign.sh default). The 2–3× HNSW speedup is real and repeatable; marginal recall impact not detectable in training metrics. `--recall-ef 64` remains available via CLI for precision-critical use.

## Action taken

`train_campaign.sh` hardcodes no `--recall-ef` flag — the CLI default is 64. Should add `--recall-ef 16` to the script.

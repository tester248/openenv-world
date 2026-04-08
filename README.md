---
title: OpenCachePolicy Environment Server
emoji: "⚡"
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
    - openenv
---

# OpenCachePolicy

[![Hugging Face Space](https://img.shields.io/badge/Hugging%20Face-Live%20Space-yellow?logo=huggingface&logoColor=black)](https://huggingface.co/spaces/tester248/open-cache-policy)

OpenCachePolicy is an OpenEnv environment for API cache optimization in realistic production-style conditions.
Agents must set endpoint TTLs and cache evictions to improve latency while protecting freshness and respecting memory budgets.

## The need

OpenCachePolicy models a real decision problem backend teams face daily:

1. Reduce p95 latency with smart cache usage.
2. Avoid stale responses on volatile endpoints.
3. Stay under finite cache memory.
4. Minimize policy churn that can destabilize systems.

The environment is deterministic and reproducible, making it suitable for fair model comparison.

## Environment Design

### Tasks

1. `task_easy`: Stable traffic and lenient freshness windows.
2. `task_medium`: Mixed volatility with tighter memory pressure.
3. `task_hard`: Bursty traffic with strict freshness constraints for critical services.

### Action Space

Actions are JSON with two controls:

1. `policy_updates`: endpoint TTL updates.
2. `evict_endpoints`: immediate cache evictions.

Allowed TTL buckets are discrete and deterministic:

1. `0`
2. `5`
3. `30`
4. `120`
5. `600`

### Observation Space

Each step returns both aggregate and endpoint-level signals:

1. Aggregate: weighted latency, baseline latency, stale ratio, memory used, memory budget.
2. Per endpoint: request rate, miss penalty, volatility, freshness SLA, current TTL, estimated hit rate, staleness risk, memory footprint.

### Reward Function

Per-step reward is bounded to `(0.01, 0.99)` and combines:

1. Latency gain ratio (0.50).
2. Freshness health (0.35).
3. Memory health (0.15).
4. Churn penalties for redundant or unstable policy flips.

Final task score is mean step reward.

## Quick Start

Install dependencies:

```bash
uv sync
```

Run local server:

```bash
uv run server
```

Run baseline inference:

```bash
uv run inference.py
```

Validate and deploy:

```bash
openenv validate
openenv push --repo-id tester248/open-cache-policy
```

## Required Environment Variables

1. `HF_TOKEN`
2. `API_BASE_URL`
3. `MODEL_NAME`

Optional:

1. `ENV_BASE_URL` (defaults to `http://localhost:8000`)

## Logging Contract

`inference.py` emits structured logs required by evaluation:

1. `[START]`
2. `[STEP]`
3. `[END]`

## Baseline Scores

Recent local baseline run with `Qwen/Qwen2.5-72B-Instruct`:

1. `task_easy`: `0.730`
2. `task_medium`: `0.713`
3. `task_hard`: `0.636`
4. Average: `0.693`

These scores are deterministic for fixed task fixtures and policy logic.

## Repository Layout

```text
.
├── client.py
├── inference.py
├── models.py
├── openenv.yaml
├── tasks.py
└── server
    ├── app.py
    └── open_cache_policy_environment.py
```

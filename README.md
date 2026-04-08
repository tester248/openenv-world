---
title: OpenCachePolicy Environment Server
emoji: ⚡
colorFrom: yellow
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# OpenCachePolicy

OpenCachePolicy is an OpenEnv benchmark for API cache-policy optimization.
The agent controls endpoint TTL values and evictions to improve latency while
staying within freshness SLAs and memory budget.

## Why This Environment

Caching policy is a real production control problem with direct cost and
reliability impact. The task requires balancing three competing objectives:

1. Latency reduction from higher cache hit rates.
2. Freshness safety for volatile endpoints.
3. Memory efficiency under constrained cache budget.

## Task Set

1. task_easy: Stable traffic and lenient freshness windows.
2. task_medium: Mixed volatility plus memory pressure.
3. task_hard: Bursty traffic and strict freshness on critical endpoints.

Each task has deterministic fixture windows and deterministic scoring.

## Action Space

The agent emits JSON actions:

1. policy_updates: list of endpoint TTL updates.
2. evict_endpoints: list of endpoints to evict immediately.

TTL choices are discrete buckets:
0, 5, 30, 120, 600 seconds.

## Observation Space

Each step returns:

1. Aggregate metrics: weighted latency, baseline latency, stale ratio,
   memory used, memory budget.
2. Endpoint metrics: request rate, miss penalty, volatility,
   freshness SLA, current TTL, hit-rate estimate, staleness risk,
   estimated memory footprint.

## Reward Design

Step reward is bounded to 0.01..0.99 and combines:

1. Latency gain ratio (weight 0.50).
2. Freshness health from stale ratio (weight 0.35).
3. Memory health from overflow ratio (weight 0.15).
4. Churn penalty for excessive policy flips and invalid TTL updates.

Episode score is the mean of step rewards.

## Quick Start

1. Install dependencies.

uv sync

2. Run server locally.

uv run server

3. Run baseline inference.

uv run inference.py

4. Validate and deploy.

openenv validate
openenv push --repo-id your-username/open-cache-policy

## Required Inference Environment Variables

1. API_BASE_URL
2. MODEL_NAME
3. HF_TOKEN
4. Optional: ENV_BASE_URL (defaults to http://localhost:8000)

## Output Logging Contract

The baseline runner emits strict structured logs:

1. [START] task=... env=... model=...
2. [STEP] step=... action=... reward=... done=... error=...
3. [END] success=... steps=... score=... rewards=...

## Project Layout

open_cache_policy/
  models.py
  tasks.py
  client.py
  inference.py
  openenv.yaml
  server/
    app.py
    open_cache_policy_environment.py

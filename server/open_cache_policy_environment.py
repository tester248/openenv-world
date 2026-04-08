# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Open Cache Policy environment implementation."""

import math
from typing import Any, Dict, Tuple
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

try:
    from ..models import (
        EndpointRuntime,
        OpenCachePolicyAction,
        OpenCachePolicyObservation,
        OpenCachePolicyState,
    )
    from ..tasks import TASKS, TTL_BUCKETS
except ImportError:
    from models import (
        EndpointRuntime,
        OpenCachePolicyAction,
        OpenCachePolicyObservation,
        OpenCachePolicyState,
    )
    from tasks import TASKS, TTL_BUCKETS


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class OpenCachePolicyEnvironment(
    Environment[OpenCachePolicyAction, OpenCachePolicyObservation, OpenCachePolicyState]
):
    """Environment where agents optimize endpoint-level cache policies."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._task_id = "task_easy"
        self._task_cfg: Dict[str, Any] = TASKS[self._task_id]
        self._step_idx = 0
        self._done = False
        self._ttl_policy: Dict[str, int] = {}
        self._memory_used_mb = 0.0
        self._cumulative_reward = 0.0
        self._last_window_metrics: Dict[str, Any] = {}
        self._recent_updates: Dict[str, int] = {}
        self._state = OpenCachePolicyState(episode_id=str(uuid4()), step_count=0)

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs: Any,
    ) -> OpenCachePolicyObservation:
        requested_task = kwargs.get("task_id") or kwargs.get("task") or "task_easy"
        self._task_id = requested_task if requested_task in TASKS else "task_easy"
        self._task_cfg = TASKS[self._task_id]
        self._step_idx = 0
        self._done = False
        self._ttl_policy = {name: 0 for name in self._task_cfg["endpoints"]}
        self._memory_used_mb = 0.0
        self._cumulative_reward = 0.0
        self._recent_updates = {}

        ep_id = episode_id or str(uuid4())
        self._state = OpenCachePolicyState(
            episode_id=ep_id,
            step_count=0,
            task_id=self._task_id,
            max_steps=self._task_cfg["max_steps"],
            done=False,
            memory_budget_mb=self._task_cfg["memory_budget_mb"],
            memory_used_mb=0.0,
            ttl_policy=dict(self._ttl_policy),
            cumulative_reward=0.0,
        )

        self._last_window_metrics = self._compute_window_metrics(self._step_idx)
        return self._build_observation(reward=0.0, done=False)

    def step(self, action: OpenCachePolicyAction) -> OpenCachePolicyObservation:
        if self._done:
            raise RuntimeError("Episode is finished. Call reset() before step().")

        changed_count, invalid_count, repeated_update_count = self._apply_action(action)
        metrics = self._compute_window_metrics(self._step_idx)
        self._last_window_metrics = metrics

        budget = self._task_cfg["memory_budget_mb"]
        latency_gain_ratio = (
            metrics["latency_gain_ms"] / metrics["baseline_latency_ms"]
            if metrics["baseline_latency_ms"] > 0
            else 0.0
        )
        freshness_health = 1.0 - metrics["stale_ratio"]
        overflow_ratio = max(0.0, (metrics["memory_used_mb"] - budget) / budget)
        memory_health = 1.0 - _clip(overflow_ratio, 0.0, 1.0)

        churn_penalty = min(
            0.25,
            (changed_count / max(1, len(metrics["endpoint_runtimes"]))) * 0.12
            + repeated_update_count * 0.02
            + invalid_count * 0.05,
        )

        raw_reward = (
            0.5 * _clip(latency_gain_ratio, 0.0, 1.0)
            + 0.35 * _clip(freshness_health, 0.0, 1.0)
            + 0.15 * _clip(memory_health, 0.0, 1.0)
            - churn_penalty
        )
        reward = round(_clip(raw_reward, 0.01, 0.99), 3)

        self._step_idx += 1
        self._done = self._step_idx >= self._task_cfg["max_steps"]
        self._cumulative_reward += reward
        self._memory_used_mb = metrics["memory_used_mb"]

        self._state.step_count = self._step_idx
        self._state.task_id = self._task_id
        self._state.max_steps = self._task_cfg["max_steps"]
        self._state.done = self._done
        self._state.memory_budget_mb = budget
        self._state.memory_used_mb = self._memory_used_mb
        self._state.ttl_policy = dict(self._ttl_policy)
        self._state.cumulative_reward = round(self._cumulative_reward, 4)

        metadata = {
            "latency_gain_ratio": round(_clip(latency_gain_ratio, 0.0, 1.0), 4),
            "freshness_health": round(_clip(freshness_health, 0.0, 1.0), 4),
            "memory_health": round(_clip(memory_health, 0.0, 1.0), 4),
            "overflow_ratio": round(overflow_ratio, 4),
            "churn_penalty": round(churn_penalty, 4),
            "step": self._step_idx,
        }
        return self._build_observation(reward=reward, done=self._done, metadata=metadata)

    @property
    def state(self) -> OpenCachePolicyState:
        return self._state

    def _nearest_ttl_bucket(self, ttl_seconds: int) -> Tuple[int, bool]:
        if ttl_seconds in TTL_BUCKETS:
            return ttl_seconds, False
        nearest = min(TTL_BUCKETS, key=lambda x: abs(x - ttl_seconds))
        return nearest, True

    def _apply_action(self, action: OpenCachePolicyAction) -> Tuple[int, int, int]:
        changed_count = 0
        invalid_count = 0
        repeated_update_count = 0

        for endpoint_id in action.evict_endpoints:
            if endpoint_id in self._ttl_policy and self._ttl_policy[endpoint_id] != 0:
                self._ttl_policy[endpoint_id] = 0
                changed_count += 1

        for update in action.policy_updates:
            if update.endpoint_id not in self._ttl_policy:
                continue
            snapped_ttl, invalid = self._nearest_ttl_bucket(int(update.ttl_seconds))
            if invalid:
                invalid_count += 1
            if self._ttl_policy[update.endpoint_id] != snapped_ttl:
                changed_count += 1
            else:
                repeated_update_count += 1
            self._ttl_policy[update.endpoint_id] = snapped_ttl
            self._recent_updates[update.endpoint_id] = self._step_idx

        return changed_count, invalid_count, repeated_update_count

    def _compute_window_metrics(self, step_idx: int) -> Dict[str, Any]:
        window = self._task_cfg["traffic_windows"][step_idx % len(self._task_cfg["traffic_windows"])]

        weighted_latency = 0.0
        weighted_baseline = 0.0
        weighted_stale = 0.0
        memory_total = 0.0
        total_rps = 0.0
        endpoint_runtimes = []

        for endpoint_id, cfg in self._task_cfg["endpoints"].items():
            ttl = self._ttl_policy[endpoint_id]
            rps = cfg["base_rps"] * window.get(endpoint_id, 1.0)
            volatility = cfg["volatility"]
            sla = cfg["freshness_sla_seconds"]

            if ttl == 0:
                hit_rate = 0.02
            else:
                ttl_factor = math.log1p(ttl) / math.log1p(600)
                hit_rate = _clip(0.05 + 0.8 * ttl_factor + 0.15 * (1.0 - volatility), 0.02, 0.98)

            if ttl <= sla:
                stale_ratio = 0.0
            else:
                stale_ratio = _clip(((ttl - sla) / max(1, sla)) * volatility * 0.55, 0.0, 0.95)

            baseline_latency = cfg["miss_penalty_ms"] + 12.0
            latency = baseline_latency * (1.0 - hit_rate) + 8.0
            memory_mb = cfg["payload_kb"] * (rps * ttl) / 1024.0 * 0.12

            weighted_latency += latency * rps
            weighted_baseline += baseline_latency * rps
            weighted_stale += stale_ratio * rps
            memory_total += memory_mb
            total_rps += rps

            endpoint_runtimes.append(
                EndpointRuntime(
                    endpoint_id=endpoint_id,
                    request_rate_rps=round(rps, 3),
                    miss_penalty_ms=cfg["miss_penalty_ms"],
                    data_volatility=volatility,
                    freshness_sla_seconds=sla,
                    current_ttl_seconds=ttl,
                    estimated_hit_rate=round(hit_rate, 4),
                    staleness_risk_score=round(stale_ratio, 4),
                    estimated_memory_mb=round(memory_mb, 4),
                )
            )

        weighted_latency_ms = weighted_latency / total_rps if total_rps else 0.0
        baseline_latency_ms = weighted_baseline / total_rps if total_rps else 1.0
        stale_ratio = weighted_stale / total_rps if total_rps else 0.0

        return {
            "weighted_latency_ms": weighted_latency_ms,
            "baseline_latency_ms": baseline_latency_ms,
            "latency_gain_ms": max(0.0, baseline_latency_ms - weighted_latency_ms),
            "stale_ratio": stale_ratio,
            "memory_used_mb": memory_total,
            "endpoint_runtimes": endpoint_runtimes,
        }

    def _build_observation(
        self,
        reward: float,
        done: bool,
        metadata: Dict[str, Any] | None = None,
    ) -> OpenCachePolicyObservation:
        metrics = self._last_window_metrics or self._compute_window_metrics(self._step_idx)
        return OpenCachePolicyObservation(
            task_id=self._task_id,
            step_index=self._step_idx,
            max_steps=self._task_cfg["max_steps"],
            memory_budget_mb=self._task_cfg["memory_budget_mb"],
            memory_used_mb=round(metrics["memory_used_mb"], 4),
            weighted_latency_ms=round(metrics["weighted_latency_ms"], 4),
            baseline_latency_ms=round(metrics["baseline_latency_ms"], 4),
            weighted_stale_ratio=round(metrics["stale_ratio"], 4),
            endpoints=metrics["endpoint_runtimes"],
            reward=reward,
            done=done,
            metadata=metadata or {},
        )

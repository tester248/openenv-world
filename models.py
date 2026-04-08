# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Data models for the Open Cache Policy environment."""

from typing import Dict, List

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field


class EndpointPolicyUpdate(BaseModel):
    """A policy update for a single API endpoint."""

    endpoint_id: str = Field(..., description="Endpoint identifier")
    ttl_seconds: int = Field(..., description="Desired TTL in seconds")


class EndpointRuntime(BaseModel):
    """Runtime metrics for an endpoint at the current step."""

    endpoint_id: str
    request_rate_rps: float
    miss_penalty_ms: float
    data_volatility: float
    freshness_sla_seconds: int
    current_ttl_seconds: int
    estimated_hit_rate: float
    staleness_risk_score: float
    estimated_memory_mb: float


class OpenCachePolicyAction(Action):
    """Agent action to update endpoint caching strategy."""

    policy_updates: List[EndpointPolicyUpdate] = Field(
        default_factory=list,
        description="List of endpoint TTL updates.",
    )
    evict_endpoints: List[str] = Field(
        default_factory=list,
        description="Endpoints to evict from cache immediately.",
    )


class OpenCachePolicyObservation(Observation):
    """Environment observation with aggregate and endpoint-level signals."""

    task_id: str = ""
    step_index: int = 0
    max_steps: int = 0
    memory_budget_mb: float = 0.0
    memory_used_mb: float = 0.0
    weighted_latency_ms: float = 0.0
    baseline_latency_ms: float = 0.0
    weighted_stale_ratio: float = 0.0
    endpoints: List[EndpointRuntime] = Field(default_factory=list)


class OpenCachePolicyState(State):
    """Detailed state used for debugging and grading."""

    task_id: str = ""
    max_steps: int = 0
    done: bool = False
    memory_budget_mb: float = 0.0
    memory_used_mb: float = 0.0
    ttl_policy: Dict[str, int] = Field(default_factory=dict)
    cumulative_reward: float = 0.0

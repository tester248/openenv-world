# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Open Cache Policy Environment client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
try:
    from .models import (
        EndpointRuntime,
        OpenCachePolicyAction,
        OpenCachePolicyObservation,
        OpenCachePolicyState,
    )
except ImportError:
    from models import (
        EndpointRuntime,
        OpenCachePolicyAction,
        OpenCachePolicyObservation,
        OpenCachePolicyState,
    )


class OpenCachePolicyEnv(
    EnvClient[OpenCachePolicyAction, OpenCachePolicyObservation, OpenCachePolicyState]
):
    """
    Client for the Open Cache Policy Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with OpenCachePolicyEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.echoed_message)
        ...
        ...     result = client.step(OpenCachePolicyAction(message="Hello!"))
        ...     print(result.observation.echoed_message)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = OpenCachePolicyEnv.from_docker_image("open_cache_policy-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(OpenCachePolicyAction(message="Test"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: OpenCachePolicyAction) -> Dict:
        """Convert OpenCachePolicyAction to a JSON step payload."""

        return {
            "policy_updates": [
                {
                    "endpoint_id": u.endpoint_id,
                    "ttl_seconds": u.ttl_seconds,
                }
                for u in action.policy_updates
            ],
            "evict_endpoints": action.evict_endpoints,
        }

    def _parse_result(self, payload: Dict) -> StepResult[OpenCachePolicyObservation]:
        """Parse server response into StepResult."""

        obs_data = payload.get("observation", {})
        endpoints = [EndpointRuntime(**item) for item in obs_data.get("endpoints", [])]
        observation = OpenCachePolicyObservation(
            task_id=obs_data.get("task_id", ""),
            step_index=obs_data.get("step_index", 0),
            max_steps=obs_data.get("max_steps", 0),
            memory_budget_mb=obs_data.get("memory_budget_mb", 0.0),
            memory_used_mb=obs_data.get("memory_used_mb", 0.0),
            weighted_latency_ms=obs_data.get("weighted_latency_ms", 0.0),
            baseline_latency_ms=obs_data.get("baseline_latency_ms", 0.0),
            weighted_stale_ratio=obs_data.get("weighted_stale_ratio", 0.0),
            endpoints=endpoints,
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> OpenCachePolicyState:
        """Parse server response into OpenCachePolicyState."""

        return OpenCachePolicyState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_id=payload.get("task_id", ""),
            max_steps=payload.get("max_steps", 0),
            done=payload.get("done", False),
            memory_budget_mb=payload.get("memory_budget_mb", 0.0),
            memory_used_mb=payload.get("memory_used_mb", 0.0),
            ttl_policy=payload.get("ttl_policy", {}),
            cumulative_reward=payload.get("cumulative_reward", 0.0),
        )

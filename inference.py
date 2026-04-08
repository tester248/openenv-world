#!/usr/bin/env python3
"""Baseline inference script for OpenCachePolicy."""

import json
import os
from typing import Any, Dict, List, Optional

from openai import OpenAI
from dotenv import load_dotenv

from client import OpenCachePolicyEnv
from models import EndpointPolicyUpdate, OpenCachePolicyAction

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:8000")
BENCHMARK_NAME = "OpenCachePolicy"

TTL_BUCKETS = [0, 5, 30, 120, 600]
TASKS = ["task_easy", "task_medium", "task_hard"]
SUCCESS_SCORE_THRESHOLD = 0.6

SYSTEM_PROMPT = (
    "You are optimizing API cache policy. "
    "Choose TTL values from [0,5,30,120,600]. "
    "Lower TTL for high volatility or strict freshness endpoints. "
    "Higher TTL for stable high-penalty endpoints. "
    "Return JSON only: {\"policy_updates\":[{\"endpoint_id\":\"...\",\"ttl_seconds\":30}],\"evict_endpoints\":[]}."
)


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{value:.2f}" for value in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def _nearest_bucket(ttl: int) -> int:
    return min(TTL_BUCKETS, key=lambda x: abs(x - ttl))


def _heuristic_policy(observation: Dict[str, Any]) -> Dict[str, Any]:
    updates: List[Dict[str, Any]] = []
    endpoints = observation.get("endpoints", [])
    memory_used = observation.get("memory_used_mb", 0.0)
    budget = observation.get("memory_budget_mb", 1.0)

    for endpoint in endpoints:
        volatility = float(endpoint.get("data_volatility", 0.5))
        sla = int(endpoint.get("freshness_sla_seconds", 60))
        miss_penalty = float(endpoint.get("miss_penalty_ms", 100.0))
        endpoint_id = endpoint.get("endpoint_id", "")

        if volatility >= 0.75 or sla <= 10:
            ttl = 5
        elif volatility >= 0.55 or sla <= 30:
            ttl = 30
        elif miss_penalty >= 150 and volatility <= 0.3:
            ttl = 120
        elif miss_penalty >= 100 and volatility <= 0.2:
            ttl = 120
        else:
            ttl = 30

        if memory_used > budget * 0.95 and ttl > 30:
            ttl = 30

        updates.append({"endpoint_id": endpoint_id, "ttl_seconds": _nearest_bucket(ttl)})

    return {"policy_updates": updates, "evict_endpoints": []}


def _guard_and_diff_action(
    action_payload: Dict[str, Any],
    observation: Dict[str, Any],
    task_name: str,
) -> Dict[str, Any]:
    """Apply safety constraints and only emit effective policy changes."""

    endpoint_map = {item["endpoint_id"]: item for item in observation.get("endpoints", [])}
    memory_used = float(observation.get("memory_used_mb", 0.0))
    memory_budget = float(observation.get("memory_budget_mb", 1.0))

    requested_ttls: Dict[str, int] = {}
    for update in action_payload.get("policy_updates", []):
        endpoint_id = str(update.get("endpoint_id", ""))
        if endpoint_id:
            requested_ttls[endpoint_id] = _nearest_bucket(int(update.get("ttl_seconds", 0)))

    # Use heuristic defaults when model omits an endpoint or proposes unsafe values.
    heuristic_defaults = {
        item["endpoint_id"]: int(item["ttl_seconds"])
        for item in _heuristic_policy(observation).get("policy_updates", [])
    }

    effective_updates: List[Dict[str, Any]] = []
    for endpoint_id, endpoint in endpoint_map.items():
        current_ttl = int(endpoint.get("current_ttl_seconds", 0))
        volatility = float(endpoint.get("data_volatility", 0.5))
        sla = int(endpoint.get("freshness_sla_seconds", 60))
        miss_penalty = float(endpoint.get("miss_penalty_ms", 100.0))

        target_ttl = requested_ttls.get(endpoint_id, heuristic_defaults.get(endpoint_id, current_ttl))

        if sla <= 8 or volatility >= 0.85:
            target_ttl = 5
        elif sla <= 15 or volatility >= 0.7:
            target_ttl = min(target_ttl, 30)
        elif volatility <= 0.2 and sla >= 240 and miss_penalty >= 100:
            target_ttl = max(target_ttl, 120)

        if memory_used > memory_budget * 0.95 and target_ttl > 30:
            target_ttl = 30

        # task_hard guardrail: keep high-volatility services conservative.
        if task_name == "task_hard" and (endpoint_id in {"offers", "inventory", "pricing", "realtime_stock"}):
            target_ttl = min(target_ttl, 5 if endpoint_id == "realtime_stock" else 30)

        target_ttl = _nearest_bucket(target_ttl)
        if target_ttl != current_ttl:
            effective_updates.append({"endpoint_id": endpoint_id, "ttl_seconds": target_ttl})

    evict_candidates = [str(item) for item in action_payload.get("evict_endpoints", [])]
    effective_evictions = [
        endpoint_id
        for endpoint_id in evict_candidates
        if endpoint_id in endpoint_map and int(endpoint_map[endpoint_id].get("current_ttl_seconds", 0)) != 0
    ]

    return {"policy_updates": effective_updates, "evict_endpoints": effective_evictions}


def _parse_model_action(raw_text: str, observation: Dict[str, Any]) -> Dict[str, Any]:
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "").strip()

    try:
        parsed = json.loads(text)
        updates = []
        for update in parsed.get("policy_updates", []):
            endpoint_id = str(update.get("endpoint_id", ""))
            ttl = _nearest_bucket(int(update.get("ttl_seconds", 0)))
            if endpoint_id:
                updates.append({"endpoint_id": endpoint_id, "ttl_seconds": ttl})
        evictions = [str(item) for item in parsed.get("evict_endpoints", [])]
        return {"policy_updates": updates, "evict_endpoints": evictions}
    except Exception:
        return _heuristic_policy(observation)


def _get_action(client: OpenAI, observation: Dict[str, Any], task_name: str) -> Dict[str, Any]:
    user_prompt = json.dumps(
        {
            "step_index": observation.get("step_index"),
            "memory_budget_mb": observation.get("memory_budget_mb"),
            "memory_used_mb": observation.get("memory_used_mb"),
            "weighted_latency_ms": observation.get("weighted_latency_ms"),
            "weighted_stale_ratio": observation.get("weighted_stale_ratio"),
            "endpoints": observation.get("endpoints", []),
        },
        ensure_ascii=True,
    )

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=700,
            stream=False,
        )
        content = completion.choices[0].message.content or ""
        raw_action = _parse_model_action(content, observation)
        return _guard_and_diff_action(raw_action, observation, task_name)
    except Exception:
        return _guard_and_diff_action(_heuristic_policy(observation), observation, task_name)


def run_task(llm_client: OpenAI, task_name: str) -> None:
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    last_error: Optional[str] = None

    log_start(task=task_name, env=BENCHMARK_NAME, model=MODEL_NAME)

    try:
        with OpenCachePolicyEnv(base_url=ENV_BASE_URL).sync() as env:
            result = env.reset(task_id=task_name)

            while not result.done:
                observation = result.observation.model_dump()
                action_payload = _get_action(llm_client, observation, task_name)
                action = OpenCachePolicyAction(
                    policy_updates=[EndpointPolicyUpdate(**item) for item in action_payload["policy_updates"]],
                    evict_endpoints=action_payload["evict_endpoints"],
                )

                try:
                    result = env.step(action)
                    last_error = None
                except Exception as exc:
                    last_error = str(exc)
                    break

                reward = float(result.reward or 0.0)
                rewards.append(reward)
                steps_taken += 1

                compact_action = (
                    ";".join(f"{item.endpoint_id}:{item.ttl_seconds}" for item in action.policy_updates)
                    or "none"
                )
                log_step(
                    step=steps_taken,
                    action=compact_action,
                    reward=reward,
                    done=result.done,
                    error=last_error,
                )

            score = sum(rewards) / len(rewards) if rewards else 0.0
            score = max(0.01, min(0.99, score))
            success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        last_error = str(exc)

    if last_error and not rewards:
        log_step(step=1, action="none", reward=0.0, done=True, error=last_error)

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def main() -> None:
    if not API_KEY:
        raise RuntimeError("Missing HF_TOKEN or API_KEY for inference")

    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    for task in TASKS:
        run_task(llm_client, task)


if __name__ == "__main__":
    main()

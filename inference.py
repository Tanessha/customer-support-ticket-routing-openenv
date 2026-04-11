from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from openai import OpenAI
from pydantic import ValidationError

from environment import CustomerSupportTicketRoutingEnvironment
from models import Action
from tasks import TASKS
from tools.policy_engine import TicketPolicyEngine


LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
TASK_NAME = os.getenv("MY_ENV_V4_TASK", "all")
BENCHMARK = os.getenv("MY_ENV_V4_BENCHMARK", "customer-support-ticket-routing")
MAX_STEPS = int(os.getenv("MAX_STEPS", "8"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "outputs"))

SYSTEM_PROMPT = """You are an agent routing support tickets.
Return JSON only with:
- ticket_id (int)
- category (billing|technical|general|spam)
- priority (low|medium|high)
- response (string or null)
Select exactly one unprocessed ticket for each step.
"""


@dataclass
class EpisodeMemory:
    processed_ticket_ids: list[int] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    rationale: list[str] = field(default_factory=list)


def _to_bool_str(value: bool) -> str:
    return "true" if value else "false"


def _line_safe_error(error: Optional[str]) -> str:
    if not error:
        return "null"
    return " ".join(str(error).split())


def _build_prompt(task: dict[str, Any], observation: dict[str, Any]) -> str:
    unresolved = [ticket for ticket in observation["tickets"] if ticket["id"] not in set(observation["processed_ticket_ids"])]
    return json.dumps(
        {
            "goal": task["goal"],
            "difficulty": task["difficulty"],
            "require_response": task["require_response"],
            "tickets": unresolved,
            "processed_ticket_ids": observation["processed_ticket_ids"],
            "instruction": (
                "Process one unprocessed ticket. Prioritize urgent SLA/high-priority items first. "
                "For hard tasks provide concise, safe, customer-facing response."
            ),
        },
        separators=(",", ":"),
    )


def _llm_action(
    client: Optional[OpenAI],
    model_name: str,
    task: dict[str, Any],
    observation: dict[str, Any],
) -> Action:
    if client is None:
        raise RuntimeError("MissingHFToken")
    response = client.chat.completions.create(
        model=model_name,
        temperature=TEMPERATURE,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_prompt(task, observation)},
        ],
    )
    content = response.choices[0].message.content or "{}"
    payload = json.loads(content)
    return Action.model_validate(payload)


def _select_action(
    client: Optional[OpenAI],
    model_name: str,
    task: dict[str, Any],
    observation: dict[str, Any],
    memory: EpisodeMemory,
    policy: TicketPolicyEngine,
) -> tuple[Action, Optional[str]]:
    llm_error: Optional[str] = None
    llm_candidate: Optional[Action] = None
    try:
        llm_candidate = _llm_action(client, model_name, task, observation)
    except (ValidationError, json.JSONDecodeError, Exception) as exc:
        llm_error = f"llm_error:{type(exc).__name__}"
        memory.errors.append(llm_error)

    action = policy.repair_action(task, observation, llm_candidate)
    inferred = policy.infer_ticket(task, next(ticket for ticket in observation["tickets"] if ticket["id"] == action.ticket_id))
    memory.rationale.append(f"ticket={action.ticket_id}:{inferred.rationale}")
    return action, llm_error


def _action_str(action: Action) -> str:
    return json.dumps(action.model_dump(mode="json"), separators=(",", ":"))


def _persist_episode_report(
    task_id: str,
    model_name: str,
    rewards: list[float],
    success: bool,
    final_score: float,
) -> None:
    """Persist lightweight run artifacts for debugging and judge-friendly transparency."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "task_id": task_id,
        "model": model_name,
        "success": success,
        "steps": len(rewards),
        "score": round(final_score, 4),
        "rewards": [round(reward, 4) for reward in rewards],
    }
    jsonl_path = OUTPUT_DIR / "episode_reports.jsonl"
    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def _run_episode(client: Optional[OpenAI], model_name: str, task_id: str) -> None:
    env = CustomerSupportTicketRoutingEnvironment()
    observation = env.reset(task_id)
    task = TASKS[task_id]
    memory = EpisodeMemory()
    policy = TicketPolicyEngine()

    print(f"[START] task={task_id} env={BENCHMARK} model={model_name}")
    rewards: list[float] = []

    for step_n in range(1, MAX_STEPS + 1):
        action, llm_error = _select_action(
            client=client,
            model_name=model_name,
            task=task,
            observation=observation.model_dump(mode="json"),
            memory=memory,
            policy=policy,
        )

        result = env.step(action)
        rewards.append(result.reward)
        observation = result.observation
        memory.processed_ticket_ids = list(observation.processed_ticket_ids)

        step_error = llm_error
        if isinstance(result.info, dict) and result.info.get("message"):
            msg = str(result.info["message"])
            step_error = f"{step_error};{msg}" if step_error else msg

        print(
            f"[STEP]  step={step_n} action={_action_str(action)} "
            f"reward={result.reward:.2f} done={_to_bool_str(result.done)} error={_line_safe_error(step_error)}"
        )

        if result.done:
            break

    final_breakdown = env.grader_breakdown()
    final_state = env.state()
    final_score = float(final_breakdown["score"])
    rewards_csv = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END]   success={_to_bool_str(final_state.done)} steps={len(rewards)} "
        f"score={final_score:.2f} rewards={rewards_csv}"
    )
    _persist_episode_report(
        task_id=task_id,
        model_name=model_name,
        rewards=rewards,
        success=final_state.done,
        final_score=final_score,
    )


def _task_ids() -> list[str]:
    if TASK_NAME.lower() in {"all", "*"}:
        return list(TASKS.keys())
    if TASK_NAME in TASKS:
        return [TASK_NAME]
    return [next(iter(TASKS))]


def main() -> None:
    client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL) if HF_TOKEN else None
    for task_id in _task_ids():
        _run_episode(client, MODEL_NAME, task_id)


if __name__ == "__main__":
    main()



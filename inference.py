from __future__ import annotations

import json
import os
from typing import Any, Optional

from openai import OpenAI
from pydantic import ValidationError

from environment import CustomerSupportTicketRoutingEnvironment
from graders import grade_episode
from models import Action
from tasks import TASKS


LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") or os.getenv("IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "hf_missing_token"
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
TASK_NAME = os.getenv("MY_ENV_V4_TASK", "all")
BENCHMARK = os.getenv("MY_ENV_V4_BENCHMARK", "customer-support-ticket-routing")
MAX_STEPS = int(os.getenv("MAX_STEPS", "8"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))

SYSTEM_PROMPT = """You are an agent routing support tickets.
Return JSON only with:
- ticket_id (int)
- category (billing|technical|general|spam)
- priority (low|medium|high)
- response (string or null)
Select exactly one unprocessed ticket for each step.
"""


def _to_bool_str(value: bool) -> str:
    return "true" if value else "false"


def _line_safe_error(error: Optional[str]) -> str:
    if not error:
        return "null"
    return " ".join(str(error).split())


def _build_prompt(task: dict[str, Any], observation: dict[str, Any]) -> str:
    return json.dumps(
        {
            "goal": task["goal"],
            "difficulty": task["difficulty"],
            "require_response": task["require_response"],
            "tickets": observation["tickets"],
            "processed_ticket_ids": observation["processed_ticket_ids"],
            "instruction": "Pick the next best ticket to process.",
        },
        separators=(",", ":"),
    )


def _heuristic_action(task: dict[str, Any], observation: dict[str, Any]) -> Action:
    processed = set(observation["processed_ticket_ids"])
    unprocessed = [t for t in observation["tickets"] if t["id"] not in processed]
    ticket = unprocessed[0]
    text = ticket["text"].lower()

    if any(term in text for term in ["charged", "payment", "invoice", "billed", "refund", "billing"]):
        category = "billing"
    elif any(term in text for term in ["crash", "log in", "login", "dashboard", "release", "export"]):
        category = "technical"
    elif any(term in text for term in ["iphone", "click here", "won", "free money"]):
        category = "spam"
    else:
        category = "general"

    if category == "spam":
        priority = "low"
    elif ticket["customer_tier"] == "premium" and ticket["sentiment"] == "angry":
        priority = "high"
    elif category in {"billing", "technical"} and ticket["sla_hours"] <= 8:
        priority = "high"
    elif category == "general" and bool(task["score_priority"]):
        priority = "medium"
    else:
        priority = "medium"

    response: Optional[str] = None
    if bool(task["require_response"]) and category != "spam":
        if category == "billing":
            response = "Thanks for reporting this billing issue. We will review the charge and refund path."
        elif category == "technical":
            response = "We are urgently investigating the login problem and will share updates."
        else:
            response = "Happy to help with workspace admin setup and next steps."

    return Action(ticket_id=ticket["id"], category=category, priority=priority, response=response)


def _llm_action(
    client: OpenAI,
    model_name: str,
    task: dict[str, Any],
    observation: dict[str, Any],
) -> Action:
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


def _action_str(action: Action) -> str:
    return json.dumps(action.model_dump(mode="json"), separators=(",", ":"))


def _run_episode(client: OpenAI, model_name: str, task_id: str) -> None:
    env = CustomerSupportTicketRoutingEnvironment()
    observation = env.reset(task_id)
    task = TASKS[task_id]

    print(f"[START] task={task_id} env={BENCHMARK} model={model_name}")
    rewards: list[float] = []

    for step_n in range(1, MAX_STEPS + 1):
        llm_error: Optional[str] = None
        try:
            action = _llm_action(client, model_name, task, observation.model_dump(mode="json"))
        except (ValidationError, json.JSONDecodeError, Exception) as exc:
            llm_error = f"llm_error:{type(exc).__name__}"
            action = _heuristic_action(task, observation.model_dump(mode="json"))

        result = env.step(action)
        rewards.append(result.reward)
        observation = result.observation

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

    final_state = env.state()
    final_score = grade_episode(task, final_state.predictions)
    final_score = max(0.0, min(1.0, final_score))
    rewards_csv = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END]   success={_to_bool_str(final_state.done)} steps={len(rewards)} "
        f"score={final_score:.2f} rewards={rewards_csv}"
    )


def _task_ids() -> list[str]:
    if TASK_NAME.lower() in {"all", "*"}:
        return list(TASKS.keys())
    if TASK_NAME in TASKS:
        return [TASK_NAME]
    return [next(iter(TASKS))]


def main() -> None:
    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    for task_id in _task_ids():
        _run_episode(client, MODEL_NAME, task_id)


if __name__ == "__main__":
    main()



from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

from openai import OpenAI

from app.env import SupportOpsEnv
from app.models import Action
from app.tasks import list_task_descriptors


SYSTEM_PROMPT = """You are operating a customer-support environment.
Choose one action at a time and return valid JSON with keys:
action_type, query, priority, tag, message, note.
Only include keys that matter for the chosen action.
Reason carefully from the observation and objective.
"""


def build_user_prompt(task: Dict, observation: Dict) -> str:
    return json.dumps(
        {
            "task": task,
            "observation": observation,
            "instructions": [
                "Use the available tools before making workflow decisions.",
                "Provide concise, policy-compliant replies.",
                "End the episode with resolve_ticket, escalate_ticket, or request_more_info when appropriate.",
            ],
        },
        indent=2,
    )


def run_episode(env: SupportOpsEnv, client: OpenAI, model: str, task_id: str, max_agent_steps: int = 8) -> Dict:
    observation = env.reset(task_id)
    task = next(item for item in list_task_descriptors() if item.task_id == task_id).model_dump()

    for _ in range(max_agent_steps):
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(task, observation.model_dump())},
            ],
        )
        payload = json.loads(response.choices[0].message.content)
        action = Action.model_validate(payload)
        result = env.step(action)
        observation = result.observation
        if result.done:
            break

    return env.grader()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a reproducible baseline against all tasks.")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON only.")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required.")

    client = OpenAI()
    env = SupportOpsEnv()

    results: List[Dict] = []
    for descriptor in list_task_descriptors():
        results.append(run_episode(env, client, args.model, descriptor.task_id))

    summary = {
        "model": args.model,
        "tasks": results,
        "average_score": round(sum(item["score"] for item in results) / len(results), 4),
    }

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(f"Model: {summary['model']}")
        for item in summary["tasks"]:
            print(f"- {item['task_id']}: {item['score']}")
        print(f"Average: {summary['average_score']}")


if __name__ == "__main__":
    main()
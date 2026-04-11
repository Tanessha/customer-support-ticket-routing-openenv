from __future__ import annotations

from copy import deepcopy
from typing import Dict, Optional

from graders import grade_episode_breakdown, grade_ticket
from models import Action, Observation, State, StepResult, TicketView
from tasks import TASKS


class CustomerSupportTicketRoutingEnvironment:
    """OpenEnv-compatible support routing environment with step-based SLA simulation."""

    def __init__(self, default_task_id: str = "task_easy_classification") -> None:
        self.default_task_id = default_task_id
        self._task: Dict[str, object] = {}
        self._predictions: Dict[int, Dict[str, Optional[str]]] = {}
        self._processed_ticket_ids: list[int] = []
        self._cumulative_reward = 0.0
        self._latest_grader_breakdown: Dict[int, Dict[str, float]] = {}
        self._current_step = 0
        self._done = False
        self.reset(default_task_id)

    def reset(self, task_id: Optional[str] = None) -> Observation:
        """Reset the environment to the selected task and return the initial observation."""
        selected_task_id = task_id or self.default_task_id
        if selected_task_id not in TASKS:
            raise ValueError(f"Unknown task_id: {selected_task_id}")

        self._task = deepcopy(TASKS[selected_task_id])
        self._predictions = {}
        self._processed_ticket_ids = []
        self._cumulative_reward = 0.0
        self._latest_grader_breakdown = {}
        self._current_step = 0
        self._done = False
        return self._build_observation()

    def step(self, action: Action) -> StepResult:
        """Process one routing action and return dense reward plus updated observation."""
        if self._done:
            return StepResult(
                observation=self._build_observation(),
                reward=0.0,
                done=True,
                info={
                    "message": "Episode already complete.",
                    "score": grade_episode_breakdown(self._task, self._predictions, self._processed_ticket_ids)["score"],
                },
            )

        self._current_step += 1
        valid_ticket_ids = {ticket["id"] for ticket in self._task["tickets"]}
        if action.ticket_id not in valid_ticket_ids:
            return StepResult(
                observation=self._build_observation(),
                reward=-0.3,
                done=False,
                info={"message": f"Invalid ticket_id: {action.ticket_id}"},
            )

        if action.ticket_id in self._processed_ticket_ids:
            return StepResult(
                observation=self._build_observation(),
                reward=-0.3,
                done=False,
                info={"message": f"Ticket {action.ticket_id} already processed."},
            )

        ticket = next(ticket for ticket in self._task["tickets"] if ticket["id"] == action.ticket_id)
        self._predictions[action.ticket_id] = {
            "category": action.category.value,
            "priority": action.priority.value,
            "response": action.response,
        }
        self._processed_ticket_ids.append(action.ticket_id)

        ticket_grade = grade_ticket(
            predicted=self._predictions[action.ticket_id],
            truth=self._task["ground_truth"][action.ticket_id],
            score_priority=bool(self._task["score_priority"]),
            require_response=bool(self._task["require_response"]),
        )
        step_reward, reward_details = self._compute_step_reward(ticket, ticket_grade)
        self._cumulative_reward += step_reward
        self._done = len(self._processed_ticket_ids) == len(self._task["tickets"])
        breakdown = grade_episode_breakdown(self._task, self._predictions, self._processed_ticket_ids)
        self._latest_grader_breakdown = breakdown["tickets"]

        info = {
            "current_step": self._current_step,
            "ticket_score": ticket_grade["score"],
            "category_accuracy": ticket_grade["category_accuracy"],
            "priority_accuracy": ticket_grade["priority_accuracy"],
            "response_quality": ticket_grade["response_quality"],
            "category_correct": ticket_grade["category_correct"],
            "priority_correct": ticket_grade["priority_correct"],
            "response_content_score": ticket_grade["response_content_score"],
            "response_format_score": ticket_grade["response_format_score"],
            "response_safety_penalty": ticket_grade["response_safety_penalty"],
            "response_score": ticket_grade["response_score"],
            "fast_resolution_bonus": reward_details["fast_resolution_bonus"],
            "urgent_ticket_penalty": reward_details["urgent_ticket_penalty"],
            "sla_delay_penalty": reward_details["sla_delay_penalty"],
            "coverage": breakdown["coverage"],
            "episode_score": breakdown["score"],
            "base_episode_score": breakdown["base_score"],
            "completion_bonus": breakdown["completion_bonus"],
        }
        return StepResult(
            observation=self._build_observation(),
            reward=step_reward,
            done=self._done,
            info=info,
        )

    def state(self) -> State:
        """Return current state for validation and external inspection."""
        return State(
            task_id=self._task["task_id"],
            goal=self._task["goal"],
            difficulty=self._task["difficulty"],
            current_step=self._current_step,
            total_tickets=len(self._task["tickets"]),
            processed_ticket_ids=self._processed_ticket_ids,
            predictions=self._predictions,
            cumulative_reward=round(self._cumulative_reward, 4),
            latest_grader_breakdown=self._latest_grader_breakdown,
            done=self._done,
        )

    def grader(self) -> float:
        return grade_episode_breakdown(self._task, self._predictions, self._processed_ticket_ids)["score"]

    def grader_breakdown(self) -> Dict[str, object]:
        return grade_episode_breakdown(self._task, self._predictions, self._processed_ticket_ids)

    def _compute_step_reward(self, ticket: Dict[str, object], ticket_grade: Dict[str, float]) -> tuple[float, Dict[str, float]]:
        """Compute dense per-step reward with correctness, urgency, and SLA timing signals."""
        reward = 0.0

        if ticket_grade["category_correct"] == 1.0:
            reward += 0.4
        else:
            reward -= 0.3

        if bool(self._task["score_priority"]) and ticket_grade["priority_correct"] == 1.0:
            reward += 0.3

        if bool(self._task["require_response"]):
            reward += 0.1 * ticket_grade["response_quality"]

        fast_resolution_bonus = self._fast_resolution_bonus(ticket, ticket_grade["priority_correct"])
        urgent_ticket_penalty = self._urgent_ticket_penalty(ticket_grade["priority_correct"], ticket)
        sla_delay_penalty = self._sla_delay_penalty(ticket)

        reward += fast_resolution_bonus
        reward -= urgent_ticket_penalty
        reward -= sla_delay_penalty

        return round(reward, 4), {
            "fast_resolution_bonus": round(fast_resolution_bonus, 4),
            "urgent_ticket_penalty": round(urgent_ticket_penalty, 4),
            "sla_delay_penalty": round(sla_delay_penalty, 4),
        }

    def _fast_resolution_bonus(self, ticket: Dict[str, object], priority_correct: float) -> float:
        if self._current_step <= ticket["sla_hours"] and priority_correct == 1.0:
            return 0.2
        return 0.0

    def _urgent_ticket_penalty(self, priority_correct: float, ticket: Dict[str, object]) -> float:
        truth = self._task["ground_truth"][ticket["id"]]
        if truth["priority"] == "high" and priority_correct == 0.0:
            return 0.5
        return 0.0

    def _sla_delay_penalty(self, ticket: Dict[str, object]) -> float:
        overdue_steps = max(0, self._current_step - ticket["created_at_step"] - ticket["sla_hours"])
        return min(1.0, 0.05 * overdue_steps)

    def _build_observation(self) -> Observation:
        ticket_views = []
        for ticket in self._task["tickets"]:
            prediction = self._predictions.get(ticket["id"], {})
            ticket_views.append(
                TicketView(
                    id=ticket["id"],
                    text=ticket["text"],
                    channel=ticket["channel"],
                    customer_tier=ticket["customer_tier"],
                    created_at_step=ticket["created_at_step"],
                    sla_hours=ticket["sla_hours"],
                    sentiment=ticket["sentiment"],
                    response_required=ticket["response_required"],
                    predicted_category=prediction.get("category"),
                    predicted_priority=prediction.get("priority"),
                    predicted_response=prediction.get("response"),
                )
            )

        return Observation(
            task_id=self._task["task_id"],
            goal=self._task["goal"],
            difficulty=self._task["difficulty"],
            current_step=self._current_step,
            tickets=ticket_views,
            processed_ticket_ids=self._processed_ticket_ids,
            remaining_tickets=len(self._task["tickets"]) - len(self._processed_ticket_ids),
        )

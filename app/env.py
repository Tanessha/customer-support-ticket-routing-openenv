from __future__ import annotations

from copy import deepcopy
from difflib import SequenceMatcher
from typing import Dict, List, Optional

from .graders import grade_episode
from .models import Action, ActionType, EnvironmentState, Observation, Priority, Reward, StepResult, Ticket, TicketStatus
from .tasks import KB, TASKS


class SupportOpsEnv:
    def __init__(self, default_task_id: str = "easy_refund_triage", max_steps: int = 12) -> None:
        self.default_task_id = default_task_id
        self.max_steps = max_steps
        self._task_id = default_task_id
        self._task_data: Dict = {}
        self._ticket: Optional[Ticket] = None
        self._ticket_visible = False
        self._kb_results = []
        self._applied_tags: List[str] = []
        self._priority: Optional[Priority] = None
        self._draft_reply: Optional[str] = None
        self._action_history: List[str] = []
        self._notes: List[str] = []
        self._used_articles: List[str] = []
        self._status = TicketStatus.open
        self._step_count = 0
        self._done = False
        self._score = 0.0
        self.reset(default_task_id)

    def reset(self, task_id: Optional[str] = None) -> Observation:
        self._task_id = task_id or self.default_task_id
        self._task_data = deepcopy(TASKS[self._task_id])
        self._ticket = Ticket(**self._task_data["ticket"])
        self._ticket_visible = False
        self._kb_results = []
        self._applied_tags = []
        self._priority = None
        self._draft_reply = None
        self._action_history = []
        self._notes = []
        self._used_articles = []
        self._status = TicketStatus.open
        self._step_count = 0
        self._done = False
        self._score = 0.0
        return self._build_observation(last_action="reset")

    def state(self) -> EnvironmentState:
        descriptor = self._task_data["descriptor"]
        graded = grade_episode(self._build_observation(), self._task_data["target"], self._used_articles)
        self._score = graded["score"]
        return EnvironmentState(
            task_id=self._task_id,
            task_title=descriptor.title,
            difficulty=descriptor.difficulty,
            step_count=self._step_count,
            max_steps=self.max_steps,
            done=self._done,
            observation=self._build_observation(),
            score=self._score,
            target_snapshot=self._safe_target_snapshot(),
        )

    def step(self, action: Action) -> StepResult:
        if self._done:
            observation = self._build_observation(last_action=f"{action.action_type.value} (ignored: episode finished)")
            reward = Reward(value=-0.1, reason="Episode already finished.", progress={"terminal": self._score})
            return StepResult(observation=observation, reward=reward, done=True, info={"score": self._score})

        self._step_count += 1
        action_summary = self._apply_action(action)
        observation = self._build_observation(last_action=action_summary)
        reward = self._compute_reward(action)

        if self._step_count >= self.max_steps and not self._done:
            self._done = True
            reward.value = max(-1.0, reward.value - 0.2)
            reward.reason += " Reached max steps."

        graded = grade_episode(observation, self._task_data["target"], self._used_articles)
        self._score = graded["score"]
        return StepResult(
            observation=observation,
            reward=reward,
            done=self._done,
            info={"score": self._score, "grader": graded},
        )

    def grader(self) -> Dict:
        graded = grade_episode(self._build_observation(), self._task_data["target"], self._used_articles)
        self._score = graded["score"]
        return {
            "task_id": self._task_id,
            "score": self._score,
            "done": self._done,
            "grader": graded,
        }

    def _apply_action(self, action: Action) -> str:
        if action.action_type == ActionType.read_ticket:
            self._ticket_visible = True
            summary = "read_ticket"
        elif action.action_type == ActionType.search_kb:
            query = (action.query or "").strip().lower()
            ranked = sorted(KB, key=lambda article: self._score_article(article, query), reverse=True)
            self._kb_results = [article for article in ranked if self._score_article(article, query) > 0][:3]
            if self._kb_results:
                self._used_articles.extend(article.article_id for article in self._kb_results[:1])
            summary = f"search_kb:{query}"
        elif action.action_type == ActionType.set_priority and action.priority:
            self._priority = action.priority
            summary = f"set_priority:{action.priority.value}"
        elif action.action_type == ActionType.add_tag and action.tag:
            normalized = action.tag.strip().lower()
            if normalized and normalized not in self._applied_tags:
                self._applied_tags.append(normalized)
            summary = f"add_tag:{normalized}"
        elif action.action_type == ActionType.draft_reply and action.message:
            self._draft_reply = action.message.strip()
            summary = "draft_reply"
        elif action.action_type == ActionType.resolve_ticket:
            self._status = TicketStatus.resolved
            self._done = True
            summary = "resolve_ticket"
        elif action.action_type == ActionType.escalate_ticket:
            self._status = TicketStatus.escalated
            self._done = True
            summary = "escalate_ticket"
        elif action.action_type == ActionType.request_more_info:
            self._status = TicketStatus.pending
            if action.note:
                self._notes.append(action.note.strip())
            self._done = True
            summary = "request_more_info"
        else:
            summary = f"invalid_or_incomplete:{action.action_type.value}"

        self._action_history.append(summary)
        return summary

    def _compute_reward(self, action: Action) -> Reward:
        target = self._task_data["target"]
        progress = grade_episode(self._build_observation(), target, self._used_articles)["components"]
        value = 0.0
        reason_parts: List[str] = []

        if action.action_type == ActionType.read_ticket:
            value += 0.05
            reason_parts.append("Ticket inspected.")

        if action.action_type == ActionType.search_kb:
            top_hit = self._used_articles[-1] if self._used_articles else None
            if top_hit in target.get("required_articles", []):
                value += 0.15
                reason_parts.append("Relevant knowledge article found.")
            else:
                value -= 0.03
                reason_parts.append("Search did not improve progress much.")

        if action.action_type == ActionType.add_tag:
            if action.tag and action.tag.strip().lower() in target.get("required_tags", []):
                value += 0.10
                reason_parts.append("Useful tag applied.")
            else:
                value -= 0.02
                reason_parts.append("Tag was not aligned with the task.")

        if action.action_type == ActionType.set_priority:
            if action.priority and action.priority.value == target.get("priority"):
                value += 0.10
                reason_parts.append("Priority matches the situation.")
            else:
                value -= 0.05
                reason_parts.append("Priority choice was suboptimal.")

        if action.action_type == ActionType.draft_reply:
            reply = action.message or ""
            must_hits = sum(1 for snippet in target.get("reply_must_include", []) if snippet.lower() in reply.lower())
            denom = max(1, len(target.get("reply_must_include", [])))
            value += 0.20 * (must_hits / denom)
            if target.get("reply_must_exclude") and any(term.lower() in reply.lower() for term in target["reply_must_exclude"]):
                value -= 0.15
                reason_parts.append("Reply included a discouraged promise.")
            else:
                reason_parts.append("Reply moved the task forward.")

        if action.action_type in {ActionType.resolve_ticket, ActionType.escalate_ticket, ActionType.request_more_info}:
            expected = target.get("final_status")
            if self._status.value == expected:
                value += 0.20
                reason_parts.append("Episode ended with the correct workflow decision.")
            else:
                value -= 0.20
                reason_parts.append("Episode ended with the wrong workflow decision.")

        repeated_penalty = self._repetition_penalty()
        if repeated_penalty:
            value -= repeated_penalty
            reason_parts.append("Repeated actions reduced reward.")

        if not reason_parts:
            reason_parts.append("Action processed.")

        value = max(-1.0, min(1.0, round(value, 4)))
        return Reward(value=value, reason=" ".join(reason_parts), progress=progress)

    def _repetition_penalty(self) -> float:
        if len(self._action_history) < 3:
            return 0.0
        if self._action_history[-1] == self._action_history[-2] == self._action_history[-3]:
            return 0.10
        return 0.0

    def _build_observation(self, last_action: Optional[str] = None) -> Observation:
        return Observation(
            task_id=self._task_id,
            step_count=self._step_count,
            max_steps=self.max_steps,
            status=self._status,
            ticket_visible=self._ticket_visible,
            ticket=self._ticket if self._ticket_visible else None,
            kb_results=self._kb_results,
            applied_tags=self._applied_tags,
            priority=self._priority,
            draft_reply=self._draft_reply,
            last_action=last_action,
            action_history=self._action_history[-8:],
            notes=self._notes,
        )

    def _score_article(self, article, query: str) -> float:
        if not query:
            return 0.0
        haystacks = [article.title.lower(), article.body.lower(), " ".join(article.keywords).lower()]
        overlap = max(SequenceMatcher(None, query, hay).ratio() for hay in haystacks)
        keyword_bonus = sum(0.25 for keyword in article.keywords if keyword in query)
        return overlap + keyword_bonus

    def _safe_target_snapshot(self) -> Dict:
        descriptor = self._task_data["descriptor"]
        return {
            "difficulty": descriptor.difficulty,
            "success_criteria": descriptor.success_criteria,
        }
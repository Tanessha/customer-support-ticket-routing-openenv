from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class Priority(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    urgent = "urgent"


class TicketStatus(str, Enum):
    open = "open"
    pending = "pending"
    resolved = "resolved"
    escalated = "escalated"


class ActionType(str, Enum):
    read_ticket = "read_ticket"
    search_kb = "search_kb"
    set_priority = "set_priority"
    add_tag = "add_tag"
    draft_reply = "draft_reply"
    resolve_ticket = "resolve_ticket"
    escalate_ticket = "escalate_ticket"
    request_more_info = "request_more_info"


class KnowledgeArticle(BaseModel):
    article_id: str
    title: str
    body: str
    keywords: List[str] = Field(default_factory=list)


class Ticket(BaseModel):
    ticket_id: str
    customer_name: str
    subject: str
    body: str
    channel: str
    account_tier: str
    created_at: str


class Observation(BaseModel):
    task_id: str
    step_count: int
    max_steps: int
    status: TicketStatus
    ticket_visible: bool
    ticket: Optional[Ticket] = None
    kb_results: List[KnowledgeArticle] = Field(default_factory=list)
    applied_tags: List[str] = Field(default_factory=list)
    priority: Optional[Priority] = None
    draft_reply: Optional[str] = None
    last_action: Optional[str] = None
    action_history: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class Reward(BaseModel):
    value: float = Field(ge=-1.0, le=1.0)
    reason: str
    progress: Dict[str, float] = Field(default_factory=dict)


class Action(BaseModel):
    action_type: ActionType
    query: Optional[str] = None
    priority: Optional[Priority] = None
    tag: Optional[str] = None
    message: Optional[str] = None
    note: Optional[str] = None


class StepResult(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class TaskDescriptor(BaseModel):
    task_id: str
    title: str
    difficulty: Literal["easy", "medium", "hard"]
    objective: str
    success_criteria: List[str]


class EnvironmentState(BaseModel):
    task_id: str
    task_title: str
    difficulty: str
    step_count: int
    max_steps: int
    done: bool
    observation: Observation
    score: float = 0.0
    target_snapshot: Dict[str, Any] = Field(default_factory=dict)

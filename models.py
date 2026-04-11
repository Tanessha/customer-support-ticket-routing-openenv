from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class TicketCategory(str, Enum):
    billing = "billing"
    technical = "technical"
    general = "general"
    spam = "spam"


class TicketPriority(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class CustomerTier(str, Enum):
    free = "free"
    premium = "premium"


class TicketSentiment(str, Enum):
    calm = "calm"
    neutral = "neutral"
    angry = "angry"


class TicketView(BaseModel):
    id: int
    text: str
    channel: str
    customer_tier: CustomerTier
    created_at_step: int
    sla_hours: int
    sentiment: TicketSentiment
    response_required: bool = False
    predicted_category: Optional[TicketCategory] = None
    predicted_priority: Optional[TicketPriority] = None
    predicted_response: Optional[str] = None


class Action(BaseModel):
    ticket_id: int
    category: TicketCategory
    priority: TicketPriority
    response: Optional[str] = None


class Observation(BaseModel):
    task_id: str
    goal: str
    difficulty: str
    current_step: int
    tickets: List[TicketView]
    processed_ticket_ids: List[int] = Field(default_factory=list)
    remaining_tickets: int


class State(BaseModel):
    task_id: str
    goal: str
    difficulty: str
    current_step: int
    total_tickets: int
    processed_ticket_ids: List[int] = Field(default_factory=list)
    predictions: Dict[int, Dict[str, Optional[str]]] = Field(default_factory=dict)
    action_history: List[Dict[str, object]] = Field(default_factory=list)
    unresolved_high_priority: int = 0
    cumulative_reward: float = 0.0
    latest_grader_breakdown: Dict[int, Dict[str, float]] = Field(default_factory=dict)
    done: bool = False


class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, object] = Field(default_factory=dict)

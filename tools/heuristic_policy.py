from __future__ import annotations

from typing import Any

from models import Action
from tools.policy_engine import TicketPolicyEngine


def fallback_ticket_policy(task: dict[str, Any], observation: dict[str, Any]) -> Action:
    """Deterministic fallback policy for offline/local evaluation reliability."""
    engine = TicketPolicyEngine()
    return engine.build_action(task, observation)

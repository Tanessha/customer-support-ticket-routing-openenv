from __future__ import annotations

from typing import Dict, List, Tuple

from .models import Observation, TicketStatus


def _contains_all(text: str, snippets: List[str]) -> Tuple[int, int]:
    lowered = text.lower()
    hits = sum(1 for snippet in snippets if snippet.lower() in lowered)
    return hits, len(snippets)


def _contains_any(text: str, snippets: List[str]) -> bool:
    lowered = text.lower()
    return any(snippet.lower() in lowered for snippet in snippets)


def grade_episode(observation: Observation, target: Dict, used_articles: List[str]) -> Dict:
    reply = observation.draft_reply or ""

    article_hits = len(set(target.get("required_articles", [])) & set(used_articles))
    article_total = max(1, len(target.get("required_articles", [])))
    tag_hits = len(set(target.get("required_tags", [])) & set(observation.applied_tags))
    tag_total = max(1, len(target.get("required_tags", [])))

    reply_hits, reply_total = _contains_all(reply, target.get("reply_must_include", []))
    exclude_terms = target.get("reply_must_exclude", [])
    exclude_penalty = 0.0
    if exclude_terms and _contains_any(reply, exclude_terms):
        exclude_penalty = 0.2

    any_bonus = 1.0 if _contains_any(reply, target.get("reply_any_of", [])) else 0.0
    priority_score = 1.0 if (observation.priority and observation.priority.value == target.get("priority")) else 0.0
    status_score = 1.0 if observation.status.value == target.get("final_status") else 0.0

    components = {
        "article_usage": article_hits / article_total,
        "tagging": tag_hits / tag_total,
        "priority": priority_score,
        "reply_required_content": reply_hits / max(1, reply_total),
        "reply_supporting_content": any_bonus,
        "final_status": status_score,
    }

    weighted_score = (
        0.20 * components["article_usage"]
        + 0.15 * components["tagging"]
        + 0.15 * components["priority"]
        + 0.25 * components["reply_required_content"]
        + 0.10 * components["reply_supporting_content"]
        + 0.15 * components["final_status"]
    )
    weighted_score = max(0.0, min(1.0, weighted_score - exclude_penalty))

    return {
        "score": round(weighted_score, 4),
        "components": components,
        "exclude_penalty": exclude_penalty,
        "done": observation.status in {TicketStatus.resolved, TicketStatus.escalated, TicketStatus.pending},
    }
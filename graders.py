from __future__ import annotations

from typing import Dict, Optional


def _normalize_text(text: Optional[str]) -> str:
    return (text or "").strip().lower()


def _priority_accuracy(predicted_priority: Optional[str], truth_priority: str, score_priority: bool) -> float:
    if not score_priority:
        return 1.0
    return 1.0 if predicted_priority == truth_priority else 0.0


def response_score(predicted_response: Optional[str], expected_keywords: list[str], required: bool) -> float:
    """Return a deterministic response relevance score in [0, 1]."""
    if not required:
        return 1.0
    if not predicted_response:
        return 0.0

    lowered = _normalize_text(predicted_response)
    hits = sum(1 for keyword in expected_keywords if keyword.lower() in lowered)
    if not expected_keywords:
        return 1.0
    return hits / len(expected_keywords)


def response_safety_penalty(predicted_response: Optional[str], forbidden_keywords: list[str]) -> float:
    """Penalize unsafe promises or prohibited wording in a deterministic way."""
    lowered = _normalize_text(predicted_response)
    if not lowered:
        return 0.0

    hits = sum(1 for keyword in forbidden_keywords if keyword.lower() in lowered)
    if hits == 0:
        return 0.0
    return min(1.0, hits / max(1, len(forbidden_keywords)))


def response_format_score(predicted_response: Optional[str], required: bool, target_category: str) -> float:
    """Reward concise response format and correct silence on spam tickets."""
    lowered = _normalize_text(predicted_response)
    if not required:
        return 1.0
    if target_category == "spam":
        return 1.0 if not lowered else 0.0
    if not lowered:
        return 0.0

    word_count = len(lowered.split())
    if 4 <= word_count <= 50:
        return 1.0
    if 1 <= word_count < 4 or 50 < word_count <= 80:
        return 0.5
    return 0.0


def grade_ticket(
    predicted: Dict[str, Optional[str]],
    truth: Dict[str, object],
    score_priority: bool,
    require_response: bool,
) -> Dict[str, float]:
    """Grade one ticket using deterministic weighted scoring."""

    category_correct = predicted.get("category") == truth["category"]
    category_accuracy = 1.0 if category_correct else 0.0
    priority_accuracy = _priority_accuracy(predicted.get("priority"), truth["priority"], score_priority)
    priority_correct = priority_accuracy == 1.0 if score_priority else True
    content_component = response_score(
        predicted.get("response"),
        truth.get("response_keywords", []),
        require_response,
    )
    format_component = response_format_score(
        predicted.get("response"),
        require_response,
        str(truth["category"]),
    )
    safety_penalty = response_safety_penalty(
        predicted.get("response"),
        truth.get("forbidden_keywords", []),
    )

    response_component = max(0.0, min(1.0, (0.7 * content_component) + (0.3 * format_component) - (0.5 * safety_penalty)))
    if not require_response:
        response_component = 1.0

    weighted_score = (
        0.5 * category_accuracy
        + 0.3 * priority_accuracy
        + 0.2 * response_component
    )
    weighted_score = max(0.0, min(1.0, weighted_score))

    return {
        "score": round(weighted_score, 4),
        "category_accuracy": round(category_accuracy, 4),
        "priority_accuracy": round(priority_accuracy, 4),
        "response_quality": round(response_component, 4),
        "category_correct": 1.0 if category_correct else 0.0,
        "priority_correct": 1.0 if priority_correct else 0.0,
        "response_content_score": round(content_component, 4),
        "response_format_score": round(format_component, 4),
        "response_safety_penalty": round(safety_penalty, 4),
        "response_score": round(response_component, 4),
    }


def grade_episode(task: Dict[str, object], predictions: Dict[int, Dict[str, Optional[str]]]) -> float:
    """Return the mean deterministic episode score in [0, 1]."""
    ground_truth = task["ground_truth"]
    per_ticket_scores = []

    for ticket in task["tickets"]:
        ticket_id = ticket["id"]
        predicted = predictions.get(ticket_id, {})
        ticket_grade = grade_ticket(
            predicted=predicted,
            truth=ground_truth[ticket_id],
            score_priority=bool(task["score_priority"]),
            require_response=bool(task["require_response"]),
        )
        per_ticket_scores.append(ticket_grade["score"])

    if not per_ticket_scores:
        return 0.0
    return round(sum(per_ticket_scores) / len(per_ticket_scores), 4)


def grade_episode_breakdown(task: Dict[str, object], predictions: Dict[int, Dict[str, Optional[str]]]) -> Dict[str, object]:
    """Return detailed deterministic grader output for the full episode."""
    ground_truth = task["ground_truth"]
    ticket_breakdown: Dict[int, Dict[str, float]] = {}

    for ticket in task["tickets"]:
        ticket_id = ticket["id"]
        ticket_breakdown[ticket_id] = grade_ticket(
            predicted=predictions.get(ticket_id, {}),
            truth=ground_truth[ticket_id],
            score_priority=bool(task["score_priority"]),
            require_response=bool(task["require_response"]),
        )

    coverage = len(predictions) / max(1, len(task["tickets"]))
    episode_score = grade_episode(task, predictions)

    return {
        "score": episode_score,
        "base_score": episode_score,
        "completion_bonus": 0.0,
        "coverage": round(coverage, 4),
        "tickets": ticket_breakdown,
    }

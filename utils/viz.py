from __future__ import annotations

from typing import Any

import pandas as pd


def build_visual_frames(
    cumulative_reward: float,
    reward_history: list[dict[str, float]],
    grader_payload: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build chart dataframes for progress and grading diagnostics."""
    if not reward_history:
        progress_df = pd.DataFrame(
            [
                {
                    "step": 0,
                    "cumulative_reward": float(cumulative_reward),
                    "episode_score": float(grader_payload.get("score", 0.0)),
                }
            ]
        )
    else:
        progress_df = pd.DataFrame(reward_history)

    ticket_rows: list[dict[str, float | str]] = []
    tickets = grader_payload.get("tickets", {})
    if isinstance(tickets, dict):
        for ticket_id, breakdown in tickets.items():
            if isinstance(breakdown, dict):
                ticket_rows.append(
                    {
                        "ticket_id": str(ticket_id),
                        "score": float(breakdown.get("score", 0.0)),
                    }
                )

    ticket_df = pd.DataFrame(ticket_rows) if ticket_rows else pd.DataFrame([{"ticket_id": "none", "score": 0.0}])

    if ticket_rows:
        n = len(ticket_rows)
        category_acc = 0.0
        priority_acc = 0.0
        response_quality = 0.0
        for _, breakdown in tickets.items():
            if isinstance(breakdown, dict):
                category_acc += float(breakdown.get("category_accuracy", 0.0))
                priority_acc += float(breakdown.get("priority_accuracy", 0.0))
                response_quality += float(breakdown.get("response_quality", 0.0))

        quality_df = pd.DataFrame(
            [
                {"metric": "category_accuracy", "value": round(category_acc / n, 4)},
                {"metric": "priority_accuracy", "value": round(priority_acc / n, 4)},
                {"metric": "response_quality", "value": round(response_quality / n, 4)},
                {"metric": "urgency_order_score", "value": float(grader_payload.get("urgency_order_score", 0.0))},
                {"metric": "coverage", "value": float(grader_payload.get("coverage", 0.0))},
                {"metric": "episode_score", "value": float(grader_payload.get("score", 0.0))},
            ]
        )
    else:
        quality_df = pd.DataFrame(
            [
                {"metric": "category_accuracy", "value": 0.0},
                {"metric": "priority_accuracy", "value": 0.0},
                {"metric": "response_quality", "value": 0.0},
                {"metric": "urgency_order_score", "value": float(grader_payload.get("urgency_order_score", 0.0))},
                {"metric": "coverage", "value": float(grader_payload.get("coverage", 0.0))},
                {"metric": "episode_score", "value": float(grader_payload.get("score", 0.0))},
            ]
        )

    return progress_df, ticket_df, quality_df

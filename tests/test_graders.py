from __future__ import annotations

import unittest

from graders import grade_episode, grade_episode_breakdown, grade_ticket
from tasks import TASKS


class GraderTests(unittest.TestCase):
    def test_unprocessed_ticket_scores_zero(self) -> None:
        task = TASKS["task_hard_full_resolution"]
        truth = task["ground_truth"][20]
        result = grade_ticket(
            predicted={},
            truth=truth,
            score_priority=True,
            require_response=True,
        )
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["category_accuracy"], 0.0)
        self.assertEqual(result["response_quality"], 0.0)

    def test_episode_score_is_strictly_between_zero_and_one(self) -> None:
        task = TASKS["task_easy_classification"]
        empty_predictions: dict[int, dict[str, str | None]] = {}
        low = grade_episode(task, empty_predictions)
        self.assertGreater(low, 0.0)
        self.assertLess(low, 1.0)

        perfect_predictions = {
            1: {"category": "billing", "priority": "medium", "response": None},
            2: {"category": "spam", "priority": "low", "response": None},
            3: {"category": "general", "priority": "low", "response": None},
        }
        high = grade_episode(task, perfect_predictions, processed_order=[1, 2, 3])
        self.assertGreater(high, 0.0)
        self.assertLess(high, 1.0)

    def test_breakdown_contains_trajectory_metric(self) -> None:
        task = TASKS["task_medium_routing"]
        preds = {
            10: {"category": "billing", "priority": "high", "response": None},
            11: {"category": "technical", "priority": "high", "response": None},
            12: {"category": "general", "priority": "medium", "response": None},
        }
        breakdown = grade_episode_breakdown(task, preds, processed_order=[10, 11, 12])
        self.assertIn("urgency_order_score", breakdown)
        self.assertGreaterEqual(float(breakdown["urgency_order_score"]), 0.0)
        self.assertLessEqual(float(breakdown["urgency_order_score"]), 1.0)


if __name__ == "__main__":
    unittest.main()


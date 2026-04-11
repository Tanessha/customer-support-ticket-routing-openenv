from __future__ import annotations

import unittest

from tasks import TASKS
from tools.policy_engine import TicketPolicyEngine


class PolicyEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = TicketPolicyEngine()

    def test_hard_policy_generates_required_response(self) -> None:
        task = TASKS["task_hard_full_resolution"]
        observation = {
            "tickets": task["tickets"],
            "processed_ticket_ids": [],
        }
        action = self.engine.build_action(task, observation)
        self.assertIsNotNone(action.response)

    def test_repair_enforces_unprocessed_ticket(self) -> None:
        task = TASKS["task_easy_classification"]
        observation = {
            "tickets": task["tickets"],
            "processed_ticket_ids": [1, 2],
        }
        repaired = self.engine.repair_action(task, observation, candidate=None)
        self.assertEqual(repaired.ticket_id, 3)


if __name__ == "__main__":
    unittest.main()

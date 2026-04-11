from __future__ import annotations

import unittest

from environment import CustomerSupportTicketRoutingEnvironment
from models import Action


class EnvironmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = CustomerSupportTicketRoutingEnvironment()

    def test_reset_and_state_consistency(self) -> None:
        obs = self.env.reset("task_easy_classification")
        state = self.env.state()
        self.assertEqual(obs.task_id, "task_easy_classification")
        self.assertEqual(state.task_id, obs.task_id)
        self.assertFalse(state.done)
        self.assertEqual(state.current_step, 0)

    def test_duplicate_ticket_is_penalized(self) -> None:
        self.env.reset("task_easy_classification")
        action = Action(ticket_id=1, category="billing", priority="medium", response=None)
        _ = self.env.step(action)
        second = self.env.step(action)
        self.assertFalse(second.done)
        self.assertLess(second.reward, 0.0)
        self.assertIn("already processed", str(second.info.get("message", "")).lower())

    def test_episode_completion(self) -> None:
        self.env.reset("task_easy_classification")
        self.env.step(Action(ticket_id=1, category="billing", priority="medium", response=None))
        self.env.step(Action(ticket_id=2, category="spam", priority="low", response=None))
        final = self.env.step(Action(ticket_id=3, category="general", priority="low", response=None))
        self.assertTrue(final.done)
        final_state = self.env.state()
        self.assertEqual(len(final_state.processed_ticket_ids), final_state.total_tickets)

    def test_state_tracks_history_and_urgent_backlog(self) -> None:
        self.env.reset("task_hard_full_resolution")
        state0 = self.env.state()
        self.assertEqual(state0.unresolved_high_priority, 2)
        self.assertEqual(len(state0.action_history), 0)

        self.env.step(Action(ticket_id=21, category="technical", priority="high", response="We are urgently investigating login issues."))
        state1 = self.env.state()
        self.assertEqual(state1.unresolved_high_priority, 1)
        self.assertEqual(len(state1.action_history), 1)


if __name__ == "__main__":
    unittest.main()

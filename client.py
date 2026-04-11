"""HTTP client for the Customer Support Ticket Routing OpenEnv environment."""

from typing import Dict

from openenv_core.client_types import StepResult
from openenv_core.env_server.types import State
from openenv_core.http_env_client import HTTPEnvClient

from models import Action, Observation


class CustomerSupportTicketRoutingEnv(HTTPEnvClient[Action, Observation]):
    """Client for interacting with a running customer-support-routing environment server."""

    def _step_payload(self, action: Action) -> Dict:
        return action.model_dump(mode="json")

    def _parse_result(self, payload: Dict) -> StepResult[Observation]:
        obs_data = payload.get("observation", {})
        observation = Observation.model_validate(obs_data)
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("task_id"),
            step_count=payload.get("current_step", 0),
        )

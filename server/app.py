from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from environment import CustomerSupportTicketRoutingEnvironment
from models import Action, Observation, State, StepResult
from tasks import TASKS, list_tasks


app = FastAPI(
    title="Customer Support Ticket Routing Environment",
    version="1.0.0",
    description="An OpenEnv-compatible customer support ticket routing environment.",
)

env = CustomerSupportTicketRoutingEnvironment()


class ResetRequest(BaseModel):
    task_id: Optional[str] = None


@app.get("/")
def root() -> dict[str, str]:
    return {"status": "ok", "environment": "customer-support-ticket-routing"}


@app.post("/reset", response_model=Observation)
def reset(request: Optional[ResetRequest] = None) -> Observation:
    task_id = request.task_id if request else None
    if task_id and task_id not in TASKS:
        raise HTTPException(status_code=404, detail=f"Unknown task_id: {task_id}")
    return env.reset(task_id)


@app.post("/step", response_model=StepResult)
def step(action: Action) -> StepResult:
    return env.step(action)


@app.get("/state", response_model=State)
def state() -> State:
    return env.state()


@app.get("/tasks")
def tasks() -> dict[str, object]:
    return {
        "tasks": list_tasks(),
        "action_schema": Action.model_json_schema(),
    }


@app.get("/grader")
def grader() -> dict[str, object]:
    return {"task_id": env.state().task_id, **env.grader_breakdown()}

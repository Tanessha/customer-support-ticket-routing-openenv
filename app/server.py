from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .env import SupportOpsEnv
from .models import Action, Observation, StepResult
from .tasks import TASKS, list_task_descriptors


app = FastAPI(
    title="OpenEnv Support Ops",
    description="A real-world customer support triage environment for OpenEnv.",
    version="0.1.0",
)

ENV = SupportOpsEnv()


class ResetRequest(BaseModel):
    task_id: Optional[str] = None


@app.get("/")
def root() -> Dict[str, str]:
    return {"status": "ok", "environment": "support-ops-openenv"}


@app.post("/reset", response_model=Observation)
def reset(request: Optional[ResetRequest] = None) -> Observation:
    task_id = (request.task_id if request else None) or ENV.default_task_id
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail=f"Unknown task_id: {task_id}")
    return ENV.reset(task_id)


@app.post("/step", response_model=StepResult)
def step(action: Action) -> StepResult:
    return ENV.step(action)


@app.get("/state")
def state() -> Dict:
    return ENV.state().model_dump()


@app.get("/tasks")
def tasks() -> Dict:
    return {
        "tasks": [descriptor.model_dump() for descriptor in list_task_descriptors()],
        "action_schema": Action.model_json_schema(),
    }


@app.get("/grader")
def grader() -> Dict:
    return ENV.grader()


@app.post("/baseline")
def baseline() -> Dict:
    project_root = Path(__file__).resolve().parent.parent
    env = os.environ.copy()
    cmd = [sys.executable, str(project_root / "baseline.py"), "--json"]
    completed = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True, env=env, check=False)
    if completed.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Baseline script failed.",
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            },
        )
    return json.loads(completed.stdout)
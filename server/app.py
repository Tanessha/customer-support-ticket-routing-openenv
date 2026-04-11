from __future__ import annotations

import os
from typing import Optional

import gradio as gr
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from environment import CustomerSupportTicketRoutingEnvironment
from models import Action, Observation, State, StepResult
from tasks import TASKS, list_tasks
from utils.json_utils import model_or_value_to_json
from utils.viz import build_visual_frames


app = FastAPI(
    title="Customer Support Ticket Routing Environment",
    version="1.0.0",
    description="An OpenEnv-compatible customer support ticket routing environment.",
)

env = CustomerSupportTicketRoutingEnvironment()
ui_env = CustomerSupportTicketRoutingEnvironment()
ui_reward_history: list[dict[str, float]] = []


class ResetRequest(BaseModel):
    task_id: Optional[str] = None


@app.get("/")
def root() -> dict[str, str]:
    return {"status": "ok", "environment": "customer-support-ticket-routing"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


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


def _json_dump(value: object) -> dict[str, object]:
    return model_or_value_to_json(value)


def _reset_visual_buffers() -> None:
    ui_reward_history.clear()


def _build_visual_frames(current_state: State, grader_payload: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return build_visual_frames(
        cumulative_reward=float(current_state.cumulative_reward),
        reward_history=ui_reward_history,
        grader_payload=grader_payload,
    )


def _ui_reset(task_id: str) -> tuple[dict[str, object], dict[str, object], dict[str, object], str, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _reset_visual_buffers()
    observation = ui_env.reset(task_id or None)
    current_state = ui_env.state()
    grader_payload = _json_dump({"task_id": current_state.task_id, **ui_env.grader_breakdown()})
    progress_df, ticket_df, quality_df = _build_visual_frames(current_state, grader_payload)
    return (
        _json_dump(observation),
        _json_dump(current_state),
        grader_payload,
        f"Reset complete: task={current_state.task_id}",
        progress_df,
        ticket_df,
        quality_df,
    )


def _ui_step(
    ticket_id: float,
    category: str,
    priority: str,
    response: str,
) -> tuple[dict[str, object], dict[str, object], dict[str, object], str, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    action = Action(
        ticket_id=int(ticket_id),
        category=category,
        priority=priority,
        response=response.strip() or None,
    )
    result = ui_env.step(action)
    current_state = ui_env.state()
    grader_payload = _json_dump({"task_id": current_state.task_id, **ui_env.grader_breakdown()})
    step_n = int(result.info.get("current_step", current_state.current_step)) if isinstance(result.info, dict) else int(current_state.current_step)
    episode_score = float(result.info.get("episode_score", grader_payload.get("score", 0.0))) if isinstance(result.info, dict) else float(grader_payload.get("score", 0.0))
    ui_reward_history.append(
        {
            "step": float(step_n),
            "cumulative_reward": float(current_state.cumulative_reward),
            "episode_score": episode_score,
        }
    )
    progress_df, ticket_df, quality_df = _build_visual_frames(current_state, grader_payload)
    return (
        _json_dump(result),
        _json_dump(current_state),
        grader_payload,
        "Step processed.",
        progress_df,
        ticket_df,
        quality_df,
    )


def _ui_state() -> tuple[dict[str, object], str, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    current_state = ui_env.state()
    grader_payload = _json_dump({"task_id": current_state.task_id, **ui_env.grader_breakdown()})
    progress_df, ticket_df, quality_df = _build_visual_frames(current_state, grader_payload)
    return _json_dump(current_state), "State updated.", progress_df, ticket_df, quality_df


def _ui_tasks() -> tuple[dict[str, object], str]:
    return _json_dump({"tasks": list_tasks(), "action_schema": Action.model_json_schema()}), "Tasks loaded."


def _ui_grader() -> tuple[dict[str, object], str, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    current_state = ui_env.state()
    grader_payload = _json_dump({"task_id": current_state.task_id, **ui_env.grader_breakdown()})
    progress_df, ticket_df, quality_df = _build_visual_frames(current_state, grader_payload)
    return grader_payload, "Grader updated.", progress_df, ticket_df, quality_df


def _build_web_ui() -> gr.Blocks:
    initial_state = ui_env.state()
    initial_grader = _json_dump({"task_id": initial_state.task_id, **ui_env.grader_breakdown()})
    initial_progress, initial_tickets, initial_quality = _build_visual_frames(initial_state, initial_grader)

    with gr.Blocks(title="Customer Support Ticket Routing Web UI") as demo:
        gr.Markdown("# Customer Support Ticket Routing Environment")
        gr.Markdown("Interactive controls for `/reset`, `/step`, `/state`, `/tasks`, and `/grader`.")

        with gr.Row():
            task_id = gr.Dropdown(
                choices=list(TASKS.keys()),
                value=env.state().task_id,
                label="Task ID",
            )
            reset_btn = gr.Button("Reset Task", variant="primary")
            state_btn = gr.Button("Get State")
            tasks_btn = gr.Button("Get Tasks")
            grader_btn = gr.Button("Get Grader")

        with gr.Row():
            ticket_id = gr.Number(label="ticket_id", value=1, precision=0)
            category = gr.Dropdown(
                choices=["billing", "technical", "general", "spam"],
                value="general",
                label="category",
            )
            priority = gr.Dropdown(
                choices=["low", "medium", "high"],
                value="medium",
                label="priority",
            )

        response = gr.Textbox(
            label="response (optional)",
            placeholder="Only required for hard task when response_required=true",
            lines=3,
        )
        step_btn = gr.Button("Step", variant="primary")

        with gr.Row():
            step_json = gr.JSON(label="Step Result")
            state_json = gr.JSON(label="State")
        grader_json = gr.JSON(label="Grader")
        tasks_json = gr.JSON(label="Tasks")
        status_box = gr.Textbox(label="Status", interactive=False)

        gr.Markdown("## Evaluation Visuals")
        reward_plot = gr.LinePlot(
            value=initial_progress,
            x="step",
            y="cumulative_reward",
            title="Cumulative Reward by Step",
            x_title="Step",
            y_title="Cumulative Reward",
        )
        with gr.Row():
            ticket_plot = gr.BarPlot(
                value=initial_tickets,
                x="ticket_id",
                y="score",
                title="Per-Ticket Score",
                y_lim=[0, 1],
            )
            quality_plot = gr.BarPlot(
                value=initial_quality,
                x="metric",
                y="value",
                title="Quality Components",
                y_lim=[0, 1],
            )

        reset_btn.click(
            _ui_reset,
            inputs=[task_id],
            outputs=[step_json, state_json, grader_json, status_box, reward_plot, ticket_plot, quality_plot],
        )
        step_btn.click(
            _ui_step,
            inputs=[ticket_id, category, priority, response],
            outputs=[step_json, state_json, grader_json, status_box, reward_plot, ticket_plot, quality_plot],
        )
        state_btn.click(_ui_state, outputs=[state_json, status_box, reward_plot, ticket_plot, quality_plot])
        tasks_btn.click(_ui_tasks, outputs=[tasks_json, status_box])
        grader_btn.click(_ui_grader, outputs=[grader_json, status_box, reward_plot, ticket_plot, quality_plot])

    return demo


app = gr.mount_gradio_app(app, _build_web_ui(), path="/web")


def main() -> None:
    uvicorn.run("server.app:app", host="0.0.0.0", port=int(os.getenv("PORT", "7860")))


if __name__ == "__main__":
    main()

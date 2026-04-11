---
title: Customer Support Ticket Routing Environment
sdk: docker
app_port: 7860
tags:
  - openenv
  - customer-support
  - ticket-routing
  - agentic-evaluation
---

# Customer Support Ticket Routing Environment

A production-style OpenEnv environment that simulates support operations triage under SLA pressure.

This project is intentionally designed to align with the architecture and evaluation patterns used in stronger OpenEnv references: clear API contracts, deterministic grading, trajectory-aware evaluation, and an interactive debugging surface.

## Overview

The environment evaluates an agent on three levels of support-ops difficulty:

1. Easy: category classification (`billing`, `technical`, `general`, `spam`)
2. Medium: category + priority routing (`low`, `medium`, `high`)
3. Hard: category + priority + safe, relevant response drafting

The grader returns deterministic scores while preserving dense reward signals in `step()`.

## Architecture

```text
+-------------------------------+
| inference.py                  |
| - OpenAI client call loop     |
| - strict START/STEP/END logs  |
| - episode reports in outputs/ |
+---------------+---------------+
                |
                v
+-------------------------------+
| server/app.py                 |
| FastAPI + Gradio (/web)       |
| /reset /step /state /tasks    |
| /grader + visual diagnostics  |
+---------------+---------------+
                |
                v
+-------------------------------+
| environment.py                |
| CustomerSupport...Environment |
| - state transitions           |
| - action validation           |
| - dense reward shaping        |
+---------------+---------------+
                |
                v
+-------------------------------+
| graders.py                    |
| - deterministic per-ticket    |
| - trajectory-aware metric     |
| - strict task score bounds    |
+---------------+---------------+
                |
                v
+-------------------------------+
| tasks.py / models.py          |
| - task fixtures               |
| - pydantic schemas            |
+-------------------------------+
```

## Notable Design Decisions

- Deterministic grader: same trajectory produces same score every run.
- Strict scoring bounds: task score is always strictly in `(0,1)` to satisfy validator constraints.
- Trajectory-quality scoring: includes an urgency ordering signal to reward handling high-priority tickets earlier.
- UI diagnostics: `/web` includes status updates + charts for cumulative reward and per-ticket score breakdowns.
- Resilience: inference loop falls back to deterministic policy when provider calls fail.

## Unique Feature (High Impact)

This environment adds **trajectory-quality KPI scoring** on top of correctness:

- `urgency_order_score`: pairwise ranking metric that measures whether high-priority tickets were resolved before lower-priority ones.
- `deferred_urgent_penalty`: step-level penalty if low-priority tickets are processed while urgent tickets remain unresolved.
- `missing_required_response_penalty`: hard-task penalty for skipping customer-facing response when required.

This makes evaluation more agentic than static label matching and better reflects real support operations behavior.

## Multi-Step Workflow Upgrades

- Action history is now persisted in state (`action_history`) for trajectory-aware debugging.
- State exposes `unresolved_high_priority` so agents can reason about backlog pressure.
- Inference includes a policy-repair layer:
  1. LLM proposes action
  2. Policy engine validates/repairs ticket selection and arguments
  3. Recovery path auto-corrects invalid/unsafe actions

## API

- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /tasks`
- `GET /grader`
- `GET /health`
- `GET /web` (interactive UI)

## Quickstart

```bash
pip install -r requirements.txt
py -3.12 -m uvicorn server.app:app --host 127.0.0.1 --port 7860
```

Open:

- `http://127.0.0.1:7860/web`
- `http://127.0.0.1:7860/health`

## Inference Script Contract

`inference.py` prints exactly these line types per episode:

- `[START] task=<task_name> env=<benchmark> model=<model_name>`
- `[STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>`
- `[END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>`

Mandatory env vars:

- `HF_TOKEN`
- `API_BASE_URL` (default set)
- `MODEL_NAME` (default set)
- Optional: `LOCAL_IMAGE_NAME`

Run:

```bash
HF_TOKEN=... py -3.12 inference.py
```

Artifacts:

- `outputs/episode_reports.jsonl`

## Docker

```bash
docker build --network=host -t customer-support-ticket-routing .
docker run -p 7860:7860 customer-support-ticket-routing
```

## Validation

```bash
openenv validate
bash pre_validation.sh
```

## Tests

Unit tests are included for grader and environment invariants:

```bash
py -3.12 -m unittest discover -s tests -p "test_*.py"
```

## Sample Agent Walkthrough

Example high-quality trajectory for `task_hard_full_resolution`:

1. `ticket_id=20`, `category=billing`, `priority=high`, response contains `billing/refund/review`
2. `ticket_id=21`, `category=technical`, `priority=high`, response contains `investigating/login/urgent`
3. `ticket_id=22`, `category=general`, `priority=medium`, response contains `workspace/admin/help`

Expected:

- `done=true`
- high cumulative reward
- near-max task score (but still strict `< 1.0` by design)
- `response_safety_penalty=0`

## Project Layout

- `models.py`
- `tasks.py`
- `graders.py`
- `environment.py`
- `server/app.py`
- `tools/heuristic_policy.py`
- `utils/json_utils.py`
- `utils/viz.py`
- `inference.py`
- `openenv.yaml`
- `pyproject.toml`
- `Dockerfile`
- `server/Dockerfile`
- `tests/`
- `outputs/`

---
title: Customer Support Ticket Routing Environment
sdk: docker
app_port: 7860
tags:
  - openenv
  - customer-support
  - ticket-routing
---

# Customer Support Ticket Routing Environment

This project is an OpenEnv-compatible environment for a real customer support workflow: routing inbound tickets by category, operational priority, and, on the hardest task, drafting short customer-facing responses. The benchmark is intentionally grounded in work support teams actually do every day rather than synthetic puzzle tasks.

The design is optimized for hackathon scoring in three areas:

- real-world utility through realistic support contexts such as billing disputes, production incidents, self-serve questions, and spam
- grader quality through deterministic per-ticket scoring, response relevance checks, safety penalties, and episode-level coverage tracking
- reward design through dense partial rewards that credit incremental progress while still penalizing clear routing mistakes

## Action Space

`Action` is defined in [models.py](/C:/OpenEnv/models.py).

- `ticket_id: int`
- `category: billing | technical | general | spam`
- `priority: low | medium | high`
- `response: Optional[str]`

## Observation Space

`Observation` is defined in [models.py](/C:/OpenEnv/models.py).

- `task_id`
- `goal`
- `difficulty`
- `current_step`
- `tickets`
- `processed_ticket_ids`
- `remaining_tickets`

Each ticket includes operational context that makes routing feel more realistic:

- `id`
- `text`
- `channel`
- `customer_tier`
- `created_at_step`
- `sla_hours`
- `sentiment`
- `response_required`
- `predicted_category`
- `predicted_priority`
- `predicted_response`

## State

`State` is defined in [models.py](/C:/OpenEnv/models.py) and returned by `state()`.

It includes:

- task metadata
- processed ticket ids
- current predictions
- cumulative reward
- latest grader breakdown
- done flag

## Reward Design

The environment provides dense per-ticket reward instead of only terminal success:

- correct category: `+0.4`
- correct priority: `+0.3`
- fast resolution within SLA: `+0.2`
- relevant response: up to `+0.1`
- wrong classification: `-0.3`
- ignoring a high-priority ticket: `-0.5`
- SLA delay: increasing `-0.05` per overdue step

Additional scoring properties improve learning value:

- each `step()` increments simulated time
- response quality is split into content relevance and response format
- unsafe wording such as overpromising or guarantees is penalized deterministically
- duplicate or invalid ticket processing is penalized at step time

This makes the reward function useful for partial progress while still preserving a deterministic final score in `[0.0, 1.0]`.

## Tasks

Three tasks are defined in [tasks.py](/C:/OpenEnv/tasks.py):

1. `task_easy_classification`
   Simple queue classification focused on distinguishing valid support from spam.
2. `task_medium_routing`
   Classification plus priority assignment using realistic context like account tier and urgency.
3. `task_hard_full_resolution`
   Classification, priority assignment, and concise customer response generation with relevance and safety checks.

Each task includes:

- concrete ticket inputs
- deterministic ground truth
- explicit response requirements
- a deterministic grader returning a `0.0` to `1.0` score

## Grader Design

The grader logic lives in [graders.py](/C:/OpenEnv/graders.py).

Per ticket, it measures:

- category correctness
- priority correctness
- response content match against required keywords
- response format quality
- response safety penalty against forbidden promises

At the episode level, it tracks:

- mean ticket score
- ticket coverage
- small completion bonus for fully processed queues
- per-ticket breakdown for debugging and analysis

## API

The FastAPI app is implemented in [server/app.py](/C:/OpenEnv/server/app.py).

- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /tasks`
- `GET /grader`

## Project Files

- [models.py](/C:/OpenEnv/models.py)
- [environment.py](/C:/OpenEnv/environment.py)
- [tasks.py](/C:/OpenEnv/tasks.py)
- [graders.py](/C:/OpenEnv/graders.py)
- [server/app.py](/C:/OpenEnv/server/app.py)
- [inference.py](/C:/OpenEnv/inference.py)
- [openenv.yaml](/C:/OpenEnv/openenv.yaml)
- [Dockerfile](/C:/OpenEnv/Dockerfile)

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the API server:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

Run the Hugging Face baseline:

```bash
export HF_TOKEN=...
export HF_PROVIDER=hf-inference
export MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
python inference.py
```

Optional OpenAI-compatible local or hosted gateway:

```bash
export API_BASE_URL=https://your-compatible-endpoint/v1
```

The baseline tries Hugging Face inference first and falls back to a deterministic local agent if the hosted provider/model route is unavailable, so it remains reproducible for submission checks.

## Docker

Build:

```bash
docker build -t customer-support-ticket-routing .
```

Run:

```bash
docker run -p 7860:7860 customer-support-ticket-routing
```

## Validation

Once the server is running, validate the environment with your OpenEnv validator against:

- `/reset`
- `/step`
- `/state`
- `/tasks`
- `/grader`

The environment metadata is defined in [openenv.yaml](/C:/OpenEnv/openenv.yaml).

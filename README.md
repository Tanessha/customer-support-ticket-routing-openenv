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

A real-world OpenEnv environment for routing inbound customer support tickets by category, priority, and (for hard tasks) concise reply generation.

## What This Submission Includes

- Real-world environment and deterministic grader
- OpenEnv-compatible API (`/reset`, `/step`, `/state`, `/tasks`, `/grader`)
- Interactive frontend at `/web` (Gradio)
- Submission-ready `inference.py` using OpenAI Client against HF router
- `pre_validation.sh` with 3 checks: inference script, Docker build, `openenv validate`

## Tasks

- `task_easy_classification`
- `task_medium_routing`
- `task_hard_full_resolution`

## Mandatory Env Vars (Inference)

- `HF_TOKEN` (or `API_KEY`) for auth
- `API_BASE_URL` (default: `https://router.huggingface.co/v1`)
- `MODEL_NAME` (default: `Qwen/Qwen2.5-72B-Instruct`)
- `LOCAL_IMAGE_NAME` (optional, for docker-image workflows)
- `MY_ENV_V4_TASK` (optional: a task id or `all`)

## Inference Stdout Contract

`inference.py` emits only these line types in order per episode:

- `[START] task=<task_name> env=<benchmark> model=<model_name>`
- `[STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>`
- `[END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>`

## Run Locally

```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

- API: `http://localhost:7860`
- Interactive UI: `http://localhost:7860/web`

Run inference:

```bash
HF_TOKEN=... MODEL_NAME=Qwen/Qwen2.5-72B-Instruct python inference.py
```

## Docker

```bash
docker build -t customer-support-ticket-routing .
docker run -p 7860:7860 customer-support-ticket-routing
```

## Pre-validation

```bash
bash pre_validation.sh
```

## OpenEnv Validation

```bash
openenv validate
```

## Project Files

- `models.py`
- `environment.py`
- `tasks.py`
- `graders.py`
- `server/app.py`
- `inference.py`
- `pre_validation.sh`
- `openenv.yaml`
- `Dockerfile`

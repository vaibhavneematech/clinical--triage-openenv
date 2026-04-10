---
title: Clinical Triage Env
emoji: 🏥
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
---

<div align="center">

# 🏥 ClinicalTriageEnv

### An OpenEnv Reinforcement Learning Environment for Emergency Department Triage

*AI agents step into the role of ED clinicians — assessing patients, ordering diagnostics, assigning acuity levels, and making disposition decisions under real-time pressure.*

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)](https://fastapi.tiangolo.com)
[![Next.js](https://img.shields.io/badge/Next.js-16-black.svg)](https://nextjs.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Final Submission](https://img.shields.io/badge/Status-Final_Submission_Ready-success.svg)](#)

**Team Unfazed** · Meta × Scaler OpenEnv Hackathon

---

</div>

## Why This Exists

Emergency triage is one of the highest-stakes sequential decision tasks in medicine:

- **Wrong triage kills** — a delayed STEMI costs heart muscle every minute past the 90-minute door-to-balloon window
- **Over-triage gridlocks the ED** — assigning ESI-1 to a stable patient blocks resuscitation bays
- **No standardized RL benchmark exists** for clinical decision-making under uncertainty

ClinicalTriageEnv is a **fully synthetic**, HIPAA-free simulation that benchmarks AI agents on three clinical scenarios of increasing difficulty — from a clear-cut STEMI to a five-patient mass casualty surge with only three beds.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         ClinicalTriageEnv                          │
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌────────────────────┐    │
│  │   Patient     │    │   Dynamic    │    │    Deterministic   │    │
│  │  Generator    │───▶│   Vitals     │───▶│     Graders        │    │
│  │  (3 tasks)    │    │   Engine     │    │  (STEMI/Chest/MCI) │    │
│  └──────────────┘    └──────────────┘    └────────────────────┘    │
│         │                    │                      │               │
│         ▼                    ▼                      ▼               │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              Environment Engine (step / reset / grade)       │   │
│  │         Dense Reward · Fatal Delay Detection · ESI Logic     │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│              ┌───────────────┼───────────────┐                      │
│              ▼               ▼               ▼                      │
│        REST API         WebSocket       inference.py                │
│       (FastAPI)         (Real-time)    (Agent Runner)               │
│       :7860             :7860/ws      Deterministic + LLM           │
└─────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
                ┌──────────────────────────┐
                │   Brutalist Dashboard    │
                │       (Next.js)          │
                │                          │
                │  ┌────┐ ┌─────┐ ┌────┐  │
                │  │Wait│ │Live │ │Neur│  │
                │  │Room│ │Vital│ │Trac│  │
                │  │    │ │  s  │ │  e │  │
                │  └────┘ └─────┘ └────┘  │
                │    :3000 (dev)           │
                └──────────────────────────┘
```

---

## Features

### 🧬 Dynamic Vitals Engine
Patient vitals deteriorate probabilistically over time. A STEMI patient's blood pressure drops, heart rate climbs, and SpO2 falls unless the agent intervenes. Each action costs simulated time — ordering bloodwork burns 30 minutes, administering epinephrine takes 1 minute.

### ⏱️ Fatal Delay Detection
Miss the 90-minute door-to-balloon window for a STEMI? **-10.0 penalty, episode terminated.** Fail to treat anaphylaxis within 15 minutes? Same. Clinical time windows are enforced with hard stops.

### 📊 Dense Reward Signal
Every step returns a reward composed of 5 weighted components:

| Component | Range | What It Rewards |
|-----------|-------|-----------------|
| Clinical Correctness | +0.10 to +0.30 | Ordering indicated tests, correct ESI assignment |
| Efficiency | -0.05 | Penalty per unnecessary or redundant test |
| Time Pressure | -0.02/step | ESI-1 patients bleed reward each step of delay |
| Sequence Bonus | +0.05 | Following evidence-based diagnostic ordering |
| Safety Guardrails | -0.50 to -10.0 | Discharging ESI-1, fatal window violations |

### 🖥️ Brutalist Clinical Dashboard
A real-time monitoring station built with Next.js:
- **Waiting Room** — Patient queue with chief complaints and vitals at a glance
- **Live Vitals** — Heart rate, blood pressure, SpO2, GCS with sparkline trend graphs
- **Neural Trace** — Watch the agent's `<thought>` reasoning stream live from the LLM
- **Post-Episode Audit** — Letter-grade report card with clinical performance breakdown

### 🧠 ReAct Clinical Reasoning
LLM agent mode uses structured Observation → Thought → Action reasoning with `<thought>` tags. Supports OpenAI, Together AI (Llama-3-70B), and Groq endpoints.

---

## Tasks

### Task 1: `task_stemi_code` · Easy
> 58-year-old male. Crushing substernal chest pain radiating to left arm. ST-elevation in leads V1-V4. Hypotensive (85/50), diaphoretic, HR 110.

**The agent must**: Recognize STEMI → Assign ESI-1 → Activate cath lab → Aspirin 325mg → Admit to ICU. All within the 90-minute window.

**Max Steps**: 15 · **Baseline**: 0.72 · **Optimal**: 0.90

### Task 2: `task_chest_pain_workup` · Medium
> 44-year-old female. Pleuritic chest pain, worse with inspiration. Recent 14-hour transatlantic flight. Currently on oral contraceptives. Wells score elevated.

**Differential**: PE vs ACS vs MSK vs Anxiety. **The agent must**: Navigate the diagnostic sequence — EKG first (rule out ACS), then D-dimer, then CT-PA if positive. Order matters.

**Max Steps**: 20 · **Baseline**: 0.48 · **Optimal**: 0.95

### Task 3: `task_mci_surge` · Hard
> Mass casualty incident. Five patients arrive simultaneously. Three beds available.

| Patient | Presentation | Expected ESI |
|---------|-------------|--------------|
| P1 · 72M | Unresponsive, GCS 6, HR 40, hypotensive | **ESI-1** |
| P2 · 28F | Deformed forearm fracture, stable vitals | ESI-3 |
| P3 · 15M | Anaphylaxis, stridor, BP 70/40, SpO2 88% | **ESI-1** |
| P4 · 60M | Rapid AFib at 148 bpm, dizzy | ESI-2 |
| P5 · 35F | Anxiety, hyperventilation, vitals normal | ESI-4 |

**The agent must**: Triage the sickest first (P1 and P3 before anyone else), allocate beds under scarcity, and manage five concurrent patients without losing anyone.

**Max Steps**: 25 · **Baseline**: 0.31 · **Optimal**: 0.95

---

## Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/malc3om/clinical--triage-openenv.git
cd clinical--triage-openenv
pip install -r requirements.txt
```

### 2. Run the Environment Server

```bash
uvicorn clinical_triage_env.app:app --host 0.0.0.0 --port 7860
```

### 3. Run Inference (Deterministic Optimal)

```bash
python inference.py
```

### 4. Run with LLM Agent (Optional)

```bash
USE_LLM=true API_KEY=<your-key> python inference.py
```

### 5. Launch the Dashboard (Optional)

```bash
cd clinical-triage-dashboard
npm install && npm run dev
# Open http://localhost:3000
```

### Docker

```bash
docker build -t clinical-triage-env .
docker run -p 7860:7860 clinical-triage-env
```

---

## Programmatic API

```python
from clinical_triage_env.models import TriageAction
from clinical_triage_env.server.environment import ClinicalTriageEnvironment

env = ClinicalTriageEnvironment()
obs = env.reset(task_id="task_stemi_code")

action = TriageAction(
    action_type="assign_esi_level",
    patient_id="P1",
    parameter="1",
    rationale="Acute STEMI requires ESI-1"
)
result = env.step(action)
print(f"Reward: {result.reward:.3f} | Done: {result.done}")

# Get grader score at any time
score = env.get_task_grader_score()
print(f"Grader Score: {score.score:.3f}")
```

---

## REST API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `GET` | `/health` | Health check (alias) |
| `POST` | `/reset` | Start new episode `{"task_id": "task_stemi_code"}` |
| `POST` | `/step` | Execute action `{"action_type": "...", "patient_id": "P1", "parameter": "..."}` |
| `GET` | `/state` | Current episode state |
| `GET` | `/tasks` | List all available tasks |
| `POST` | `/grade` | Grade an episode history |
| `WS` | `/ws` | WebSocket for real-time dashboard streaming |

---

## Action Space

| `action_type` | `parameter` examples | Time Cost |
|---------------|---------------------|-----------|
| `order_diagnostic` | `EKG`, `d_dimer`, `CT_PA`, `troponin_I`, `cbc` | 5–30 min |
| `assign_esi_level` | `1`, `2`, `3`, `4`, `5` | 2 min |
| `activate_pathway` | `cath_lab`, `stroke_code`, `trauma` | 5 min |
| `disposition` | `admit`, `discharge`, `transfer`, `waiting_room` | 3 min |
| `request_consult` | `cardiology`, `pulmonology` | 15 min |
| `administer_medication` | `epinephrine`, `aspirin_325mg` | 1 min |
| `assign_bed` | `resus_bay`, `monitored`, `hallway` | 2 min |
| `wait` | `""` | 10 min |

---

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | `str` | Current task identifier |
| `step_number` / `max_steps` | `int` | Progress tracking |
| `elapsed_minutes` | `int` | Simulated time elapsed |
| `patients` | `List[Patient]` | All patients with vitals, labs, history |
| `available_beds` | `int` | Remaining beds |
| `reward` | `float` | Cumulative reward |
| `last_action_result` | `str` | Feedback from last action |
| `done` | `bool` | Episode termination flag |

Each patient includes: `vitals` (HR, BP, SpO2, RR, Temp, GCS), `vitals_trend` (↑/↓/→ per vital), `medical_history`, `available_labs`, `pending_labs`, and `resource_tokens_remaining`.

---

## Inference Output Format

The inference script emits OpenEnv-compatible structured output:

```
[START] task=task_stemi_code env=clinical_triage model=deterministic-optimal-v1
[STEP]  step=1 action={"action_type":"assign_esi_level",...} reward=0.25 done=false error=null
[STEP]  step=2 action={"action_type":"activate_pathway",...}  reward=0.30 done=false error=null
[STEP]  step=3 action={"action_type":"order_diagnostic",...}  reward=0.10 done=false error=null
[STEP]  step=4 action={"action_type":"disposition",...}       reward=0.25 done=true  error=null
[END]   success=true steps=4 score=0.900 rewards=0.25,0.30,0.10,0.25
```

---

## Scores

| Task | Deterministic Optimal | Difficulty |
|------|----------------------|------------|
| `task_stemi_code` | **0.900** | Easy |
| `task_chest_pain_workup` | **0.950** | Medium |
| `task_mci_surge` | **0.950** | Hard |
| **Average** | **0.933** | |

---

## Project Structure

```
clinical_triage_env/
├── inference.py                    # Agent runner (deterministic + LLM modes)
├── run_demo.py                     # Full demo orchestrator
├── validate_submission.py          # Pre-submission validation (8 checks)
├── requirements.txt
├── Dockerfile                      # Multi-stage (dashboard + backend)
├── openenv.yaml                    # OpenEnv task manifest
│
├── clinical_triage_env/            # Core Python package
│   ├── app.py                      # FastAPI server (REST + WebSocket)
│   ├── models.py                   # Pydantic models (observation/action/state)
│   └── server/
│       ├── environment.py          # Main environment engine
│       ├── reward.py               # 5-component dense reward function
│       ├── patient_generator.py    # Seeded patient generation (3 tasks)
│       ├── vitals_engine.py        # Dynamic vitals deterioration
│       ├── time_costs.py           # Action → simulated time mapping
│       ├── graders/                # Deterministic graders
│       │   ├── stemi_grader.py
│       │   ├── chest_workup_grader.py
│       │   └── mci_grader.py
│       └── tasks/                  # Task YAML definitions
│
└── clinical-triage-dashboard/      # Next.js brutalist dashboard
    ├── src/
    │   ├── app/
    │   │   ├── page.tsx            # 4-panel monitoring station
    │   │   ├── layout.tsx          # Root layout with JetBrains Mono
    │   │   └── globals.css         # Terminal-green design system
    │   └── types.ts                # TypeScript interfaces
    ├── package.json
    └── next.config.ts
```

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `API_BASE_URL` | No | `https://api.openai.com/v1` | LLM API endpoint |
| `MODEL_NAME` | No | `gpt-4o-mini` | Model identifier |
| `HF_TOKEN` | Yes* | — | Hugging Face API token |
| `USE_LLM` | No | `false` | Enable LLM agent mode |
| `AGENT_PROVIDER` | No | `openai` | `openai` / `together` / `groq` |
| `API_KEY` | If LLM | — | API key for LLM provider |
| `PORT` | No | `7860` | Server port |

*Required for HF Spaces deployment. Not needed for local deterministic mode.

## Validation

Pre-submission validation passes all mandatory checks:

---

## License

MIT

---

<div align="center">

*Built by **Team Unfazed** for the Meta × Scaler OpenEnv Hackathon*

</div>

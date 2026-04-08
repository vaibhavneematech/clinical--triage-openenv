"""
app.py — FastAPI server for ClinicalTriageEnv.

Exposes the full OpenEnv HTTP API:
  GET  /           → Health check
  POST /reset      → Start new episode
  POST /step       → Execute one action
  GET  /state      → Current episode state
  GET  /tasks      → List all tasks
  POST /grade      → Grade an episode history
  GET  /health     → Health check (alias)

Runs on port 7860 for HF Spaces compatibility.
"""

from __future__ import annotations

import os
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from clinical_triage_env.models import (
    TriageAction,
    TriageObservation,
    TriageState,
    GradeResult,
)
from clinical_triage_env.server.environment import ClinicalTriageEnvironment

# ─── Create FastAPI app ─────────────────────────────────────────────────

app = FastAPI(
    title="ClinicalTriageEnv",
    description=(
        "An OpenEnv environment simulating Emergency Department triage. "
        "AI agent assesses patients, orders diagnostics, assigns ESI levels, "
        "and makes disposition decisions under time pressure and resource constraints."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global environment instance (sufficient for hackathon scope)
env = ClinicalTriageEnvironment()


# ─── Request/Response Models ────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = "task_stemi_code"


class GradeRequest(BaseModel):
    task_id: str
    episode_history: list[dict]


class StepRequest(BaseModel):
    action_type: str
    patient_id: str
    parameter: str
    rationale: Optional[str] = None


# ─── Endpoints ──────────────────────────────────────────────────────────

@app.get("/")
async def root():
    """Health check — returns 200 with environment info."""
    return {
        "status": "ok",
        "env": "clinical-triage-env",
        "version": "1.0.0",
        "description": "Emergency Department Triage Environment for OpenEnv",
    }


@app.get("/health")
async def health():
    """Health check alias."""
    return {"status": "healthy"}


@app.post("/reset", response_model=TriageObservation)
async def reset(request: Optional[ResetRequest] = None):
    """Reset the environment for a new episode."""
    task_id = request.task_id if request else "task_stemi_code"
    try:
        obs = env.reset(task_id=task_id)
        return obs
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step", response_model=TriageObservation)
async def step(request: StepRequest):
    """Execute one action in the environment."""
    try:
        action = TriageAction(
            action_type=request.action_type,
            patient_id=request.patient_id,
            parameter=request.parameter,
            rationale=request.rationale,
        )
        result = env.step(action)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state")
async def get_state():
    """Return current episode state."""
    return env.state.model_dump()


@app.get("/tasks")
async def list_tasks():
    """Return list of all available tasks with descriptions."""
    tasks = env.get_tasks()
    return {
        "tasks": [t.model_dump() for t in tasks],
        "action_schema": {
            "action_type": {
                "type": "enum",
                "values": [
                    "order_diagnostic",
                    "assign_esi_level",
                    "activate_pathway",
                    "disposition",
                    "request_consult",
                    "wait",
                ],
                "description": "Type of clinical action to take",
            },
            "patient_id": {
                "type": "string",
                "description": "ID of the patient this action targets (e.g. P1, P2)",
            },
            "parameter": {
                "type": "string",
                "description": "Action parameter: test name, ESI level (1-5), pathway type, disposition type (admit/discharge/transfer)",
            },
            "rationale": {
                "type": "string",
                "description": "Optional: agent's reasoning for this action",
            },
        },
    }


@app.post("/grade", response_model=GradeResult)
async def grade(request: GradeRequest):
    """Grade an episode history using the deterministic grader."""
    try:
        from clinical_triage_env.server.graders.stemi_grader import grade_stemi
        from clinical_triage_env.server.graders.chest_workup_grader import grade_chest_workup
        from clinical_triage_env.server.graders.mci_grader import grade_mci

        graders = {
            "task_stemi_code": grade_stemi,
            "task_chest_pain_workup": grade_chest_workup,
            "task_mci_surge": grade_mci,
        }

        grader = graders.get(request.task_id)
        if grader is None:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown task_id: {request.task_id}",
            )

        result = grader(request.episode_history)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Run server ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)

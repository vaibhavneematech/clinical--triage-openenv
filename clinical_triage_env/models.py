"""
ClinicalTriageEnv — Type-safe data contracts.

All models use Pydantic v2 and extend the OpenEnv base classes where required.
These models define the complete API surface for the environment.
"""

from __future__ import annotations

from typing import List, Optional, Literal, Dict
from pydantic import BaseModel, Field
from enum import Enum


# ─── Supporting Medical Models (plain Pydantic) ────────────────────────

class VitalSigns(BaseModel):
    """Patient vital signs measured at triage."""
    heart_rate: int = Field(..., description="Beats per minute")
    systolic_bp: int = Field(..., description="mmHg")
    diastolic_bp: int = Field(..., description="mmHg")
    respiratory_rate: int = Field(..., description="Breaths per minute")
    spo2: float = Field(..., description="Oxygen saturation 0-100%")
    temperature: float = Field(..., description="Celsius")
    gcs: int = Field(..., ge=3, le=15, description="Glasgow Coma Scale 3-15")


class LabResult(BaseModel):
    """A single laboratory result."""
    name: str = Field(..., description="e.g. troponin_I, d_dimer, cbc_wbc")
    value: float
    unit: str
    reference_range: str
    critical: bool = Field(False, description="True if critically abnormal")


class PatientState(BaseModel):
    """Complete state of a single patient in the ED."""
    patient_id: str
    age: int
    sex: Literal["M", "F"]
    chief_complaint: str
    onset_minutes: int = Field(..., description="Minutes since symptom onset")
    vitals: VitalSigns
    medical_history: List[str] = Field(default_factory=list)
    current_medications: List[str] = Field(default_factory=list)
    available_labs: List[LabResult] = Field(default_factory=list)
    pending_labs: List[str] = Field(default_factory=list)
    imaging_available: List[str] = Field(default_factory=list)
    pending_imaging: List[str] = Field(default_factory=list)
    time_in_department_minutes: int = 0
    resource_tokens_remaining: int = Field(10, description="Budget for ordering tests")
    vitals_trend: Dict[str, str] = Field(default_factory=dict, description="Trend indicators for vitals e.g. HR: ↑")



# ─── OpenEnv Action Model ──────────────────────────────────────────────

class TriageAction(BaseModel):
    """An action the triage agent can take during an episode."""
    action_type: Literal[
        "order_diagnostic",
        "assign_esi_level",
        "activate_pathway",
        "disposition",
        "request_consult",
        "administer_medication",
        "assign_bed",
        "wait",
    ] = Field(..., description="Type of clinical action")
    patient_id: str = Field(..., description="Which patient this action targets")
    parameter: str = Field(..., description="Test name, ESI level, pathway type, etc.")
    rationale: Optional[str] = Field(None, description="Agent reasoning for logging")


# ─── OpenEnv Observation Model ─────────────────────────────────────────

class TriageObservation(BaseModel):
    """Observation returned to the agent after each step."""
    done: bool = False
    reward: Optional[float] = None
    task_id: str = ""
    task_difficulty: Literal["easy", "medium", "hard"] = "easy"
    step_number: int = 0
    max_steps: int = 15
    patients: List[PatientState] = Field(default_factory=list)
    available_beds: int = 10
    elapsed_minutes: int = 0
    last_action_result: Optional[str] = None
    last_action_error: Optional[str] = None
    reward_components: Optional[Dict[str, float]] = None
    reward_explanation: Optional[str] = None


# ─── OpenEnv State Model ───────────────────────────────────────────────

class TriageState(BaseModel):
    """Internal episode state (includes hidden info like grader scores)."""
    episode_id: Optional[str] = None
    step_count: int = 0
    elapsed_minutes: int = 0
    task_id: str = ""
    episode_history: List[dict] = Field(default_factory=list)
    cumulative_reward: float = 0.0
    esi_assignments: Dict[str, int] = Field(default_factory=dict)
    dispositions: Dict[str, str] = Field(default_factory=dict)
    diagnostics_ordered: List[str] = Field(default_factory=list)
    pathways_activated: List[str] = Field(default_factory=list)


# ─── Grading Result ────────────────────────────────────────────────────

class GradeResult(BaseModel):
    """Result from a deterministic grader."""
    score: float = Field(..., ge=0.0, le=1.0)
    breakdown: Dict[str, float] = Field(default_factory=dict)
    explanation: str = ""


# ─── Task Metadata ─────────────────────────────────────────────────────

class TaskInfo(BaseModel):
    """Metadata for a single task."""
    id: str
    difficulty: Literal["easy", "medium", "hard"]
    description: str
    max_steps: int
    baseline_score: float

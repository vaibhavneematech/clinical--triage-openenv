"""
environment.py — Core ClinicalTriageEnv simulation engine.

Implements the OpenEnv 3-method interface: reset(), step(), state.
Wires together patient generation, reward computation, and grading.
"""

from __future__ import annotations

import uuid
import copy
from typing import Optional, Dict

from clinical_triage_env.models import (
    TriageAction,
    TriageObservation,
    TriageState,
    GradeResult,
    TaskInfo,
    PatientState,
)
from clinical_triage_env.server.patient_generator import (
    generate_patients,
    get_lab_result,
    get_imaging_result,
)
from clinical_triage_env.server.reward import compute_step_reward
from clinical_triage_env.server.graders.stemi_grader import grade_stemi
from clinical_triage_env.server.graders.chest_workup_grader import grade_chest_workup
from clinical_triage_env.server.graders.mci_grader import grade_mci
from clinical_triage_env.server.vitals_engine import update_vitals
from clinical_triage_env.server.time_costs import get_action_time_cost


# ─── Task registry ──────────────────────────────────────────────────────

TASKS = {
    "task_stemi_code": TaskInfo(
        id="task_stemi_code",
        difficulty="easy",
        description="Clear STEMI presentation. Agent must activate cath lab pathway within time window.",
        max_steps=15,
        baseline_score=0.72,
    ),
    "task_chest_pain_workup": TaskInfo(
        id="task_chest_pain_workup",
        difficulty="medium",
        description="Ambiguous chest pain. Agent must navigate differential diagnosis with ordered test sequencing.",
        max_steps=20,
        baseline_score=0.48,
    ),
    "task_mci_surge": TaskInfo(
        id="task_mci_surge",
        difficulty="hard",
        description="Mass casualty: 5 simultaneous patients, 3 beds. Agent must correctly triage under scarcity.",
        max_steps=25,
        baseline_score=0.31,
    ),
}

GRADERS = {
    "task_stemi_code": grade_stemi,
    "task_chest_pain_workup": grade_chest_workup,
    "task_mci_surge": grade_mci,
}


class ClinicalTriageEnvironment:
    """
    Core environment class for ClinicalTriageEnv.

    Simulates an Emergency Department triage scenario where an AI agent
    must assess patients, order diagnostics, assign ESI levels, and
    make disposition decisions.
    """

    def __init__(self):
        self._state = TriageState()
        self._patients: list[PatientState] = []
        self._task_info: Optional[TaskInfo] = None
        self._done = False

    def reset(self, task_id: str = "task_stemi_code", **kwargs) -> TriageObservation:
        """
        Reset the environment for a new episode.

        Args:
            task_id: One of 'task_stemi_code', 'task_chest_pain_workup', 'task_mci_surge'

        Returns:
            Initial TriageObservation with patient data
        """
        if task_id not in TASKS:
            raise ValueError(
                f"Unknown task_id '{task_id}'. Valid: {list(TASKS.keys())}"
            )

        self._task_info = TASKS[task_id]
        self._patients = generate_patients(task_id)
        self._done = False

        # Determine available beds
        if task_id == "task_mci_surge":
            available_beds = 3
        else:
            available_beds = 10

        self._state = TriageState(
            episode_id=kwargs.get("episode_id") or str(uuid.uuid4()),
            step_count=0,
            elapsed_minutes=0,
            task_id=task_id,
            episode_history=[],
            cumulative_reward=0.0,
            esi_assignments={},
            dispositions={},
            diagnostics_ordered=[],
            pathways_activated=[],
        )

        return TriageObservation(
            done=False,
            reward=None,
            task_id=task_id,
            task_difficulty=self._task_info.difficulty,
            step_number=0,
            max_steps=self._task_info.max_steps,
            patients=copy.deepcopy(self._patients),
            available_beds=available_beds,
            last_action_result=f"Episode reset. Task: {task_id}. {len(self._patients)} patient(s) awaiting triage.",
            last_action_error=None,
        )

    def step(self, action: TriageAction) -> TriageObservation:
        """
        Execute one action in the environment.

        Returns:
            TriageObservation with updated state, reward, and done flag
        """
        if self._done:
            return self._make_observation(
                reward=0.0,
                done=True,
                result="Episode already complete.",
                error=None,
                components=None,
                explanation="Episode already done.",
            )

        task_id = self._state.task_id
        self._state.step_count += 1

        # ── Validate and process action ─────────────────────────────
        action_result, action_error = self._process_action(action)

        # ── Record in history ───────────────────────────────────────
        step_record = {
            "step": self._state.step_count,
            "action": {
                "action_type": action.action_type,
                "patient_id": action.patient_id,
                "parameter": action.parameter,
                "rationale": action.rationale,
            },
            "result": action_result,
            "error": action_error,
        }
        self._state.episode_history.append(step_record)

        # ── Compute reward ──────────────────────────────────────────
        reward, components, explanation = compute_step_reward(
            action, self._state, task_id
        )
        self._state.cumulative_reward += reward
        
        # Add reward info to history
        step_record["reward"] = reward
        step_record["reward_components"] = components
        step_record["reward_explanation"] = explanation

        # ── Check termination ───────────────────────────────────────
        done = self._check_done()
        self._done = done

        # ── Update patient state (time passes) ──────────────────────
        action_copy = {
            "action_type": action.action_type,
            "parameter": action.parameter,
        }
        time_elapsed = get_action_time_cost(action_copy)
        self._state.elapsed_minutes += time_elapsed

        for p in self._patients:
            p.time_in_department_minutes += time_elapsed

        # Apply dynamic vitals update
        update_vitals(self._patients, time_elapsed)

        return self._make_observation(
            reward=reward,
            done=done,
            result=action_result,
            error=action_error,
            components=components,
            explanation=explanation,
        )

    @property
    def state(self) -> TriageState:
        """Return current episode state (internal)."""
        return self._state

    # ─────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────

    def _process_action(self, action: TriageAction) -> tuple[str, Optional[str]]:
        """Process an action and return (result_text, error_or_none)."""
        task_id = self._state.task_id
        pid = action.patient_id
        param = action.parameter
        atype = action.action_type

        # Validate patient_id exists
        valid_pids = {p.patient_id for p in self._patients}
        if pid not in valid_pids:
            return "", f"Invalid patient_id '{pid}'. Valid: {valid_pids}"

        patient = next(p for p in self._patients if p.patient_id == pid)

        if atype == "order_diagnostic":
            return self._process_diagnostic(patient, param, task_id)
        elif atype == "assign_esi_level":
            return self._process_esi(pid, param)
        elif atype == "activate_pathway":
            res, err = self._process_pathway(pid, param)
            if not err:
                patient.current_medications.append(f"PATHWAY_{param}")
            return res, err
        elif atype == "disposition":
            return self._process_disposition(pid, param)
        elif atype == "request_consult":
            return f"Consult requested for {pid}: {param}. Specialist notified.", None
        elif atype == "administer_medication":
            patient.current_medications.append(param.lower())
            return f"Patient {pid}: Administered {param}.", None
        elif atype == "assign_bed":
            return f"Patient {pid} assigned to Bed {param}.", None
        elif atype == "wait":
            return self._process_wait(patient, task_id)
        else:
            return "", f"Unknown action_type: {atype}"

    def _process_diagnostic(
        self, patient: PatientState, test_name: str, task_id: str
    ) -> tuple[str, Optional[str]]:
        """Order a diagnostic test."""
        # Check resource budget
        if patient.resource_tokens_remaining <= 0:
            return "", "No resource tokens remaining. Cannot order more tests."

        patient.resource_tokens_remaining -= 1
        self._state.diagnostics_ordered.append(test_name.lower())

        # Check for lab result
        lab = get_lab_result(task_id, test_name)
        if lab:
            patient.available_labs.append(lab)
            if test_name.lower() in patient.pending_labs:
                patient.pending_labs.remove(test_name.lower())
            critical_tag = " ⚠️ CRITICAL" if lab.critical else ""
            return (
                f"Lab result: {lab.name} = {lab.value} {lab.unit} "
                f"(ref: {lab.reference_range}){critical_tag}",
                None,
            )

        # Check for imaging result
        imaging = get_imaging_result(task_id, test_name)
        if imaging:
            patient.imaging_available.append(test_name)
            if test_name in patient.pending_imaging:
                patient.pending_imaging.remove(test_name)
            return f"Imaging result ({test_name}): {imaging}", None

        # Check for medication actions (aspirin, epinephrine, etc.)
        if any(med in test_name.lower() for med in ["aspirin", "epinephrine", "iv_access", "resuscitation"]):
            patient.current_medications.append(test_name.lower())
            return f"Medication/intervention ordered: {test_name}. Administered.", None

        return f"Test '{test_name}' ordered. Results pending.", None

    def _process_esi(self, pid: str, esi_str: str) -> tuple[str, Optional[str]]:
        """Assign ESI level to a patient."""
        try:
            esi = int(esi_str.strip())
            if esi < 1 or esi > 5:
                return "", f"ESI level must be 1-5, got {esi}"
        except ValueError:
            return "", f"ESI parameter must be a number 1-5, got '{esi_str}'"

        self._state.esi_assignments[pid] = esi
        return f"Patient {pid} assigned ESI level {esi}.", None

    def _process_pathway(self, pid: str, pathway: str) -> tuple[str, Optional[str]]:
        """Activate a clinical pathway."""
        self._state.pathways_activated.append(pathway.lower())
        return f"Pathway '{pathway}' activated for patient {pid}.", None

    def _process_disposition(self, pid: str, disposition: str) -> tuple[str, Optional[str]]:
        """Set patient disposition (admit/discharge/transfer)."""
        self._state.dispositions[pid] = disposition.lower()
        return f"Patient {pid} disposition: {disposition}.", None

    def _process_wait(self, patient: PatientState, task_id: str) -> tuple[str, Optional[str]]:
        """Wait for pending results. Some pending labs may resolve."""
        resolved = []
        if patient.pending_labs:
            # Resolve the first pending lab
            lab_name = patient.pending_labs[0]
            lab = get_lab_result(task_id, lab_name)
            if lab:
                patient.available_labs.append(lab)
                resolved.append(lab_name)
            patient.pending_labs = patient.pending_labs[1:]

        if resolved:
            return f"Waited. Results now available: {', '.join(resolved)}.", None
        return "Waited. No new results available.", None

    def _check_done(self) -> bool:
        """Check if episode should terminate."""
        task_id = self._state.task_id
        task_info = self._task_info

        # Max steps reached
        if self._state.step_count >= task_info.max_steps:
            return True

        # For single-patient tasks: done when disposition is set
        if task_id in ("task_stemi_code", "task_chest_pain_workup"):
            if self._state.dispositions:
                return True

        # For MCI: done when all 5 patients have dispositions
        if task_id == "task_mci_surge":
            if len(self._state.dispositions) >= 5:
                return True

        # Safety: catastrophic action or fatal delay terminates episode
        if self._state.episode_history:
            last = self._state.episode_history[-1]
            last_action = last.get("action", {})
            param = last_action.get("parameter", "").lower()
            pid = last_action.get("patient_id", "")
            
            # Check for fatal delay penalty in the last step
            if self._state.cumulative_reward < -5.0:
                return True
            
            # Instant termination for fatal signal from reward engine
            last_reward_components = self._state.episode_history[-1].get("reward_components", {})
            if "fatal_delay" in last_reward_components or "safety_guardrail" in last_reward_components and last_reward_components["safety_guardrail"] <= -10:
                return True

            if last_action.get("action_type") == "disposition":
                # Discharging an ESI-1 patient = terminal
                if "discharge" in param:
                    esi = self._state.esi_assignments.get(pid)
                    if esi == 1:
                        return True
                    if pid in ["P1", "P3"] and self._state.task_id == "task_mci_surge":
                        return True

        return False

    def _make_observation(
        self,
        reward: float,
        done: bool,
        result: Optional[str],
        error: Optional[str],
        components: Optional[dict],
        explanation: Optional[str],
    ) -> TriageObservation:
        """Construct observation from current state."""
        task_id = self._state.task_id

        # Calculate available beds for MCI
        available_beds = 10
        if task_id == "task_mci_surge":
            admits = sum(1 for d in self._state.dispositions.values() if "admit" in d)
            available_beds = max(0, 3 - admits)

        return TriageObservation(
            done=done,
            reward=reward,
            task_id=task_id,
            task_difficulty=self._task_info.difficulty if self._task_info else "easy",
            step_number=self._state.step_count,
            max_steps=self._task_info.max_steps if self._task_info else 15,
            patients=copy.deepcopy(self._patients),
            available_beds=available_beds,
            elapsed_minutes=self._state.elapsed_minutes,
            last_action_result=result,
            last_action_error=error,
            reward_components=components,
            reward_explanation=explanation,
        )

    def get_task_grader_score(self, task_id: str = None) -> GradeResult:
        """Run the deterministic grader on the current episode history."""
        tid = task_id or self._state.task_id
        grader = GRADERS.get(tid)
        if grader is None:
            return GradeResult(score=0.0, explanation=f"No grader for task '{tid}'")
        return grader(self._state.episode_history)

    def get_tasks(self) -> list[TaskInfo]:
        """Return list of all available tasks."""
        return list(TASKS.values())

    def close(self):
        """Cleanup (no-op for this environment)."""
        pass

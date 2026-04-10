from typing import List
from clinical_triage_env.models import PatientState

def update_vitals(patients: List[PatientState], dt_minutes: int) -> None:
    """
    Updates the vitals of all patients based on the elapsed time (dt_minutes)
    and their underlying conditions. Also sets vitals_trend.
    """
    for p in patients:
        # Reset trends for this step
        p.vitals_trend = {
            "HR": "→",
            "BP": "→",
            "SpO2": "→",
            "Temp": "→",
            "RR": "→",
            "GCS": "→"
        }
        
        # Determine condition from complaint or ID
        complaint = p.chief_complaint.lower()
        history = [h.lower() for h in p.medical_history]
        
        # Anaphylaxis (MCI P3)
        if "anaphylaxis" in complaint or "allergic" in complaint:
            epinephrine_given = any("epi" in a for a in p.current_medications)  # Check if agent gave epi?
            # Wait, current_medications are existing. Agent giving meds must be tracked somewhere!
            # Let's say if 'epinephrine' is in medical_history or current_medications, they improved?
            # The agent's actions will be added to the state. We don't have access to episode_history here.
            # But maybe we just deteriorate natively. If they gave med, the env adds it to `current_medications`?
            # We'll need the env to do that. Let's assume meds are added to current_medications.
            
            if not any("epi" in m.lower() for m in p.current_medications):
                # Deteriorate
                drops_of_5min = dt_minutes / 5.0
                if drops_of_5min > 0:
                    bp_drop = int(10 * drops_of_5min)
                    spo2_drop = 2.0 * drops_of_5min
                    
                    if bp_drop > 0:
                        p.vitals.systolic_bp -= bp_drop
                        p.vitals.diastolic_bp -= int(bp_drop * 0.6)
                        p.vitals_trend["BP"] = "↓"
                    if spo2_drop > 0:
                        p.vitals.spo2 -= spo2_drop
                        p.vitals_trend["SpO2"] = "↓"
            else:
                # Improve
                p.vitals.systolic_bp += int(5 * (dt_minutes/5))
                p.vitals.spo2 += 1.0 * (dt_minutes/5)
                p.vitals_trend["BP"] = "↑"
                p.vitals_trend["SpO2"] = "↑"

        # STEMI (Task 1 P1)
        elif "crushing chest pain" in complaint or "stemi" in complaint:
            if not any("cath" in m.lower() for m in p.current_medications): # proxy for cath lab
                intervals_15m = dt_minutes / 15.0
                if intervals_15m > 0:
                    hr_inc = int(5 * intervals_15m)
                    bp_drop = int(8 * intervals_15m)
                    spo2_drop = 1.0 * intervals_15m
                    
                    if hr_inc > 0:
                        p.vitals.heart_rate += hr_inc
                        p.vitals_trend["HR"] = "↑"
                    if bp_drop > 0:
                        p.vitals.systolic_bp -= bp_drop
                        p.vitals.diastolic_bp -= int(bp_drop * 0.5)
                        p.vitals_trend["BP"] = "↓"
                    if spo2_drop > 0:
                        p.vitals.spo2 -= spo2_drop
                        p.vitals_trend["SpO2"] = "↓"
                        
        # Sepsis (Generic rule)
        elif "fever" in complaint and ("sepsis" in complaint or "confused" in complaint):
            intervals_10m = dt_minutes / 10.0
            if intervals_10m > 0:
                p.vitals.heart_rate += int(3 * intervals_10m)
                p.vitals.systolic_bp -= int(5 * intervals_10m)
                p.vitals.temperature += 0.2 * intervals_10m
                p.vitals_trend["HR"] = "↑"
                p.vitals_trend["BP"] = "↓"
                p.vitals_trend["Temp"] = "↑"
                
        # Bounded limits (vital clamp)
        p.vitals.heart_rate = max(0, min(300, p.vitals.heart_rate))
        p.vitals.systolic_bp = max(0, min(300, p.vitals.systolic_bp))
        p.vitals.diastolic_bp = max(0, min(200, p.vitals.diastolic_bp))
        p.vitals.spo2 = max(0.0, min(100.0, p.vitals.spo2))
        p.vitals.temperature = max(30.0, min(43.0, round(p.vitals.temperature, 1)))
        
        # Sync GCS drop if BP critically low or SpO2 low
        if p.vitals.systolic_bp < 60 or p.vitals.spo2 < 80.0:
            p.vitals.gcs = max(3, p.vitals.gcs - 1)
            p.vitals_trend["GCS"] = "↓"

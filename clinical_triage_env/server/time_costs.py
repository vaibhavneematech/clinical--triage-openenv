from typing import Dict, Any

def get_action_time_cost(action: Dict[str, Any]) -> int:
    """
    Returns the time cost in minutes for a given action.
    """
    action_type = action.get("action_type")
    parameter = str(action.get("parameter", "")).lower()

    if action_type == "order_diagnostic":
        if "ct" in parameter or "pa" in parameter:
            return 45
        elif "ekg" in parameter or "ecg" in parameter:
            return 5
        elif "cxr" in parameter or "xray" in parameter or "x-ray" in parameter:
            return 15
        else: # Assumed to be bloodwork or other labs
            return 30
            
    elif action_type == "administer_medication":
        if "epi" in parameter or "epinephrine" in parameter:
            return 1 
        return 5
        
    elif action_type == "activate_pathway":
        return 2
        
    elif action_type == "assign_esi_level":
        return 1
        
    elif action_type == "assign_bed":
        return 2
        
    elif action_type == "disposition":
        return 5
        
    elif action_type == "wait":
        # Wait parameter might specify time, but default to 15
        try:
            return int(parameter)
        except ValueError:
            return 15
            
    elif action_type == "request_consult":
        return 10
        
    return 1 # Default fallback cost

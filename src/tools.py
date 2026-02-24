"""Tool functions for appointment operations.( 4 definitions currently 3 req + bonus )"""

import json
from pathlib import Path
from typing import Optional

DATA_PATH = Path(__file__).parent.parent / "data" / "appointments.json"


def load_data() -> dict:
    """Load appointment data from JSON file."""
    with open(DATA_PATH, "r") as f:
        return json.load(f)


def save_data(data: dict) -> None:
    """Save updated appointment data back to JSON file."""
    with open(DATA_PATH, "w") as f:
        json.dump(data, f, indent=2)


def lookup_appointment(appointment_id: Optional[str] = None,
                       patient_id: Optional[str] = None) -> Optional[dict]:
    """Find an appointment by ID or patient ID."""
    data = load_data()
    for apt in data["appointments"]:
        if appointment_id and apt["id"] == appointment_id:
            return apt
        if patient_id and apt["patient_id"] == patient_id:
            return apt
    return None


def reschedule_appointment(appointment_id: str, new_date: str,
                           new_time: str) -> str:
    """Reschedule an appointment to a new date/time."""
    data = load_data()
    for apt in data["appointments"]:
        if apt["id"] == appointment_id:
            old_date, old_time = apt["date"], apt["time"]
            apt["date"] = new_date
            apt["time"] = new_time
            apt["status"] = "rescheduled"
            save_data(data)
            return (f"Appointment {appointment_id} rescheduled from "
                    f"{old_date} at {old_time} to {new_date} at {new_time}.")
    return f"Appointment {appointment_id} not found."


def cancel_appointment(appointment_id: str) -> str:
    """Cancel an appointment."""
    data = load_data()
    for apt in data["appointments"]:
        if apt["id"] == appointment_id:
            apt["status"] = "cancelled"
            save_data(data)
            return f"Appointment {appointment_id} has been cancelled."
    return f"Appointment {appointment_id} not found."


def get_preparation_instructions(appointment_type: str) -> str:
    """Get preparation instructions for a given appointment type."""
    data = load_data()
    instructions = data.get("preparation_instructions", {})
    if appointment_type in instructions:
        return instructions[appointment_type]
    return f"No preparation instructions found for '{appointment_type}'."
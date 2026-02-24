"""LangGraph node functions for the appointment assistant workflow."""

import uuid
from datetime import datetime
from langchain_openai import ChatOpenAI
from src.config import OPENAI_API_KEY, MODEL_NAME
from src.tools import (
    lookup_appointment,
    reschedule_appointment,
    cancel_appointment,
    get_preparation_instructions,
)
from src.state import AppointmentState


# Initialize the LLM
llm = ChatOpenAI(model=MODEL_NAME, api_key=OPENAI_API_KEY, temperature=0)


# ──────────────────────────────────────────────
# Node 1: Initialize the run
# ──────────────────────────────────────────────
def initialize_run(state: AppointmentState) -> dict:
    """Set up run ID and timestamp for tracing."""
    run_id = f"RUN-{uuid.uuid4().hex[:8].upper()}"
    print(f"\n{'='*60}")
    print(f"  Run ID: {run_id}")
    print(f"  Time:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Input:  {state['user_input']}")
    print(f"{'='*60}")
    return {"run_id": run_id}


# ──────────────────────────────────────────────
# Node 2: Classify the patient's intent
# ──────────────────────────────────────────────
def classify_intent(state: AppointmentState) -> dict:
    """Use the LLM to classify the user's intent."""
    prompt = f"""You are a medical appointment assistant. Classify the following patient message 
into exactly ONE of these categories:

- "reschedule" — patient wants to change their appointment date/time
- "cancel" — patient wants to cancel their appointment
- "prep_info" — patient wants preparation instructions for their appointment
- "emergency" — patient describes an emergency, severe symptoms, or life-threatening situation
- "unknown" — message doesn't fit any of the above

Also extract any identifiers mentioned:
- appointment_id (format: APT-XXXX)
- patient_id (format: P-XXX)
- new_date (format: YYYY-MM-DD)
- new_time (format: HH:MM)

Patient message: "{state['user_input']}"

Respond in EXACTLY this format (no extra text):
intent: <intent>
appointment_id: <id or NONE>
patient_id: <id or NONE>
new_date: <date or NONE>
new_time: <time or NONE>"""

    response = llm.invoke(prompt)
    content = response.content.strip()

    # Parse the response
    parsed = {}
    for line in content.split("\n"):
        if ":" in line:
            key, value = line.split(":", 1)
            value = value.strip()
            if value.upper() == "NONE":
                value = None
            parsed[key.strip()] = value

    intent = parsed.get("intent", "unknown")
    print(f"  [classify_intent] Intent: {intent}")
    print(f"  [classify_intent] Extracted: apt={parsed.get('appointment_id')}, "
          f"patient={parsed.get('patient_id')}, "
          f"date={parsed.get('new_date')}, time={parsed.get('new_time')}")

    return {
        "intent": intent,
        "appointment_id": parsed.get("appointment_id"),
        "patient_id": parsed.get("patient_id"),
        "new_date": parsed.get("new_date"),
        "new_time": parsed.get("new_time"),
    }


# ──────────────────────────────────────────────
# Node 3: Safety check (emergency detection)
# ──────────────────────────────────────────────
def safety_check(state: AppointmentState) -> dict:
    """Check for emergency/risk indicators and escalate if needed."""
    if state.get("intent") == "emergency":
        print("  [safety_check] ⚠ EMERGENCY DETECTED — escalating")
        return {
            "status": "ESCALATE",
            "draft_response": (
                "⚠ EMERGENCY ALERT: Based on your message, this appears to be an urgent "
                "medical situation. Please call 911 or go to your nearest emergency room "
                "immediately. Do not wait for an appointment. If you are in immediate danger, "
                "call emergency services right away."
            ),
            "route_taken": "emergency_escalation",
        }
    print("  [safety_check] ✓ No emergency detected")
    return {}


# ──────────────────────────────────────────────
# Node 4: Validate that we have enough info
# ──────────────────────────────────────────────
def validate_info(state: AppointmentState) -> dict:
    """Check if we have the required information to proceed."""
    intent = state.get("intent")

    if intent == "reschedule":
        if not state.get("appointment_id") and not state.get("patient_id"):
            print("  [validate_info] ✗ Missing appointment/patient ID for reschedule")
            return {
                "status": "NEED_INFO",
                "draft_response": (
                    "I'd be happy to help reschedule your appointment. Could you please "
                    "provide your appointment ID (e.g., APT-1001) or patient ID (e.g., P-201)?"
                ),
                "route_taken": "reschedule_need_info",
            }
        if not state.get("new_date") or not state.get("new_time"):
            # We have the appointment but need the new date/time
            apt = lookup_appointment(
                appointment_id=state.get("appointment_id"),
                patient_id=state.get("patient_id"),
            )
            if apt:
                print("  [validate_info] ✗ Missing new date/time for reschedule")
                return {
                    "status": "NEED_INFO",
                    "draft_response": (
                        f"I found your appointment ({apt['id']}) for {apt['type']} on "
                        f"{apt['date']} at {apt['time']} with {apt['doctor']}. "
                        f"What new date and time would you like to reschedule to?"
                    ),
                    "route_taken": "reschedule_need_datetime",
                }

    elif intent == "cancel":
        if not state.get("appointment_id") and not state.get("patient_id"):
            print("  [validate_info] ✗ Missing appointment/patient ID for cancel")
            return {
                "status": "NEED_INFO",
                "draft_response": (
                    "I can help you cancel your appointment. Could you please provide "
                    "your appointment ID (e.g., APT-1001) or patient ID (e.g., P-201)?"
                ),
                "route_taken": "cancel_need_info",
            }

    elif intent == "prep_info":
        if not state.get("appointment_id") and not state.get("patient_id"):
            print("  [validate_info] ✗ Missing appointment/patient ID for prep info")
            return {
                "status": "NEED_INFO",
                "draft_response": (
                    "I can provide preparation instructions for your appointment. "
                    "Could you please provide your appointment ID (e.g., APT-1001) "
                    "or patient ID (e.g., P-201)?"
                ),
                "route_taken": "prep_need_info",
            }

    print("  [validate_info] ✓ Sufficient information available")
    return {}


# ──────────────────────────────────────────────
# Node 5: Execute the requested action
# ──────────────────────────────────────────────
def execute_action(state: AppointmentState) -> dict:
    """Perform the actual appointment operation."""
    intent = state.get("intent")
    apt_id = state.get("appointment_id")
    patient_id = state.get("patient_id")

    # Look up appointment if needed
    apt = lookup_appointment(appointment_id=apt_id, patient_id=patient_id)
    if not apt:
        print("  [execute_action] ✗ Appointment not found")
        return {
            "status": "NEED_INFO",
            "draft_response": "I couldn't find an appointment matching the information provided. "
                              "Please double-check your appointment ID or patient ID.",
            "action_result": "Appointment not found",
            "route_taken": f"{intent}_not_found",
        }

    if intent == "reschedule":
        result = reschedule_appointment(apt["id"], state["new_date"], state["new_time"])
        print(f"  [execute_action] ✓ Rescheduled: {result}")
        return {
            "action_result": result,
            "route_taken": "reschedule_success",
        }

    elif intent == "cancel":
        result = cancel_appointment(apt["id"])
        print(f"  [execute_action] ✓ Cancelled: {result}")
        return {
            "action_result": result,
            "route_taken": "cancel_success",
        }

    elif intent == "prep_info":
        instructions = get_preparation_instructions(apt["type"])
        print(f"  [execute_action] ✓ Retrieved prep instructions for {apt['type']}")
        return {
            "action_result": instructions,
            "route_taken": "prep_info_success",
        }

    return {"action_result": "No action taken", "route_taken": "no_action"}


# ──────────────────────────────────────────────
# Node 6: Generate a draft response using the LLM
# ──────────────────────────────────────────────
def generate_draft_response(state: AppointmentState) -> dict:
    """Use the LLM to create a polished patient-facing response."""
    prompt = f"""You are a friendly and professional medical appointment assistant.
Generate a clear, helpful response for the patient based on this information:

Patient's request: {state['user_input']}
Intent: {state['intent']}
Action result: {state.get('action_result', 'N/A')}

Guidelines:
- Be warm and professional
- Include all relevant details (dates, times, instructions)
- Do NOT provide any medical or clinical advice
- If the action was successful, confirm what was done
- Keep the response concise but complete

Generate the response:"""

    response = llm.invoke(prompt)
    draft = response.content.strip()
    print(f"  [generate_draft] Draft response generated ({len(draft)} chars)")
    return {"draft_response": draft, "status": "READY"}


# ──────────────────────────────────────────────
# Node 7: Human-in-the-Loop review
# ──────────────────────────────────────────────
def human_review(state: AppointmentState) -> dict:
    """Pause for human review of the draft response."""
    print(f"\n{'─'*60}")
    print("  HUMAN-IN-THE-LOOP REVIEW")
    print(f"{'─'*60}")
    print(f"\n  Draft Response:\n")
    print(f"  {state['draft_response']}\n")
    print(f"{'─'*60}")

    while True:
        action = input("  Action — [A]pprove / [E]dit / [R]eject: ").strip().upper()
        if action in ("A", "APPROVE"):
            print("  [human_review] ✓ Response APPROVED")
            return {
                "hitl_action": "approve",
                "final_response": state["draft_response"],
            }
        elif action in ("E", "EDIT"):
            print("  Enter your edited response (press Enter twice to finish):")
            lines = []
            while True:
                line = input("  ")
                if line == "":
                    break
                lines.append(line)
            edited = "\n".join(lines)
            print("  [human_review] ✎ Response EDITED")
            return {
                "hitl_action": "edit",
                "hitl_edited_response": edited,
                "final_response": edited,
            }
        elif action in ("R", "REJECT"):
            print("  [human_review] ✗ Response REJECTED — escalating")
            return {
                "hitl_action": "reject",
                "status": "ESCALATE",
                "final_response": "This request has been escalated for manual handling.",
            }
        else:
            print("  Please enter A, E, or R.")


# ──────────────────────────────────────────────
# Node 8: Finalize and output
# ──────────────────────────────────────────────
def finalize_output(state: AppointmentState) -> dict:
    """Produce the final output with tracing information."""
    final = state.get("final_response", state.get("draft_response", "No response generated."))
    status = state.get("status", "READY")
    route = state.get("route_taken", "unknown")

    print(f"\n{'='*60}")
    print(f"  FINAL OUTPUT")
    print(f"{'='*60}")
    print(f"  Run ID:   {state.get('run_id', 'N/A')}")
    print(f"  Status:   {status}")
    print(f"  Route:    {route}")
    print(f"  HITL:     {state.get('hitl_action', 'N/A')}")
    print(f"{'─'*60}")
    print(f"  Response to patient:\n")
    print(f"  {final}")
    print(f"\n{'='*60}")

    return {"final_response": final, "status": status}
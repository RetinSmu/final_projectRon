"""Flask web UI for the appointment assistant."""

import uuid
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session
from src.config import OPENAI_API_KEY, MODEL_NAME
from src.middleware import (
    PIIMiddleware,
    ModerationMiddleware,
    ToolCallLimitMiddleware,
    ModelRetryMiddleware,
    LoggingMiddleware,
)
from src.tools import (
    lookup_appointment,
    reschedule_appointment,
    cancel_appointment,
    get_preparation_instructions,
)
from langchain_openai import ChatOpenAI

app = Flask(
    __name__,
    template_folder=str(__import__("pathlib").Path(__file__).parent.parent / "templates"),
    static_folder=str(__import__("pathlib").Path(__file__).parent.parent / "static"),
)
app.secret_key = "mban5510-dev-key"

llm = ChatOpenAI(model=MODEL_NAME, api_key=OPENAI_API_KEY, temperature=0)


def run_workflow(user_input: str) -> dict:
    """Run the appointment assistant workflow and return results.
    
    This mirrors the LangGraph workflow but returns structured data
    for the web UI instead of printing to console.
    """
    run_id = f"RUN-{uuid.uuid4().hex[:8].upper()}"
    logs = []
    
    # Reset middleware
    ToolCallLimitMiddleware.reset()
    LoggingMiddleware.reset()
    LoggingMiddleware.log_node("initialize_run")
    
    logs.append(f"Run ID: {run_id}")
    logs.append(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logs.append(f"Input: {PIIMiddleware.mask_pii(user_input)}")
    
    # Middleware checks
    LoggingMiddleware.log_node("middleware_checks")
    pii_state = {"user_input": user_input}
    PIIMiddleware.process(pii_state)
    logs.append("PII check completed")
    
    mod_result = ModerationMiddleware.process(pii_state)
    if mod_result.get("status") == "ESCALATE":
        logs.append("⚠ MODERATION FLAGGED")
        LoggingMiddleware.log_node("human_review")
        LoggingMiddleware.log_node("finalize_output")
        return {
            "run_id": run_id,
            "status": "ESCALATE",
            "route": "moderation_flagged",
            "draft_response": mod_result["draft_response"],
            "logs": logs,
            "trace": LoggingMiddleware.get_trace_summary(),
            "llm_calls": ToolCallLimitMiddleware.get_count(),
        }
    logs.append("Moderation passed")
    
    # Classify intent
    LoggingMiddleware.log_node("classify_intent")
    classify_prompt = f"""You are a medical appointment assistant. Classify the following patient message 
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

Patient message: "{user_input}"

Respond in EXACTLY this format (no extra text):
intent: <intent>
appointment_id: <id or NONE>
patient_id: <id or NONE>
new_date: <date or NONE>
new_time: <time or NONE>"""

    content = ModelRetryMiddleware.call_with_retry(llm, classify_prompt)
    
    parsed = {}
    for line in content.split("\n"):
        if ":" in line:
            key, value = line.split(":", 1)
            value = value.strip()
            if value.upper() == "NONE":
                value = None
            parsed[key.strip()] = value
    
    intent = parsed.get("intent", "unknown")
    appointment_id = parsed.get("appointment_id")
    patient_id = parsed.get("patient_id")
    new_date = parsed.get("new_date")
    new_time = parsed.get("new_time")
    logs.append(f"Intent: {intent}")
    
    # Safety check
    LoggingMiddleware.log_node("safety_check")
    if intent == "emergency":
        logs.append("⚠ EMERGENCY DETECTED")
        LoggingMiddleware.log_node("human_review")
        LoggingMiddleware.log_node("finalize_output")
        return {
            "run_id": run_id,
            "status": "ESCALATE",
            "route": "emergency_escalation",
            "draft_response": (
                "⚠ EMERGENCY ALERT: Based on your message, this appears to be an urgent "
                "medical situation. Please call 911 or go to your nearest emergency room "
                "immediately. Do not wait for an appointment."
            ),
            "logs": logs,
            "trace": LoggingMiddleware.get_trace_summary(),
            "llm_calls": ToolCallLimitMiddleware.get_count(),
        }
    logs.append("Safety check passed")
    
    # Validate info
    LoggingMiddleware.log_node("validate_info")
    if intent in ("reschedule", "cancel", "prep_info"):
        if not appointment_id and not patient_id:
            logs.append("Missing appointment/patient ID")
            LoggingMiddleware.log_node("human_review")
            LoggingMiddleware.log_node("finalize_output")
            return {
                "run_id": run_id,
                "status": "NEED_INFO",
                "route": f"{intent}_need_info",
                "draft_response": (
                    f"I'd be happy to help with your {intent.replace('_', ' ')} request. "
                    f"Could you please provide your appointment ID (e.g., APT-1001) "
                    f"or patient ID (e.g., P-201)?"
                ),
                "logs": logs,
                "trace": LoggingMiddleware.get_trace_summary(),
                "llm_calls": ToolCallLimitMiddleware.get_count(),
            }
        
        if intent == "reschedule" and (not new_date or not new_time):
            apt = lookup_appointment(appointment_id=appointment_id, patient_id=patient_id)
            if apt:
                logs.append("Missing new date/time for reschedule")
                LoggingMiddleware.log_node("human_review")
                LoggingMiddleware.log_node("finalize_output")
                return {
                    "run_id": run_id,
                    "status": "NEED_INFO",
                    "route": "reschedule_need_datetime",
                    "draft_response": (
                        f"I found your appointment ({apt['id']}) for {apt['type']} on "
                        f"{apt['date']} at {apt['time']} with {apt['doctor']}. "
                        f"What new date and time would you like?"
                    ),
                    "logs": logs,
                    "trace": LoggingMiddleware.get_trace_summary(),
                    "llm_calls": ToolCallLimitMiddleware.get_count(),
                }
    
    logs.append("Validation passed")
    
    # Execute action
    LoggingMiddleware.log_node("execute_action")
    apt = lookup_appointment(appointment_id=appointment_id, patient_id=patient_id)
    
    if not apt and intent in ("reschedule", "cancel", "prep_info"):
        logs.append("Appointment not found")
        LoggingMiddleware.log_node("human_review")
        LoggingMiddleware.log_node("finalize_output")
        return {
            "run_id": run_id,
            "status": "NEED_INFO",
            "route": f"{intent}_not_found",
            "draft_response": "I couldn't find an appointment matching the information provided. "
                              "Please double-check your appointment ID or patient ID.",
            "logs": logs,
            "trace": LoggingMiddleware.get_trace_summary(),
            "llm_calls": ToolCallLimitMiddleware.get_count(),
        }
    
    action_result = ""
    route = "unknown"
    
    if intent == "reschedule" and apt:
        action_result = reschedule_appointment(apt["id"], new_date, new_time)
        route = "reschedule_success"
    elif intent == "cancel" and apt:
        action_result = cancel_appointment(apt["id"])
        route = "cancel_success"
    elif intent == "prep_info" and apt:
        action_result = get_preparation_instructions(apt["type"])
        route = "prep_info_success"
    else:
        action_result = "I can help with rescheduling, cancelling, or preparation instructions for appointments."
        route = "unknown_intent"
    
    logs.append(f"Action: {route}")
    
    # Generate draft response
    LoggingMiddleware.log_node("generate_draft_response")
    draft_prompt = f"""You are a friendly and professional medical appointment assistant.
Generate a clear, helpful response for the patient based on this information:

Patient's request: {user_input}
Intent: {intent}
Action result: {action_result}

Guidelines:
- Be warm and professional
- Include all relevant details (dates, times, instructions)
- Do NOT provide any medical or clinical advice
- Keep the response concise but complete
- Do NOT include placeholder signatures like [Your Name]

Generate the response:"""

    draft = ModelRetryMiddleware.call_with_retry(llm, draft_prompt)
    logs.append("Draft response generated")
    
    LoggingMiddleware.log_node("human_review")
    LoggingMiddleware.log_node("finalize_output")
    
    return {
        "run_id": run_id,
        "status": "READY",
        "route": route,
        "draft_response": draft,
        "logs": logs,
        "trace": LoggingMiddleware.get_trace_summary(),
        "llm_calls": ToolCallLimitMiddleware.get_count(),
    }


@app.route("/")
def index():
    """Render the main page."""
    return render_template("index.html")


@app.route("/api/process", methods=["POST"])
def process_request():
    """Process a patient request and return draft for review."""
    data = request.get_json()
    user_input = data.get("message", "").strip()
    
    if not user_input:
        return jsonify({"error": "No message provided"}), 400
    
    result = run_workflow(user_input)
    return jsonify(result)


@app.route("/api/finalize", methods=["POST"])
def finalize():
    """Finalize the response after human review."""
    data = request.get_json()
    action = data.get("action")  # "approve" or "edit"
    draft = data.get("draft_response", "")
    edited = data.get("edited_response", "")
    run_id = data.get("run_id", "")
    status = data.get("status", "READY")
    route = data.get("route", "")
    
    if action == "approve":
        final_response = draft
        hitl_action = "approve"
    elif action == "edit":
        final_response = edited
        hitl_action = "edit"
    elif action == "reject":
        final_response = "This request has been escalated for manual handling."
        hitl_action = "reject"
        status = "ESCALATE"
    else:
        return jsonify({"error": "Invalid action"}), 400
    
    return jsonify({
        "run_id": run_id,
        "status": status,
        "route": route,
        "hitl_action": hitl_action,
        "final_response": final_response,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
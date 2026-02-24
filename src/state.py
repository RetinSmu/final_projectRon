"""LangGraph state definition for the appointment assistant."""

from typing import TypedDict, Optional


class AppointmentState(TypedDict):
    """State that flows through the LangGraph workflow.
    
    Each key represents a piece of information that nodes can
    read from or write to as the workflow progresses.
    """
    # -- Input --
    user_input: str

    # -- Intent Classification --
    intent: Optional[str]  # "reschedule" | "cancel" | "prep_info" | "emergency" | "unknown"

    # -- Extracted Details --
    patient_id: Optional[str]
    appointment_id: Optional[str]
    new_date: Optional[str]
    new_time: Optional[str]

    # -- Processing --
    action_result: Optional[str]
    draft_response: Optional[str]

    # -- Human-in-the-Loop --
    hitl_action: Optional[str]  # "approve" | "edit"
    hitl_edited_response: Optional[str]

    # -- Output --
    final_response: Optional[str]
    status: Optional[str]  # "READY" | "NEED_INFO" | "ESCALATE"

    # -- Tracing --
    run_id: Optional[str]
    route_taken: Optional[str]
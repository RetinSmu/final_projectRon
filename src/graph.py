"""LangGraph workflow definition for the appointment assistant."""

from langgraph.graph import StateGraph, END
from src.state import AppointmentState
from src.nodes import (
    initialize_run,
    classify_intent,
    safety_check,
    validate_info,
    execute_action,
    generate_draft_response,
    human_review,
    finalize_output,
)


def should_escalate(state: AppointmentState) -> str:
    """Route after safety check: escalate emergencies or continue."""
    if state.get("status") == "ESCALATE":
        return "human_review"
    return "validate_info"


def has_enough_info(state: AppointmentState) -> str:
    """Route after validation: proceed if we have info, otherwise finalize with NEED_INFO."""
    if state.get("status") == "NEED_INFO":
        return "human_review"
    return "execute_action"


def build_graph() -> StateGraph:
    """Build and compile the LangGraph workflow.

    Workflow:
        initialize_run
              │
        classify_intent
              │
        safety_check
              │
        ┌─────┴──────┐
        │ ESCALATE   │ NORMAL
        ▼            ▼
    human_review  validate_info
        │            │
        │       ┌────┴─────┐
        │       │NEED_INFO │ HAS_INFO
        │       ▼          ▼
        │  human_review  execute_action
        │       │          │
        │       │   generate_draft_response
        │       │          │
        │       │     human_review
        │       │          │
        ▼       ▼          ▼
           finalize_output
              │
             END
    """
    workflow = StateGraph(AppointmentState)

    # Add all nodes
    workflow.add_node("initialize_run", initialize_run)
    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("safety_check", safety_check)
    workflow.add_node("validate_info", validate_info)
    workflow.add_node("execute_action", execute_action)
    workflow.add_node("generate_draft_response", generate_draft_response)
    workflow.add_node("human_review", human_review)
    workflow.add_node("finalize_output", finalize_output)

    # Set entry point
    workflow.set_entry_point("initialize_run")

    # Define edges
    workflow.add_edge("initialize_run", "classify_intent")
    workflow.add_edge("classify_intent", "safety_check")

    # Conditional: after safety check
    workflow.add_conditional_edges(
        "safety_check",
        should_escalate,
        {
            "human_review": "human_review",
            "validate_info": "validate_info",
        },
    )

    # Conditional: after validation
    workflow.add_conditional_edges(
        "validate_info",
        has_enough_info,
        {
            "human_review": "human_review",
            "execute_action": "execute_action",
        },
    )

    # Normal flow continues
    workflow.add_edge("execute_action", "generate_draft_response")
    workflow.add_edge("generate_draft_response", "human_review")

    # All paths end through human review → finalize
    workflow.add_edge("human_review", "finalize_output")
    workflow.add_edge("finalize_output", END)

    return workflow.compile()

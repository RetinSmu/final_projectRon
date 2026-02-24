# Appointment Assistant — MBAN 5510 Final Project

An appointment-assistance system built with LangGraph that demonstrates middleware-driven orchestration, safety controls, and Human-in-the-Loop (HITL) review.

**Author:** Ron Etin  
**Built with:** Claude AI (Opus 4.6) assistance via claude.ai  
**Course:** MBAN 5510 — Agentic AI  
**Instructor:** Michael Zhang  

---

## Overview

This system helps patients manage medical appointments through both a CLI and web interface. It uses a LangGraph stateful workflow to classify patient intent, execute actions, apply safety checks through middleware, and route responses through human review before delivery.

### Supported Request Types

- **Reschedule** an appointment to a new date/time
- **Cancel** an appointment
- **Request preparation instructions** (e.g., MRI prep, blood work prep)
- **Emergency detection** — escalates to appropriate services (no clinical advice given)

### Terminal Statuses

| Status      | Meaning                                                  |
|-------------|----------------------------------------------------------|
| `READY`     | Request processed successfully; response is finalized    |
| `NEED_INFO` | Missing information required to complete the request     |
| `ESCALATE`  | Emergency or risk case detected; routed for escalation   |

---

## Setup

### Prerequisites

- Python 3.10+
- OpenAI API key

### Installation
```bash
# Clone the repository
git clone https://github.com/RetinSmu/final_projectRon.git
cd final_projectRon

# Install dependencies
python3 -m pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Then edit .env and add your OpenAI API key
```

### Environment Variables

| Variable         | Description             | Required |
|------------------|-------------------------|----------|
| `OPENAI_API_KEY` | Your OpenAI API key     | Yes      |

> **Note:** Never commit your `.env` file. It is excluded via `.gitignore`.

---

## Usage

### CLI (Command Line Interface)
```bash
python3 -m src.main
```

**Example commands you can type at the prompt:**
```
What prep do I need for appointment APT-1001?
Cancel my appointment APT-1002
I need to reschedule appointment APT-1001 to 2026-03-20 at 15:00
I'm having severe chest pain and difficulty breathing
I want to cancel my appointment
```

**Expected output for each run includes:**
- Run ID (unique identifier)
- Terminal status (READY / NEED_INFO / ESCALATE)
- Route taken (e.g., prep_info_success, emergency_escalation)
- HITL action (approve / edit / reject)
- LLM call count
- Full node trace

Type `quit` to exit.

### Web UI
```bash
python3 -m src.web_app
```

Then open `http://localhost:5000` in your browser. The web UI provides:
- Chat-style interface with example request chips
- Visual status badges (READY, NEED_INFO, ESCALATE)
- Execution trace display with run details
- Human-in-the-Loop panel with Approve / Edit / Reject buttons

---

## Architecture

### Project Structure
```
final_projectRon/
├── .env.example              # Environment variable template
├── .gitignore                # Git ignore rules
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── data/
│   └── appointments.json     # Local mock appointment database
├── src/
│   ├── __init__.py
│   ├── config.py             # Environment and configuration loading
│   ├── state.py              # LangGraph state definition (TypedDict)
│   ├── tools.py              # Appointment operation functions
│   ├── middleware.py          # Middleware components (PII, moderation, limits, retry, logging)
│   ├── nodes.py              # LangGraph node functions
│   ├── graph.py              # LangGraph workflow definition
│   ├── main.py               # CLI entry point
│   └── web_app.py            # Flask web UI
├── templates/
│   └── index.html            # Web UI template
└── tests/
    ├── __init__.py
    └── test_tools.py          # Unit tests (13 tests)
```

### Workflow Design
```
    initialize_run
          │
    run_middleware_checks (PII + Moderation)
          │
    ┌─────┴──────┐
    │ FLAGGED    │ CLEAN
    ▼            ▼
human_review  classify_intent (LLM)
    │              │
    │         safety_check
    │              │
    │         ┌────┴─────┐
    │         │ESCALATE  │NORMAL
    │         ▼          ▼
    │    human_review  validate_info
    │         │          │
    │         │     ┌────┴─────┐
    │         │   NEED_INFO  HAS_INFO
    │         │     ▼          ▼
    │         │  human_review  execute_action
    │         │     │            │
    │         │     │   generate_draft_response (LLM)
    │         │     │            │
    │         │     │       human_review
    │         │     │            │
    ▼         ▼     ▼            ▼
          finalize_output
                │
               END
```

### Middleware Components

| Middleware | Purpose | Integration |
|-----------|---------|-------------|
| **PIIMiddleware** | Detects and masks personally identifiable information (names, patient IDs, phone numbers, emails, SSNs) in log output | Runs as part of `run_middleware_checks` node; masks PII in all logged output via `mask_pii()` |
| **ModerationMiddleware** | Screens user input for threatening, abusive, or harmful content using pattern matching | Runs as part of `run_middleware_checks` node; returns `ESCALATE` status if flagged, bypassing LLM calls entirely |
| **ToolCallLimitMiddleware** | Tracks and enforces a maximum number of LLM calls per run (default: 5) | Wraps every LLM invocation; prevents runaway execution; count displayed in final output |
| **ModelRetryMiddleware** | Retries failed LLM calls with exponential backoff (3 attempts) | Wraps `llm.invoke()` calls in `classify_intent` and `generate_draft_response` nodes |
| **LoggingMiddleware** | Tracks node execution sequence and timing for the execution trace | Called at the start of every node; produces the trace summary shown in final output |

All middleware resets at the start of each run via `initialize_run`.

---

## Human-in-the-Loop (HITL)

The system pauses for human review before any response is sent to the patient.

### HITL Workflow

1. **Draft Generation** — The system generates a draft response (either from the LLM or from a template for escalation/need-info cases)
2. **Human Review** — The reviewer sees the draft and chooses one of:
   - **Approve (A)** — The draft is sent as-is to the patient
   - **Edit (E)** — The reviewer modifies the text, and the edited version becomes the final response
   - **Reject (R)** — The request is escalated for manual handling
3. **Finalization** — The final response is produced along with the HITL action taken

### Where HITL Occurs

HITL review happens on **every path** through the workflow:
- Normal flow (reschedule/cancel/prep) → draft generated by LLM → human review
- Emergency escalation → template response → human review
- Moderation flagged → template response → human review
- Missing info → template response → human review

This ensures no response reaches the patient without human oversight.

---

## Running Tests
```bash
python3 -m tests.test_tools
```

**13 tests** covering:
- Data loading and appointment lookup
- Reschedule and cancel operations
- Preparation instruction retrieval
- PII masking (names, IDs, phone numbers)
- Moderation detection (flagged and clean content)
- Tool call limit enforcement

Expected output:
```
RUNNING TESTS (13 tests)
  ✓ test_load_data passed
  ✓ test_lookup_by_appointment_id passed
  ✓ test_lookup_by_patient_id passed
  ✓ test_lookup_not_found passed
  ✓ test_get_prep_instructions passed
  ✓ test_get_prep_instructions_not_found passed
  ✓ test_cancel_appointment passed
  ✓ test_reschedule_appointment passed
  ✓ test_pii_masking passed
  ✓ test_pii_no_pii passed
  ✓ test_moderation_flagged passed
  ✓ test_moderation_clean passed
  ✓ test_tool_call_limit passed
  RESULTS: 13 passed, 0 failed, 13 total
```

---

## Demo

**LinkedIn Demo Video:** *(link to be added)*

The demo covers:
1. **Normal scenario** — Requesting preparation instructions for an MRI appointment
2. **Escalation scenario** — Emergency detection with immediate escalation
3. **Human-in-the-Loop** — Draft review with approve and edit actions

---

## Design Decisions & Assumptions

- **Mock data:** Appointments are stored in a local JSON file (`data/appointments.json`) to simulate a database. In production, this would connect to a real scheduling system.
- **No clinical advice:** The system does not provide medical advice. Emergency cases are escalated with instructions to contact emergency services.
- **OpenAI GPT-4o:** Used as the primary LLM for intent classification and response generation.
- **Middleware as nodes:** Middleware components are implemented as Python classes and integrated into the LangGraph workflow as a dedicated `run_middleware_checks` node, with retry and limit middleware wrapping LLM calls throughout.
- **Single-turn design:** Each request is processed independently. Multi-turn conversation (e.g., providing missing info in a follow-up message) is not implemented in the current version.
- **PII masking in logs only:** PII is masked in console/log output but preserved in the actual workflow state for correct processing.

---

## License

This project is for academic purposes (MBAN 5510, Sobey School of Business, Saint Mary's University).
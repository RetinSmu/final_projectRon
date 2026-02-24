An appointment-assistance system built with LangGraph that demonstrates middleware-driven orchestration, safety controls, and Human-in-the-Loop (HITL) review.

**Author:** Ron Etin  
**Built with:** Claude AI (Opus 4.6) assistance via claude.ai  
**Course:** MBAN 5510 — Agentic AI  
**Instructor:** Michael Zhang  

---

## Overview

This system helps patients manage medical appointments through a CLI interface. It uses a LangGraph stateful workflow to classify patient intent, execute actions, apply safety checks, and route responses through human review before delivery.

### Supported Request Types

- **Reschedule** an appointment to a new date/time
- **Cancel** an appointment
- **Request preparation instructions** (e.g., MRI prep, blood work prep)
- **Emergency detection** — escalates to appropriate services (no clinical advice given)

### Terminal Statuses

Each run ends with one of these statuses:

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

*(CLI entry point — coming soon)*

---

## Architecture (not completed)

### Project Structure
```
final_projectRon/
├── .env.example          # Environment variable template
├── .gitignore            # Git ignore rules (Python)
├── README.md             # This file
├── requirements.txt      # Python dependencies
├── data/
│   └── appointments.json # Local mock appointment database
├── src/
│   ├── __init__.py
│   ├── config.py         # Environment and configuration loading
│   ├── state.py          # LangGraph state definition
│   ├── tools.py          # Appointment operation functions
│   ├── nodes.py          # LangGraph node functions (coming next)
│   ├── graph.py          # LangGraph workflow definition (coming next)
│   └── main.py           # CLI entry point (coming next)
└── tests/
    └── __init__.py
```

### Workflow Design

*(High-level architecture diagram and node descriptions will be added as the workflow is built.)*

### Middleware Components

*(Documentation of which middleware components are used and how they integrate with LangGraph state and routing will be added here.)*

---

## Human-in-the-Loop (HITL)

*(Description of the HITL workflow — how draft responses are reviewed, approved, or edited before being finalized — will be added here.)*

---

## Demo

*(LinkedIn demo video link will be added here.)*

---

## Design Decisions & Assumptions

- **Mock data:** Appointments are stored in a local JSON file (`data/appointments.json`) to simulate a database.
- **No clinical advice:** The system does not provide medical advice. Emergency cases are escalated with instructions to contact appropriate services.
- **OpenAI GPT-4o:** Used as the primary LLM for intent classification and response generation.

---

## License

This project is for academic purposes (MBAN 5510, Sobey School of Business, Saint Mary's University, MBAN).


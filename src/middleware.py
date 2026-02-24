"""Middleware components for the appointment assistant workflow.

This module implements middleware-style processing layers that wrap
the core LangGraph workflow. Each middleware function processes state
before or after node execution, providing safety, governance, and
operational controls.

Middleware Components Used:
─────────────────────────
1. PIIMiddleware        — Masks sensitive data (patient IDs, names) in logs
2. ModerationMiddleware — Screens input for inappropriate/harmful content
3. ToolCallLimitMiddleware — Limits the number of tool/LLM calls per run
4. ModelRetryMiddleware — Retries LLM calls on transient failures
5. HumanInTheLoopMiddleware — Integrated via the human_review node in graph.py
"""

import re
import time
from functools import wraps
from src.state import AppointmentState


# ──────────────────────────────────────────────
# 1. PII Middleware — Mask sensitive data in logs
# ──────────────────────────────────────────────
class PIIMiddleware:
    """Detects and masks personally identifiable information in log output.
    
    Masks:
    - Patient names
    - Patient IDs (P-XXX format)
    - Phone numbers
    - Email addresses
    """

    # Patterns to detect and mask
    PII_PATTERNS = {
        "patient_id": (r"P-\d{3,}", "P-***"),
        "phone": (r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "***-***-****"),
        "email": (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "***@***.***"),
        "ssn": (r"\b\d{3}-\d{2}-\d{4}\b", "***-**-****"),
    }

    # Known patient names from our data (in production, this would be dynamic)
    KNOWN_NAMES = [
        "Sarah Johnson", "James Wilson", "Maria Garcia", "Robert Chen"
    ]

    @classmethod
    def mask_pii(cls, text: str) -> str:
        """Replace PII patterns with masked versions."""
        masked = text
        # Mask known patient names
        for name in cls.KNOWN_NAMES:
            if name in masked:
                parts = name.split()
                masked_name = f"{parts[0][0]}. {parts[-1][0]}."
                masked = masked.replace(name, masked_name)
        # Mask patterns
        for pattern_name, (pattern, replacement) in cls.PII_PATTERNS.items():
            masked = re.sub(pattern, replacement, masked)
        return masked

    @classmethod
    def process(cls, state: AppointmentState) -> dict:
        """Log PII check results (masking is applied to log output only)."""
        user_input = state.get("user_input", "")
        pii_found = []

        for pattern_name, (pattern, _) in cls.PII_PATTERNS.items():
            if re.search(pattern, user_input):
                pii_found.append(pattern_name)

        for name in cls.KNOWN_NAMES:
            if name.lower() in user_input.lower():
                pii_found.append("patient_name")
                break

        if pii_found:
            print(f"  [PII Middleware] ⚠ PII detected: {', '.join(pii_found)} — masking in logs")
        else:
            print(f"  [PII Middleware] ✓ No PII detected in input")

        return {}


# ──────────────────────────────────────────────
# 2. Moderation Middleware — Content safety screening
# ──────────────────────────────────────────────
class ModerationMiddleware:
    """Screens user input for inappropriate, abusive, or harmful content.
    
    If flagged content is detected, the request is escalated rather
    than processed normally.
    """

    # Keywords/patterns that trigger moderation
    FLAGGED_PATTERNS = [
        r"\b(threat|threaten|kill|harm|attack|bomb)\b",
        r"\b(abuse|harass)\b",
    ]

    PROFANITY_PATTERNS = [
        # Basic profanity filter (kept minimal for academic context)
        r"\b(damn|hell|crap)\b",
    ]

    @classmethod
    def process(cls, state: AppointmentState) -> dict:
        """Screen input for moderation flags."""
        user_input = state.get("user_input", "").lower()

        # Check for severe flags
        for pattern in cls.FLAGGED_PATTERNS:
            if re.search(pattern, user_input):
                print(f"  [Moderation Middleware] ⚠ FLAGGED — content safety concern detected")
                return {
                    "status": "ESCALATE",
                    "draft_response": (
                        "Your message has been flagged for review. A staff member will "
                        "follow up with you directly. If you are in an emergency, "
                        "please call 911 immediately."
                    ),
                    "route_taken": "moderation_flagged",
                }

        # Check for profanity (log but don't block)
        for pattern in cls.PROFANITY_PATTERNS:
            if re.search(pattern, user_input):
                print(f"  [Moderation Middleware] ⚡ Mild language detected — proceeding with note")
                break
        else:
            print(f"  [Moderation Middleware] ✓ Content passes moderation")

        return {}


# ──────────────────────────────────────────────
# 3. Tool Call Limit Middleware
# ──────────────────────────────────────────────
class ToolCallLimitMiddleware:
    """Tracks and limits the number of tool/LLM calls per run.
    
    Prevents runaway execution by enforcing a maximum number of
    LLM invocations per single workflow run.
    """

    MAX_LLM_CALLS = 5
    _call_count = 0

    @classmethod
    def reset(cls):
        """Reset the call counter for a new run."""
        cls._call_count = 0

    @classmethod
    def increment(cls) -> bool:
        """Increment the call counter. Returns True if within limits."""
        cls._call_count += 1
        within_limit = cls._call_count <= cls.MAX_LLM_CALLS
        if not within_limit:
            print(f"  [ToolCallLimit Middleware] ⚠ LIMIT REACHED ({cls.MAX_LLM_CALLS} calls)")
        else:
            print(f"  [ToolCallLimit Middleware] ✓ LLM call {cls._call_count}/{cls.MAX_LLM_CALLS}")
        return within_limit

    @classmethod
    def get_count(cls) -> int:
        """Return current call count."""
        return cls._call_count


# ──────────────────────────────────────────────
# 4. Model Retry Middleware
# ──────────────────────────────────────────────
class ModelRetryMiddleware:
    """Retries LLM calls on transient failures with exponential backoff.
    
    Wraps LLM invocations to handle temporary API errors (rate limits,
    timeouts, server errors) gracefully.
    """

    MAX_RETRIES = 3
    BASE_DELAY = 1  # seconds

    @classmethod
    def call_with_retry(cls, llm, prompt: str) -> str:
        """Invoke the LLM with retry logic."""
        for attempt in range(1, cls.MAX_RETRIES + 1):
            try:
                # Track call count via ToolCallLimitMiddleware
                if not ToolCallLimitMiddleware.increment():
                    return "Error: LLM call limit exceeded for this run."

                response = llm.invoke(prompt)
                if attempt > 1:
                    print(f"  [ModelRetry Middleware] ✓ Succeeded on attempt {attempt}")
                return response.content.strip()

            except Exception as e:
                delay = cls.BASE_DELAY * (2 ** (attempt - 1))
                if attempt < cls.MAX_RETRIES:
                    print(f"  [ModelRetry Middleware] ⚠ Attempt {attempt} failed: {e}")
                    print(f"  [ModelRetry Middleware] Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    print(f"  [ModelRetry Middleware] ✗ All {cls.MAX_RETRIES} attempts failed: {e}")
                    return f"Error: Unable to get LLM response after {cls.MAX_RETRIES} attempts."


# ──────────────────────────────────────────────
# 5. Logging Middleware — Structured run logging
# ──────────────────────────────────────────────
class LoggingMiddleware:
    """Provides structured logging for each node execution.
    
    Tracks the sequence of nodes visited and timing information
    for the execution trace.
    """

    _node_trace = []
    _start_time = None

    @classmethod
    def reset(cls):
        """Reset for a new run."""
        cls._node_trace = []
        cls._start_time = time.time()

    @classmethod
    def log_node(cls, node_name: str):
        """Record a node visit."""
        elapsed = time.time() - cls._start_time if cls._start_time else 0
        cls._node_trace.append({
            "node": node_name,
            "elapsed_seconds": round(elapsed, 2),
        })

    @classmethod
    def get_trace(cls) -> list:
        """Return the full node trace."""
        return cls._node_trace

    @classmethod
    def get_trace_summary(cls) -> str:
        """Return a concise trace string."""
        return " → ".join(entry["node"] for entry in cls._node_trace)
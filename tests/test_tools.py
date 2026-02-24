"""Tests for the appointment assistant tools and middleware."""

import json
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tools import (
    load_data,
    lookup_appointment,
    reschedule_appointment,
    cancel_appointment,
    get_preparation_instructions,
)
from src.middleware import (
    PIIMiddleware,
    ModerationMiddleware,
    ToolCallLimitMiddleware,
)


# ──────────────────────────────────────────────
# Tool Tests
# ──────────────────────────────────────────────
def test_load_data():
    """Test that appointment data loads correctly."""
    data = load_data()
    assert "appointments" in data, "Missing 'appointments' key"
    assert "preparation_instructions" in data, "Missing 'preparation_instructions' key"
    assert len(data["appointments"]) > 0, "No appointments in data"
    print("  ✓ test_load_data passed")


def test_lookup_by_appointment_id():
    """Test looking up an appointment by ID."""
    apt = lookup_appointment(appointment_id="APT-1001")
    assert apt is not None, "APT-1001 not found"
    assert apt["patient_name"] == "Sarah Johnson", f"Wrong patient: {apt['patient_name']}"
    assert apt["type"] == "MRI Scan", f"Wrong type: {apt['type']}"
    print("  ✓ test_lookup_by_appointment_id passed")


def test_lookup_by_patient_id():
    """Test looking up an appointment by patient ID."""
    apt = lookup_appointment(patient_id="P-202")
    assert apt is not None, "P-202 not found"
    assert apt["id"] == "APT-1002", f"Wrong appointment: {apt['id']}"
    print("  ✓ test_lookup_by_patient_id passed")


def test_lookup_not_found():
    """Test that a missing appointment returns None."""
    apt = lookup_appointment(appointment_id="APT-9999")
    assert apt is None, "Should return None for missing appointment"
    print("  ✓ test_lookup_not_found passed")


def test_get_prep_instructions():
    """Test retrieving preparation instructions."""
    instructions = get_preparation_instructions("MRI Scan")
    assert "metal" in instructions.lower(), "MRI instructions should mention metal"
    assert "eat" in instructions.lower(), "MRI instructions should mention eating"
    print("  ✓ test_get_prep_instructions passed")


def test_get_prep_instructions_not_found():
    """Test prep instructions for unknown type."""
    result = get_preparation_instructions("Brain Surgery")
    assert "no preparation instructions" in result.lower(), "Should return not-found message"
    print("  ✓ test_get_prep_instructions_not_found passed")


def test_cancel_appointment():
    """Test cancelling an appointment (then restore it)."""
    # Cancel
    result = cancel_appointment("APT-1004")
    assert "cancelled" in result.lower(), f"Unexpected result: {result}"

    # Verify status changed
    apt = lookup_appointment(appointment_id="APT-1004")
    assert apt["status"] == "cancelled", "Status should be cancelled"

    # Restore for other tests
    data = load_data()
    for a in data["appointments"]:
        if a["id"] == "APT-1004":
            a["status"] = "confirmed"
    from src.tools import save_data
    save_data(data)
    print("  ✓ test_cancel_appointment passed")


def test_reschedule_appointment():
    """Test rescheduling an appointment (then restore it)."""
    # Save original
    original = lookup_appointment(appointment_id="APT-1004")
    orig_date, orig_time = original["date"], original["time"]

    # Reschedule
    result = reschedule_appointment("APT-1004", "2026-04-01", "09:00")
    assert "rescheduled" in result.lower(), f"Unexpected result: {result}"

    # Verify
    apt = lookup_appointment(appointment_id="APT-1004")
    assert apt["date"] == "2026-04-01", "Date should be updated"
    assert apt["time"] == "09:00", "Time should be updated"

    # Restore
    data = load_data()
    for a in data["appointments"]:
        if a["id"] == "APT-1004":
            a["date"] = orig_date
            a["time"] = orig_time
            a["status"] = "confirmed"
    from src.tools import save_data
    save_data(data)
    print("  ✓ test_reschedule_appointment passed")


# ──────────────────────────────────────────────
# Middleware Tests
# ──────────────────────────────────────────────
def test_pii_masking():
    """Test PII middleware masks sensitive data."""
    text = "Patient Sarah Johnson with ID P-201 called from 555-123-4567"
    masked = PIIMiddleware.mask_pii(text)
    assert "Sarah Johnson" not in masked, "Name should be masked"
    assert "P-201" not in masked, "Patient ID should be masked"
    assert "555-123-4567" not in masked, "Phone should be masked"
    print("  ✓ test_pii_masking passed")


def test_pii_no_pii():
    """Test PII middleware with clean text."""
    text = "I need to reschedule my appointment"
    masked = PIIMiddleware.mask_pii(text)
    assert masked == text, "Clean text should not be modified"
    print("  ✓ test_pii_no_pii passed")


def test_moderation_flagged():
    """Test moderation catches threatening content."""
    state = {"user_input": "I want to threaten someone"}
    result = ModerationMiddleware.process(state)
    assert result.get("status") == "ESCALATE", "Should escalate threats"
    print("  ✓ test_moderation_flagged passed")


def test_moderation_clean():
    """Test moderation passes clean content."""
    state = {"user_input": "I need to cancel my appointment please"}
    result = ModerationMiddleware.process(state)
    assert result.get("status") != "ESCALATE", "Clean input should pass"
    print("  ✓ test_moderation_clean passed")


def test_tool_call_limit():
    """Test tool call limit tracking."""
    ToolCallLimitMiddleware.reset()
    assert ToolCallLimitMiddleware.get_count() == 0, "Should start at 0"

    for i in range(5):
        assert ToolCallLimitMiddleware.increment() == True, f"Call {i+1} should be within limit"

    assert ToolCallLimitMiddleware.increment() == False, "Call 6 should exceed limit"
    ToolCallLimitMiddleware.reset()
    print("  ✓ test_tool_call_limit passed")


# ──────────────────────────────────────────────
# Run all tests
# ──────────────────────────────────────────────
def run_all_tests():
    """Execute all tests and report results."""
    tests = [
        test_load_data,
        test_lookup_by_appointment_id,
        test_lookup_by_patient_id,
        test_lookup_not_found,
        test_get_prep_instructions,
        test_get_prep_instructions_not_found,
        test_cancel_appointment,
        test_reschedule_appointment,
        test_pii_masking,
        test_pii_no_pii,
        test_moderation_flagged,
        test_moderation_clean,
        test_tool_call_limit,
    ]

    print(f"\n{'='*60}")
    print(f"  RUNNING TESTS ({len(tests)} tests)")
    print(f"{'='*60}\n")

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ {test.__name__} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ {test.__name__} ERROR: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"  RESULTS: {passed} passed, {failed} failed, {len(tests)} total")
    print(f"{'='*60}\n")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
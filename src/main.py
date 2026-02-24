"""CLI entry point for the appointment assistant."""

from src.graph import build_graph


def main():
    """Run the appointment assistant CLI."""
    print("\n" + "=" * 60)
    print("  APPOINTMENT ASSISTANT — MBAN 5510")
    print("=" * 60)
    print("  Type your request below. Type 'quit' to exit.\n")
    print("  Example requests:")
    print("    - I need to reschedule appointment APT-1001 to 2026-03-15 at 14:00")
    print("    - Cancel my appointment APT-1002")
    print("    - What prep do I need for appointment APT-1001?")
    print("    - I'm having severe chest pain and difficulty breathing")
    print("=" * 60)

    # Build the graph once
    graph = build_graph()

    while True:
        print()
        user_input = input("  Patient: ").strip()

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("\n  Goodbye!\n")
            break

        # Create initial state
        initial_state = {
            "user_input": user_input,
            "intent": None,
            "patient_id": None,
            "appointment_id": None,
            "new_date": None,
            "new_time": None,
            "action_result": None,
            "draft_response": None,
            "hitl_action": None,
            "hitl_edited_response": None,
            "final_response": None,
            "status": None,
            "run_id": None,
            "route_taken": None,
        }

        try:
            # Run the workflow
            result = graph.invoke(initial_state)
        except KeyboardInterrupt:
            print("\n  Run interrupted.")
        except Exception as e:
            print(f"\n  ✗ Error during execution: {e}")


if __name__ == "__main__":
    main()
import argparse
import sys
import uuid
from agent.orchestrator import AsterRowAgent


def run_cli(debug: bool = False):
    print("=" * 65)
    print("  Aster & Row AI Customer Support Agent (v2)")
    print("  Type your question or order ID below.")
    print("  Commands: '/reset' to clear conversation, '/exit' to quit.")
    print("=" * 65)

    agent = AsterRowAgent()
    session_id = str(uuid.uuid4())[:8]
    print(f"Session initialized: {session_id}\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Thank you for contacting Aster & Row Support!")
            break

        if not user_input:
            continue

        if user_input.lower() in ["/exit", "/quit", "exit", "quit"]:
            print("Exiting Aster & Row Support. Goodbye!")
            break

        if user_input.lower() == "/reset":
            agent.memory.clear(session_id)
            session_id = str(uuid.uuid4())[:8]
            print(f"\n[Conversation reset. New session: {session_id}]\n")
            continue

        response = agent.chat(session_id, user_input)

        print("\n" + "-" * 40)
        print(f"Aster & Row Agent:\n{response.response}")
        print("-" * 40)

        if response.sources:
            print(f"Sources cited: {', '.join(response.sources)}")

        if response.handoff:
            print("[NOTICE: Human support handoff recommended for this inquiry]")

        if debug:
            print("\n[DEBUG TRACE]")
            print(f"Tool called: {response.tool_called}")
            if response.tool_result:
                print(f"Sanitized tool result: {response.tool_result}")
            print(f"Debug info: {response.debug_info}")

        print()


def main():
    parser = argparse.ArgumentParser(description="Aster & Row Support Agent CLI")
    parser.add_argument("--debug", action="store_true", help="Print debug traces and retrieval scores")
    args = parser.parse_args()
    run_cli(debug=args.debug)


if __name__ == "__main__":
    main()

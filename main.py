"""
Main Application: CLI entry point for AI Memory Agent
"""
import os
from dotenv import load_dotenv
from agent_core import AIAgent

load_dotenv()


def run_cli():
    """Run interactive CLI chat."""
    print("=" * 50)
    print("🧠 AI Memory Agent — CLI Mode")
    print("=" * 50)
    print("Commands:")
    print("  'voice'   — toggle voice mode")
    print("  'memory'  — show what I remember about you")
    print("  'clear'   — clear this session's memory")
    print("  'wipe'    — delete ALL your long-term memories")
    print("  'quit'    — exit\n")

    agent = AIAgent()
    voice = None
    user_id = input("Enter your user ID (or press Enter for 'user_001'): ").strip() or "user_001"
    session_id = f"cli_session_{user_id}"

    print(f"\n✅ Loaded as user: {user_id}\n")

    while True:
        try:
            user_input = input("You: ").strip()

            # ── Commands ──
            if not user_input:
                if voice:
                    user_input = voice.record_audio()
                    if not user_input:
                        continue
                else:
                    continue

            cmd = user_input.lower()

            if cmd in ("quit", "exit", "q"):
                break

            elif cmd == "clear":
                agent.memory.clear_short_term(session_id)
                print("🗑️  Session memory cleared.\n")
                continue

            elif cmd == "wipe":
                confirm = input("⚠️  Delete ALL long-term memories? (yes/no): ").strip().lower()
                if confirm == "yes":
                    agent.memory.delete_user_memories(user_id)
                    agent.memory.clear_short_term(session_id)
                    print("💣 All memories wiped.\n")
                else:
                    print("Cancelled.\n")
                continue

            elif cmd == "memory":
                print("\n" + agent.get_memory_summary(user_id) + "\n")
                continue

            elif cmd == "voice":
                if voice is None:
                    try:
                        from voice_processor import VoiceProcessor
                        voice = VoiceProcessor(model_size="tiny")
                        print("🎙️  Voice mode ON — press Enter to speak.\n")
                    except Exception as e:
                        print(f"❌ Voice error: {e}\n")
                else:
                    voice = None
                    print("🔇 Voice mode OFF.\n")
                continue

            # ── Stats shortcut ──
            elif cmd == "stats":
                stats = agent.memory.get_memory_stats(user_id)
                print(f"📊 Episodic: {stats['episodic']}  |  Semantic: {stats['semantic']}\n")
                continue

            # ── Chat ──
            print("🤖 Thinking...")
            response = agent.chat(user_input, user_id, session_id)
            print(f"Agent: {response}\n")

            if voice:
                voice.speak(response)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}\n")

    print("\n👋 Goodbye!")


if __name__ == "__main__":
    run_cli()

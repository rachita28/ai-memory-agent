"""
Main Application: CLI entry point
"""
import os
from dotenv import load_dotenv
from agent_core import AIAgent

load_dotenv()

def run_cli():
    """Run interactive CLI"""
    print("🤖 AI Agent with Memory")
    print("Commands: 'voice' (toggle voice), 'quit' (exit), 'clear' (clear memory)\n")
    
    agent = AIAgent()
    voice = None
    user_id = "user_001"
    session_id = "session_001"
    
    while True:
        try:
            user_input = input("You: ")
            
            # Commands - CHECK THESE FIRST
            if user_input.lower().strip() in ['quit', 'exit', 'q']:
                break
            elif user_input.lower().strip() == 'clear':
                agent.memory.clear_short_term(session_id)
                print("🗑️ Session cleared!\n")
                continue
            elif user_input.lower().strip() == 'voice':
                if voice is None:
                    try:
                        from voice_processor import VoiceProcessor
                        voice = VoiceProcessor(model_size="tiny")
                        print("🎙️ Voice mode ON - Press Enter to speak\n")
                    except Exception as e:
                        print(f"❌ Voice error: {e}")
                else:
                    voice = None
                    print("🔇 Voice mode OFF\n")
                continue
            elif user_input.strip() == "" and voice:
                user_input = voice.record_audio()
            elif not user_input.strip():
                continue
            
            # Process with AI
            print("🤖 Thinking...")
            response = agent.chat(user_input, user_id, session_id)
            
            print(f"Agent: {response}\n")
            
            if voice:
                voice.speak(response)
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n👋 Goodbye!")

if __name__ == "__main__":
    run_cli()
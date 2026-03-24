"""
Streamlit Web Interface for AI Agent with Memory
"""
import streamlit as st
from agent_core import AIAgent
from voice_processor import VoiceProcessor
import time

# Page config
st.set_page_config(
    page_title="AI Agent with Memory",
    page_icon="🧠",
    layout="wide"
)

# Initialize
if 'agent' not in st.session_state:
    st.session_state.agent = AIAgent()
    st.session_state.messages = []
    st.session_state.voice = None
    st.session_state.user_id = "user_001"
    st.session_state.session_id = "session_001"
    st.session_state.listening = False
    st.session_state.voice_text = None

agent = st.session_state.agent

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    st.session_state.user_id = st.text_input("User ID", value=st.session_state.user_id)
    st.session_state.session_id = st.text_input("Session ID", value=st.session_state.session_id)
    
    st.divider()
    
    if st.button("🎙️ Toggle Voice Mode"):
        if st.session_state.voice is None:
            with st.spinner("Loading voice..."):
                try:
                    st.session_state.voice = VoiceProcessor(model_size="tiny")
                    st.success("Voice mode ON!")
                except Exception as e:
                    st.error(f"Voice error: {e}")
                    st.session_state.voice = None
        else:
            st.session_state.voice = None
            st.info("Voice mode OFF")
    
    if st.button("🗑️ Clear Session Memory"):
        st.session_state.messages = []
        agent.memory.clear_short_term(st.session_state.session_id)
        st.success("Memory cleared!")
    
    st.divider()
    st.subheader("📊 Memory Stats")
    try:
        episodic_count = agent.memory.episodic_collection.count()
        semantic_count = agent.memory.semantic_collection.count()
        st.metric("Episodic Memories", episodic_count)
        st.metric("Semantic Memories", semantic_count)
    except:
        st.write("Memory stats unavailable")

# Main chat
st.title("🧠 AI Agent with Long-Term Memory")
st.caption("Your personal AI that remembers everything")

# Display messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle voice text if available
if st.session_state.voice_text:
    voice_input = st.session_state.voice_text
    st.session_state.voice_text = None
    
    st.session_state.messages.append({"role": "user", "content": voice_input})
    
    with st.spinner("Thinking..."):
        response = agent.chat(
            voice_input,
            st.session_state.user_id,
            st.session_state.session_id
        )
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    if st.session_state.voice:
        st.session_state.voice.speak(response)
    
    st.rerun()

# Chat input
if prompt := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = agent.chat(
                prompt, 
                st.session_state.user_id, 
                st.session_state.session_id
            )
        st.markdown(response)
        
        if st.session_state.voice:
            st.session_state.voice.speak(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})

# Voice input section
if st.session_state.voice:
    st.divider()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("🎤 Start Listening", use_container_width=True, disabled=st.session_state.listening):
            st.session_state.listening = True
            st.rerun()
    
    with col2:
        if st.button("⏹️ Stop Listening", use_container_width=True, disabled=not st.session_state.listening):
            st.session_state.listening = False
            st.rerun()
    
    if st.session_state.listening:
        with st.spinner("🎙️ Listening... Speak now!"):
            try:
                voice_text = st.session_state.voice.record_audio(duration=5)
                st.success(f"✅ Heard: '{voice_text}'")
                
                st.session_state.voice_text = voice_text
                st.session_state.listening = False
                
                time.sleep(0.5)
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Voice error: {str(e)}")
                st.session_state.listening = False
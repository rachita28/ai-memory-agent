"""
Streamlit Web Interface for AI Agent with Memory
"""
import os
import uuid
import streamlit as st
from agent_core import AIAgent

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

IS_CLOUD = os.environ.get("STREAMLIT_SHARING_MODE") == "streamlit-community"

st.set_page_config(
    page_title="AI Memory Agent",
    page_icon="🧠",
    layout="wide"
)

# ─────────────────────────────────────────────
# Session state initialization
# ─────────────────────────────────────────────

if "agent" not in st.session_state:
    st.session_state.agent = AIAgent()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "voice" not in st.session_state:
    st.session_state.voice = None

if "user_id" not in st.session_state:
    st.session_state.user_id = "user_001"

if "session_id" not in st.session_state:
    # Unique session per browser tab by default
    st.session_state.session_id = f"session_{uuid.uuid4().hex[:8]}"

if "listening" not in st.session_state:
    st.session_state.listening = False

if "voice_text" not in st.session_state:
    st.session_state.voice_text = None

agent: AIAgent = st.session_state.agent

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Settings")

    new_user_id = st.text_input("👤 User ID", value=st.session_state.user_id)
    if new_user_id != st.session_state.user_id:
        st.session_state.user_id = new_user_id
        st.session_state.messages = []  # Reset chat on user switch

    new_session_id = st.text_input("🔑 Session ID", value=st.session_state.session_id)
    if new_session_id != st.session_state.session_id:
        st.session_state.session_id = new_session_id
        st.session_state.messages = []

    st.divider()

    # Voice toggle (local only)
    if IS_CLOUD:
        st.info("🔇 Voice mode disabled in cloud deployment")
    else:
        if st.button("🎙️ Toggle Voice Mode"):
            if st.session_state.voice is None:
                with st.spinner("Loading voice model..."):
                    try:
                        from voice_processor import VoiceProcessor
                        st.session_state.voice = VoiceProcessor(model_size="tiny")
                        st.success("🎙️ Voice mode ON!")
                    except Exception as e:
                        st.error(f"Voice error: {e}")
                        st.session_state.voice = None
            else:
                st.session_state.voice = None
                st.info("🔇 Voice mode OFF")

    st.divider()

    # Memory actions
    st.subheader("🧠 Memory Controls")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Session", use_container_width=True):
            st.session_state.messages = []
            agent.memory.clear_short_term(st.session_state.session_id)
            st.success("Session cleared!")

    with col2:
        if st.button("💣 Wipe All Memory", use_container_width=True):
            agent.memory.delete_user_memories(st.session_state.user_id)
            agent.memory.clear_short_term(st.session_state.session_id)
            st.session_state.messages = []
            st.success("All memory wiped!")

    st.divider()

    # Memory stats
    st.subheader("📊 Memory Stats")
    stats = agent.memory.get_memory_stats(user_id=st.session_state.user_id)
    st.metric("Episodic Memories", stats["episodic"])
    st.metric("Semantic Facts", stats["semantic"])

    # Memory viewer
    st.divider()
    st.subheader("🔍 Memory Viewer")
    if st.button("Show what I remember about you"):
        with st.spinner("Fetching memories..."):
            summary = agent.get_memory_summary(st.session_state.user_id)
        st.markdown(summary)

# ─────────────────────────────────────────────
# Main chat area
# ─────────────────────────────────────────────

st.title("🧠 AI Memory Agent")
st.caption(f"Chatting as **{st.session_state.user_id}** · Session `{st.session_state.session_id}`")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ─────────────────────────────────────────────
# Voice text injection (from listener)
# ─────────────────────────────────────────────

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

# ─────────────────────────────────────────────
# Text chat input
# ─────────────────────────────────────────────

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

# ─────────────────────────────────────────────
# Voice input controls (local only)
# ─────────────────────────────────────────────

if st.session_state.voice and not IS_CLOUD:
    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        if st.button(
            "🎤 Start Listening",
            use_container_width=True,
            disabled=st.session_state.listening
        ):
            st.session_state.listening = True
            st.rerun()

    with col2:
        if st.button(
            "⏹️ Stop",
            use_container_width=True,
            disabled=not st.session_state.listening
        ):
            st.session_state.listening = False
            st.rerun()

    if st.session_state.listening:
        with st.spinner("🎙️ Listening... Speak now!"):
            try:
                import time
                voice_text = st.session_state.voice.record_audio(duration=5)
                st.success(f"✅ Heard: '{voice_text}'")
                st.session_state.voice_text = voice_text
                st.session_state.listening = False
                time.sleep(0.5)
                st.rerun()
            except Exception as e:
                st.error(f"❌ Voice error: {str(e)}")
                st.session_state.listening = False

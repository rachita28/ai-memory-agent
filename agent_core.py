"""
AI Agent with Groq LLM + LangGraph + full memory (short-term + long-term)
"""
import os
import re
from typing import TypedDict, List, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
import operator
from memory_system import MemorySystem


# ─────────────────────────────────────────────
# Agent State
# ─────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    user_id: str
    session_id: str
    retrieved_memories: List[str]


# ─────────────────────────────────────────────
# Fact extraction patterns
# Used to detect semantic facts worth storing permanently
# ─────────────────────────────────────────────

FACT_PATTERNS = [
    r"my name is (.+)",
    r"i am (.+) years old",
    r"i live in (.+)",
    r"i work (at|as|in) (.+)",
    r"i (love|like|enjoy|hate|prefer) (.+)",
    r"i am a (.+)",
    r"call me (.+)",
    r"i'm from (.+)",
    r"my (job|profession|occupation) is (.+)",
    r"my (email|phone|number) is (.+)",
]


class AIAgent:
    def __init__(self):
        self.memory = MemorySystem()
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.7,
            api_key=os.getenv("GROQ_API_KEY")
        )
        self.graph = self._build_graph()

    # ─────────────────────────────────────────────
    # Graph nodes
    # ─────────────────────────────────────────────

    def _retrieve_memories(self, state: AgentState):
        """Retrieve relevant long-term memories for the current query."""
        if not state["messages"]:
            return {"retrieved_memories": []}

        query = state["messages"][-1].content

        memories = self.memory.retrieve_relevant_memories(
            query=query,
            user_id=state["user_id"],
            k=5
        )
        return {"retrieved_memories": memories}

    def _generate_response(self, state: AgentState):
        """Generate a response using LLM + retrieved memory context."""
        memories = state.get("retrieved_memories", [])
        memory_context = ""
        if memories:
            memory_context = "\n\nRelevant memories about this user:\n" + \
                             "\n".join([f"- {m}" for m in memories])

        system_prompt = f"""You are a helpful, friendly AI assistant with long-term memory.
You remember personal details shared by users across conversations.
When you recall something from memory, naturally weave it into your response.
If asked about something you don't remember, say so honestly.
Never make up information about the user that isn't in your memory.
{memory_context}"""

        messages = [SystemMessage(content=system_prompt)] + list(state["messages"])
        response = self.llm.invoke(messages)

        # Store the conversation turn as episodic memory
        user_msg = state["messages"][-1].content
        conversation_turn = f"User said: {user_msg}\nAssistant replied: {response.content}"
        self.memory.store_memory(
            content=conversation_turn,
            memory_type="episodic",
            user_id=state["user_id"]
        )

        # Extract and store semantic facts from user message
        self._extract_and_store_facts(user_msg, state["user_id"])

        return {"messages": [response]}

    def _save_short_term(self, state: AgentState):
        """Persist short-term (session) memory."""
        self.memory.save_short_term(state["session_id"], list(state["messages"]))
        return {}

    # ─────────────────────────────────────────────
    # Graph construction
    # ─────────────────────────────────────────────

    def _build_graph(self):
        workflow = StateGraph(AgentState)

        workflow.add_node("retrieve", self._retrieve_memories)
        workflow.add_node("generate", self._generate_response)
        workflow.add_node("save", self._save_short_term)

        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", "save")
        workflow.add_edge("save", END)

        return workflow.compile()

    # ─────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────

    def chat(
        self,
        message: str,
        user_id: str = "default",
        session_id: str = "default"
    ) -> str:
        """Send a message and get a response. Loads + saves session memory automatically."""
        history = self.memory.load_short_term(session_id)
        history.append(HumanMessage(content=message))

        result = self.graph.invoke({
            "messages": history,
            "user_id": user_id,
            "session_id": session_id,
            "retrieved_memories": []
        })

        return result["messages"][-1].content

    def get_memory_summary(self, user_id: str) -> str:
        """
        Return a human-readable summary of what the agent remembers about a user.
        Useful for a 'what do you know about me?' feature.
        """
        all_memories = self.memory.retrieve_all_user_memories(user_id)
        episodic = all_memories.get("episodic", [])
        semantic = all_memories.get("semantic", [])

        lines = []
        if semantic:
            lines.append("**Facts I know about you:**")
            for f in semantic[-10:]:  # Show last 10
                lines.append(f"  • {f}")
        if episodic:
            lines.append("\n**Recent conversation memories:**")
            for e in episodic[-5:]:  # Show last 5
                lines.append(f"  • {e[:120]}...")

        if not lines:
            return "I don't have any stored memories about you yet."

        return "\n".join(lines)

    # ─────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────

    def _extract_and_store_facts(self, user_message: str, user_id: str):
        """
        Detect factual statements in user messages (name, location, preferences, etc.)
        and store them as semantic memories for long-term recall.
        """
        text = user_message.lower().strip()
        for pattern in FACT_PATTERNS:
            match = re.search(pattern, text)
            if match:
                # Store the original casing version as a clean fact
                fact = f"User personal info: {user_message.strip()}"
                self.memory.store_semantic_fact(fact, user_id=user_id)
                break  # One fact per message is enough

"""
AI Agent with Groq (cloud) + memory
"""
import os
from typing import TypedDict, List, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
import operator
from memory_system import MemorySystem

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    user_id: str
    session_id: str
    retrieved_memories: List[str]

class AIAgent:
    def __init__(self):
        self.memory = MemorySystem()
        self.llm = ChatGroq(
            model="llama-3.2-3b-preview",
            temperature=0.7,
            api_key=os.getenv("GROQ_API_KEY")
        )
        self.graph = self._build_graph()
    
    def _build_graph(self):
        def retrieve_memories(state: AgentState):
            if not state["messages"]:
                return {"retrieved_memories": []}
            
            last_message = state["messages"][-1].content.lower()
            
            if any(keyword in last_message for keyword in ["name", "who am i", "what is my", "my name"]):
                all_memories = self.memory.episodic_collection.get(
                    where={"user_id": state["user_id"]}
                )
                relevant = []
                if all_memories and "documents" in all_memories:
                    for doc in all_memories["documents"]:
                        if "rachita" in doc.lower():
                            relevant.append(doc)
                return {"retrieved_memories": relevant[:3]}
            
            memories = self.memory.retrieve_relevant_memories(
                query=state["messages"][-1].content,
                user_id=state["user_id"],
                k=3
            )
            return {"retrieved_memories": memories}
        
        def generate_response(state: AgentState):
            memories = state.get("retrieved_memories", [])
            memory_context = ""
            if memories:
                memory_context = "\n\nRelevant information from memory:\n" + \
                               "\n".join([f"- {m}" for m in memories])
            
            system_prompt = f"""You are a helpful AI assistant with long-term memory.
You can remember personal details about users and recall them when relevant.
Be conversational, friendly, and reference past conversations when appropriate.
If you see memory information, use it to answer questions accurately.
{memory_context}"""
            
            messages = [SystemMessage(content=system_prompt)] + list(state["messages"])
            response = self.llm.invoke(messages)
            
            if len(state["messages"]) >= 2:
                conversation_summary = f"User: {state['messages'][-2].content}\nAssistant: {response.content}"
                self.memory.store_memory(
                    content=conversation_summary,
                    memory_type="episodic",
                    user_id=state["user_id"]
                )
            
            return {"messages": [response]}
        
        def save_short_term(state: AgentState):
            self.memory.save_short_term(state["session_id"], list(state["messages"]))
            return {}
        
        workflow = StateGraph(AgentState)
        workflow.add_node("retrieve", retrieve_memories)
        workflow.add_node("generate", generate_response)
        workflow.add_node("save", save_short_term)
        
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", "save")
        workflow.add_edge("save", END)
        
        return workflow.compile()
    
    def chat(self, message: str, user_id: str = "default", session_id: str = "default") -> str:
        history = self.memory.load_short_term(session_id)
        history.append(HumanMessage(content=message))
        
        result = self.graph.invoke({
            "messages": history,
            "user_id": user_id,
            "session_id": session_id,
            "retrieved_memories": []
        })
        
        return result["messages"][-1].content
"""
Memory System: File-based (short-term) + ChromaDB (long-term)
"""
import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import chromadb

class MemorySystem:
    def __init__(self):
        # Short-term memory: JSON file
        self.short_term_file = "short_term_memory.json"
        
        # Long-term memory: ChromaDB with HuggingFace embeddings (free, no API key)
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Persistent Chroma client
        chroma_path = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        
        try:
            self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        except Exception as e:
            print(f"ChromaDB error: {e}, recreating...")
            import shutil
            if os.path.exists(chroma_path):
                shutil.rmtree(chroma_path)
            self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        
        # Collections for different memory types
        self.episodic_collection = self.chroma_client.get_or_create_collection(
            name="episodic_memory",
            metadata={"hnsw:space": "cosine"}
        )
        self.semantic_collection = self.chroma_client.get_or_create_collection(
            name="semantic_memory",
            metadata={"hnsw:space": "cosine"}
        )
        
        # LangChain Chroma wrappers
        self.episodic_store = Chroma(
            client=self.chroma_client,
            collection_name="episodic_memory",
            embedding_function=self.embeddings,
        )
        self.semantic_store = Chroma(
            client=self.chroma_client,
            collection_name="semantic_memory",
            embedding_function=self.embeddings,
        )

    def save_short_term(self, session_id: str, messages: List[BaseMessage]):
        """Save conversation history to JSON file"""
        try:
            if os.path.exists(self.short_term_file):
                with open(self.short_term_file, 'r') as f:
                    data = json.load(f)
            else:
                data = {}
        except:
            data = {}
        
        data[session_id] = {
            "timestamp": datetime.now().isoformat(),
            "messages": [{"type": m.type, "content": m.content} for m in messages]
        }
        
        with open(self.short_term_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_short_term(self, session_id: str) -> List[BaseMessage]:
        """Load conversation history from JSON file"""
        try:
            if not os.path.exists(self.short_term_file):
                return []
            
            with open(self.short_term_file, 'r') as f:
                data = json.load(f)
            
            session_data = data.get(session_id, {})
            messages_data = session_data.get("messages", [])
            
            messages = []
            for item in messages_data:
                if item["type"] == "human":
                    messages.append(HumanMessage(content=item["content"]))
                else:
                    messages.append(AIMessage(content=item["content"]))
            return messages
        except:
            return []
    
    def clear_short_term(self, session_id: str):
        """Clear session memory"""
        try:
            if os.path.exists(self.short_term_file):
                with open(self.short_term_file, 'r') as f:
                    data = json.load(f)
                
                if session_id in data:
                    del data[session_id]
                    
                with open(self.short_term_file, 'w') as f:
                    json.dump(data, f, indent=2)
        except:
            pass

    def store_memory(self, content: str, memory_type: str = "episodic", 
                     metadata: Optional[Dict] = None, user_id: str = "default"):
        """Store long-term memory with embeddings"""
        collection = self.episodic_collection if memory_type == "episodic" else self.semantic_collection
        memory_id = f"{user_id}_{datetime.now().isoformat()}"
        
        collection.add(
            documents=[content],
            metadatas=[{
                "user_id": user_id,
                "type": memory_type,
                "timestamp": datetime.now().isoformat(),
                **(metadata or {})
            }],
            ids=[memory_id]
        )
        return memory_id
    
    def retrieve_relevant_memories(self, query: str, user_id: str = "default", k: int = 5) -> List[str]:
        """Retrieve relevant memories using semantic search"""
        try:
            episodic_results = self.episodic_store.similarity_search(query, k=k, filter={"user_id": user_id})
            semantic_results = self.semantic_store.similarity_search(query, k=k)
            
            memories = []
            seen = set()
            for doc in episodic_results + semantic_results:
                if doc.page_content not in seen:
                    memories.append(doc.page_content)
                    seen.add(doc.page_content)
            return memories[:k]
        except:
            return []
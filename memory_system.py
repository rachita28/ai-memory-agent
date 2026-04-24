"""
Memory System: File-based (short-term) + ChromaDB (long-term)
Includes: summarization, deduplication, semantic + episodic storage
"""
import os
import json
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import chromadb

# Max messages to keep in short-term before summarizing
SHORT_TERM_LIMIT = 20


class MemorySystem:
    def __init__(self):
        # Short-term memory: JSON file
        self.short_term_file = "short_term_memory.json"

        # Long-term memory: ChromaDB with HuggingFace embeddings (free, no API key)
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        try:
            self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        except Exception:
            self.chroma_client = chromadb.Client()  # Fallback to in-memory

        # Collections for different memory types
        self.episodic_collection = self.chroma_client.get_or_create_collection(
            name="episodic_memory",
            metadata={"hnsw:space": "cosine"}
        )
        self.semantic_collection = self.chroma_client.get_or_create_collection(
            name="semantic_memory",
            metadata={"hnsw:space": "cosine"}
        )

        # LangChain Chroma wrappers for similarity search
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

    # ─────────────────────────────────────────────
    # SHORT-TERM MEMORY (JSON file)
    # ─────────────────────────────────────────────

    def save_short_term(self, session_id: str, messages: List[BaseMessage]):
        """Save conversation history to JSON file. Trims if over SHORT_TERM_LIMIT."""
        try:
            data = self._load_json()
        except Exception:
            data = {}

        serialized = [{"type": m.type, "content": m.content} for m in messages]

        # Keep only last SHORT_TERM_LIMIT messages to avoid bloat
        if len(serialized) > SHORT_TERM_LIMIT:
            serialized = serialized[-SHORT_TERM_LIMIT:]

        data[session_id] = {
            "timestamp": datetime.now().isoformat(),
            "messages": serialized
        }

        self._save_json(data)

    def load_short_term(self, session_id: str) -> List[BaseMessage]:
        """Load conversation history from JSON file."""
        try:
            data = self._load_json()
            session_data = data.get(session_id, {})
            messages_data = session_data.get("messages", [])

            messages = []
            for item in messages_data:
                if item["type"] == "human":
                    messages.append(HumanMessage(content=item["content"]))
                else:
                    messages.append(AIMessage(content=item["content"]))
            return messages
        except Exception:
            return []

    def clear_short_term(self, session_id: str):
        """Clear a session's short-term memory."""
        try:
            data = self._load_json()
            if session_id in data:
                del data[session_id]
            self._save_json(data)
        except Exception:
            pass

    def get_all_sessions(self) -> List[str]:
        """Return all session IDs stored in short-term memory."""
        try:
            data = self._load_json()
            return list(data.keys())
        except Exception:
            return []

    # ─────────────────────────────────────────────
    # LONG-TERM MEMORY (ChromaDB)
    # ─────────────────────────────────────────────

    def store_memory(
        self,
        content: str,
        memory_type: str = "episodic",
        metadata: Optional[Dict] = None,
        user_id: str = "default"
    ) -> Optional[str]:
        """
        Store long-term memory with embeddings.
        Skips duplicate content (same hash already stored for this user).
        memory_type: 'episodic' (conversations) or 'semantic' (facts/knowledge)
        """
        try:
            content = content.strip()
            if not content:
                return None

            # Deduplication: skip if exact content already stored
            content_hash = self._hash(content)
            existing_id = f"{user_id}_{content_hash}"
            collection = (
                self.episodic_collection
                if memory_type == "episodic"
                else self.semantic_collection
            )

            existing = collection.get(ids=[existing_id])
            if existing and existing.get("ids"):
                return existing_id  # Already stored, skip

            collection.add(
                documents=[content],
                metadatas=[{
                    "user_id": user_id,
                    "type": memory_type,
                    "timestamp": datetime.now().isoformat(),
                    **(metadata or {})
                }],
                ids=[existing_id]
            )
            return existing_id

        except Exception as e:
            print(f"Memory storage error: {e}")
            return None

    def store_semantic_fact(self, fact: str, user_id: str = "default") -> Optional[str]:
        """
        Convenience method: store a factual/semantic memory.
        Use this for facts like 'User's name is X', 'User lives in Y', etc.
        """
        return self.store_memory(fact, memory_type="semantic", user_id=user_id)

    def retrieve_relevant_memories(
        self, query: str, user_id: str = "default", k: int = 5
    ) -> List[str]:
        """
        Retrieve relevant memories using semantic similarity search.
        Searches both episodic and semantic collections, deduplicates results.
        """
        try:
            memories = []
            seen = set()

            # Search episodic memories (conversations)
            try:
                episodic_results = self.episodic_store.similarity_search(
                    query, k=k, filter={"user_id": user_id}
                )
                for doc in episodic_results:
                    if doc.page_content not in seen:
                        memories.append(doc.page_content)
                        seen.add(doc.page_content)
            except Exception:
                pass

            # Search semantic memories (facts) — no user filter since facts can be global
            try:
                semantic_results = self.semantic_store.similarity_search(
                    query, k=k, filter={"user_id": user_id}
                )
                for doc in semantic_results:
                    if doc.page_content not in seen:
                        memories.append(doc.page_content)
                        seen.add(doc.page_content)
            except Exception:
                pass

            return memories[:k]

        except Exception as e:
            print(f"Memory retrieval error: {e}")
            return []

    def retrieve_all_user_memories(self, user_id: str) -> Dict[str, List[str]]:
        """
        Retrieve ALL stored memories for a user (episodic + semantic).
        Useful for debugging or showing the user their memory profile.
        """
        result = {"episodic": [], "semantic": []}
        try:
            ep = self.episodic_collection.get(where={"user_id": user_id})
            if ep and ep.get("documents"):
                result["episodic"] = ep["documents"]
        except Exception:
            pass

        try:
            sem = self.semantic_collection.get(where={"user_id": user_id})
            if sem and sem.get("documents"):
                result["semantic"] = sem["documents"]
        except Exception:
            pass

        return result

    def delete_user_memories(self, user_id: str):
        """Delete ALL long-term memories for a user."""
        try:
            ep = self.episodic_collection.get(where={"user_id": user_id})
            if ep and ep.get("ids"):
                self.episodic_collection.delete(ids=ep["ids"])
        except Exception:
            pass

        try:
            sem = self.semantic_collection.get(where={"user_id": user_id})
            if sem and sem.get("ids"):
                self.semantic_collection.delete(ids=sem["ids"])
        except Exception:
            pass

    def get_memory_stats(self, user_id: Optional[str] = None) -> Dict[str, int]:
        """Return memory counts."""
        try:
            if user_id:
                ep = self.episodic_collection.get(where={"user_id": user_id})
                sem = self.semantic_collection.get(where={"user_id": user_id})
                return {
                    "episodic": len(ep.get("ids", [])) if ep else 0,
                    "semantic": len(sem.get("ids", [])) if sem else 0,
                }
            return {
                "episodic": self.episodic_collection.count(),
                "semantic": self.semantic_collection.count(),
            }
        except Exception:
            return {"episodic": 0, "semantic": 0}

    # ─────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────

    def _load_json(self) -> Dict:
        if not os.path.exists(self.short_term_file):
            return {}
        with open(self.short_term_file, "r") as f:
            return json.load(f)

    def _save_json(self, data: Dict):
        with open(self.short_term_file, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()[:12]

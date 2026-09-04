from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ChatMessage:
    role: str
    content: str


class SessionMemory:
    def __init__(self, max_turns: int = 8):
        self.max_turns = max_turns
        self.sessions: Dict[str, List[ChatMessage]] = {}

    def get_history(self, session_id: str) -> List[ChatMessage]:
        return self.sessions.get(session_id, [])

    def add_user_message(self, session_id: str, content: str):
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        self.sessions[session_id].append(ChatMessage(role="user", content=content))
        self._trim(session_id)

    def add_assistant_message(self, session_id: str, content: str):
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        self.sessions[session_id].append(ChatMessage(role="assistant", content=content))
        self._trim(session_id)

    def _trim(self, session_id: str):
        history = self.sessions.get(session_id, [])
        if len(history) > self.max_turns * 2:
            self.sessions[session_id] = history[-(self.max_turns * 2):]

    def clear(self, session_id: str):
        if session_id in self.sessions:
            del self.sessions[session_id]


# Global singleton memory instance
_global_memory = SessionMemory()


def get_session_memory() -> SessionMemory:
    return _global_memory

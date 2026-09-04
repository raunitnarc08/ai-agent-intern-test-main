from typing import Any, Dict, List, Optional
import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

from agent.orchestrator import AsterRowAgent

app = FastAPI(
    title="Aster & Row Support Agent API",
    description="Thin HTTP layer exposing Aster & Row RAG Support Agent",
    version="2.0.0"
)

agent = AsterRowAgent()


class ChatRequest(BaseModel):
    message: str = Field(..., description="Customer message")
    session_id: Optional[str] = Field(None, description="Unique session identifier")


class ChatResponse(BaseModel):
    session_id: str
    response: str
    sources: List[str]
    handoff: bool
    tool_called: Optional[str] = None


@app.get("/health")
def health_check():
    return {"status": "ok", "agent": "Aster & Row RAG Support v2"}


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    session_id = req.session_id or str(uuid.uuid4())[:8]
    agent_response = agent.chat(session_id=session_id, user_message=req.message)
    
    return ChatResponse(
        session_id=session_id,
        response=agent_response.response,
        sources=agent_response.sources,
        handoff=agent_response.handoff,
        tool_called=agent_response.tool_called
    )


if __name__ == "__main__":
    uvicorn.run("server.py:app", host="0.0.0.0", port=8000, reload=True)

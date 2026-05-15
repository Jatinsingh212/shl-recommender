from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field

from app.agent import respond
from app.catalog import load_catalog
from app.vectorstore import init_vectorstore


app = FastAPI(title="SHL Assessment Recommender", version="1.0.0")


@app.get("/", response_class=HTMLResponse)
async def get_index():
    return FileResponse("index.html")


class Message(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation]
    end_of_conversation: bool


@app.on_event("startup")
def _warm_catalog() -> None:
    load_catalog()
    init_vectorstore()



@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    result = respond([message.model_dump() for message in request.messages])
    return ChatResponse(**result.__dict__)

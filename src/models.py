from typing import Any, Literal

from pydantic import BaseModel, Field

Status = Literal["answered", "abstained", "escalated", "review_required"]


class Ticket(BaseModel):
    ticket_id: str = Field(min_length=1)
    customer_id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    body: str = Field(min_length=1)


class Customer(BaseModel):
    customer_id: str = Field(min_length=1)
    name: str
    email: str
    plan: str
    verified: bool


class Queue(BaseModel):
    queue: str
    owner: str
    channel: str
    available: bool


class CaseMemory(BaseModel):
    conversation_id: str
    last_ticket_id: str | None = None
    last_customer_id: str | None = None
    notes: list[str] = []


class EvalCase(BaseModel):
    case_id: str = Field(min_length=1)
    ticket_id: str | None
    prompt: str
    expected_status: Status
    expected_tools: list[str]
    expected_sources: list[str]
    must_use_memory: bool
    forbidden_tools: list[str] = []
    expected_safety_events: list[str] = []


class Evidence(BaseModel):
    source_id: str
    text: str
    score: float


class SearchResult(BaseModel):
    query: str
    evidence: list[Evidence]
    best_score: float
    removed_suspicious: int = 0


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any]
    succeeded: bool
    summary: str = ""


class Draft(BaseModel):
    kind: Literal["ticket_update", "escalation"]
    ticket_id: str
    queue: str | None = None
    message: str


class AgentResult(BaseModel):
    status: Status
    answer: str
    citations: list[str] = []
    evidence_sources: list[str] = []
    tool_calls: list[ToolCall] = []
    safety_events: list[str] = []
    used_memory: bool = False
    ticket_id: str | None = None
    steps: int = 0
    best_score: float = 0.0
    draft: Draft | None = None

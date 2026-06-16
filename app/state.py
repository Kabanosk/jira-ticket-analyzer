from typing import Literal

from pydantic import BaseModel


class InputState(BaseModel):
    ticket_id: str
    title: str
    description: str


class ClassifyNodeState(BaseModel):
    labels: list[str]
    work_type: Literal["bug", "feature", "task", "improvement"]


class AnalyzeNodeState(BaseModel):
    priority: Literal["low", "medium", "high", "very high"]
    similar_ticket_ids: list[str]


class RecommendNodeState(BaseModel):
    recommended_action: str


class Metadata(BaseModel):
    status: Literal["completed", "running", "failed", "waiting"]
    error_msg: str = ""


class TicketState(BaseModel):
    input_state: InputState
    classify_state: ClassifyNodeState | None = None
    analyze_state: AnalyzeNodeState | None = None
    recommend_state: RecommendNodeState | None = None
    metadata: Metadata = Metadata(status="waiting")

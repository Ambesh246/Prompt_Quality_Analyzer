from __future__ import annotations
from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import Any

MAX_PROMPT_CHARS = 8000

class AnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(..., min_length=1, max_length=MAX_PROMPT_CHARS)

    @field_validator("prompt")
    @classmethod
    def prompt_must_contain_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Please enter a prompt before analyzing it.")
        return value.strip()

class ImproveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(..., min_length=1, max_length=MAX_PROMPT_CHARS)
    use_provider: bool = True

    @field_validator("prompt")
    @classmethod
    def prompt_must_contain_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Please enter a prompt before improving it.")
        return value.strip()

class AnalysisResponse(BaseModel):
    score: float
    category: str
    assessment: str
    rubric_score: float
    model_score: float | None
    model_preference_score: float | None
    scoring_method: str
    dimensions: dict[str, float]
    dimension_labels: dict[str, str]
    strengths: list[str]
    weaknesses: list[str]
    recommendations: list[str]
    features: dict[str, Any]
    model: dict[str, Any]

class ImproveResponse(BaseModel):
    status: str
    provider: str | None = None
    model: str | None = None
    original_prompt: str
    rewritten_prompt: str | None = None
    original_analysis: AnalysisResponse
    improved_analysis: AnalysisResponse | None = None
    score_delta: float | None = None
    changes: list[str] = Field(default_factory=list)
    notice: str | None = None
    error: str | None = None
    diagnostic_type: str | None = None

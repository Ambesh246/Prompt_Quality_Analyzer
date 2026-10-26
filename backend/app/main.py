from __future__ import annotations
import time
from collections import defaultdict, deque
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import allowed_origins, max_requests_per_minute
from app.ml.model import PreferenceModel
from app.schemas import AnalysisRequest, AnalysisResponse, ImproveRequest, ImproveResponse
from app.services.scoring import assess_prompt, make_score, DIMENSION_LABELS
from app.services.rewrite import improve_prompt

API_VERSION = "1.0.0"


class InMemoryRateLimiter:
    """Small-process abuse guard; use a shared gateway/Redis limiter at scale."""
    def __init__(self, limit: int):
        self.limit = limit
        self.requests: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        queue = self.requests[key]
        while queue and now - queue[0] > 60:
            queue.popleft()
        if len(queue) >= self.limit:
            return False
        queue.append(now)
        return True


def create_app(model_service: PreferenceModel | None = None) -> FastAPI:
    app = FastAPI(
        title="Prompt Quality Analyzer API",
        description="Explainable prompt rubric with a learned relative-preference signal and safe rewrite fallback.",
        version=API_VERSION,
    )
    app.state.model_service = model_service or PreferenceModel()
    app.state.rate_limiter = InMemoryRateLimiter(max_requests_per_minute())

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins(),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization"],
        max_age=600,
    )

    @app.middleware("http")
    async def request_size_and_rate_limit(request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > 100_000:
                    return JSONResponse(status_code=413, content={"detail": "Request body is too large."})
            except ValueError:
                return JSONResponse(status_code=400, content={"detail": "Invalid Content-Length header."})
        if request.method == "POST" and request.url.path in {"/analyze", "/improve"}:
            client_host = request.client.host if request.client else "unknown"
            if not app.state.rate_limiter.allow(client_host):
                return JSONResponse(status_code=429, content={"detail": "Rate limit reached. Please wait a moment and retry."})
        return await call_next(request)

    def score_prompt(prompt: str) -> AnalysisResponse:

        model: PreferenceModel = app.state.model_service
        rubric = assess_prompt(prompt)


        ml_score = (
        model.predict_score(prompt)
        if model.available
        else None
         )

        scoring_mode = (
        model.scoring_mode
        if model.available
        else "unavailable"
         )

        overall, method = make_score(
        rubric.score,
        ml_score,
         scoring_mode,
           )
        # The category and narrative reflect the combined score, not a claimed ground truth.
        if overall < 40:
            category = "Needs Improvement"
            assessment = "The prompt is a starting point, but important details may be missing or ambiguous."
        elif overall < 60:
            category = "Fair"
            assessment = "The core request is present; a few targeted details could make the expected result more predictable."
        elif overall < 80:
            category = "Good"
            assessment = "The prompt gives a useful direction. A small amount of targeted refinement may help."
        else:
            category = "Excellent"
            assessment = "The prompt is well-directed for its apparent task. Adding more words is not automatically better."
        return AnalysisResponse(
            score=overall,
            category=category,
            assessment=assessment,
            rubric_score=rubric.score,
            model_score=ml_score,

            model_preference_score=(
            ml_score
            if scoring_mode == "bpo_preference_classifier"
            else None
            ),            
            scoring_method=method,
            dimensions=rubric.dimensions,
            dimension_labels=DIMENSION_LABELS,
            strengths=rubric.strengths,
            weaknesses=rubric.weaknesses,
            recommendations=rubric.recommendations,
            features=rubric.features,
            model=model.public_info(),
        )

    @app.get("/health")
    async def health() -> dict[str, Any]:
        model: PreferenceModel = app.state.model_service
        return {
            "status": "ok" if model.available else "degraded",
            "api_version": API_VERSION,
            "model_available": model.available,
        }

    @app.get("/model-info")
    async def model_info() -> dict[str, Any]:
        model: PreferenceModel = app.state.model_service
        return {
            **model.public_info(),
            "rubric_version": "rubric-v1.0",
            "rubric_weights": {
                "clarity": 0.14, "context": 0.09, "specificity": 0.12,
                "task_definition": 0.14, "constraints": 0.10, "output_format": 0.08,
                "audience_purpose": 0.07, "completeness": 0.10,
                "actionability": 0.10, "ambiguity_redundancy": 0.06,
            },
        }

    @app.post("/analyze", response_model=AnalysisResponse)
    async def analyze(body: AnalysisRequest) -> AnalysisResponse:
        return score_prompt(body.prompt)

    @app.post("/improve", response_model=ImproveResponse)
    async def improve(body: ImproveRequest) -> ImproveResponse:
        original_analysis = score_prompt(body.prompt)
        rubric = assess_prompt(body.prompt)
        result = await improve_prompt(body.prompt, rubric, use_provider=body.use_provider)
        if result.get("status") != "success" or not result.get("rewritten_prompt"):
            return ImproveResponse(
                status="provider_error", provider=result.get("provider"), model=result.get("model"),
                original_prompt=body.prompt, original_analysis=original_analysis,
                error=result.get("error", "Rewrite provider could not complete the request."),
                diagnostic_type=result.get("diagnostic_type"),
            )
        improved_prompt = str(result["rewritten_prompt"]).strip()
        improved_analysis = score_prompt(improved_prompt)
        return ImproveResponse(
            status="success", provider=result.get("provider"), model=result.get("model"),
            original_prompt=body.prompt, rewritten_prompt=improved_prompt,
            original_analysis=original_analysis, improved_analysis=improved_analysis,
            score_delta=round(improved_analysis.score - original_analysis.score, 1),
            changes=result.get("changes", []), notice=result.get("notice"),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        # Do not echo submitted prompt content or Pydantic's raw input values in error responses.
        return JSONResponse(
            status_code=422,
            content={"detail": "Invalid request. Provide a non-empty prompt of at most 8,000 characters and use only supported fields."},
        )

    return app


app = create_app()

# Architecture

```text
React + Vite browser UI
    │  JSON over HTTPS; VITE_API_BASE_URL
    ▼
FastAPI (Pydantic validation, size guard, CORS, in-process rate guard)
    ├── /analyze
    │    ├── rubric engine (10 dimensions; transparent feature cues)
    │    ├── loaded joblib TF-IDF + logistic-regression preference model
    │    └── response explanations + metadata
    └── /improve
         ├── original analysis (kept even if rewriting fails)
         ├── optional OpenAI Responses API provider (server key, opt-in)
         └── deterministic fallback (no provider key required)
```

## Responsibilities

- **`app/main.py`:** routes, model availability checks, health reporting, CORS, input-size guard and basic rate limit. Model artifacts are loaded by `PreferenceModel` at startup; request handlers never train a model.
- **`app/schemas.py`:** strict Pydantic request/response contracts; prompts are limited to 8,000 characters.
- **`app/services/scoring.py`:** the explainable rubric, dimension weights, strengths/weaknesses/recommendations, and combined score method.
- **`app/ml/model.py`:** load the saved ML pipeline and versioned metadata; failed loading remains visible as degraded state.
- **`app/services/rewrite.py`:** rewrite abstraction and safe no-key fallback; optional OpenAI Responses API with timeout and explicit provider failures.
- **`frontend/src/App.tsx`:** accessible responsive workflow, score details, original-vs-improved comparison, editing/copy/download and theme preference.
- **`notebooks/prompt_quality_analyzer_training.ipynb`:** public dataset acquisition, EDA, cleaning, word-only baseline, word+character model, validation selection, held-out test, robustness probes and artifact export.

## Trust boundaries

1. Prompt text is untrusted data. The analyzer does not execute submitted code or follow submitted prompts as server instructions.
2. The only outbound prompt flow is the optional rewrite provider, explicitly surfaced in the UI. API credentials are loaded on the backend.
3. Analysis requests are not stored. Application code does not log the prompt body.
4. The current in-memory rate limiter is process-local and can be bypassed by deploying multiple instances. Put a shared limiter/API gateway in front for production.
5. No user database is required for the main flow. If history is introduced later, require explicit opt-in, retention rules and deletion control.

## Score composition

The rubric produces a weighted score and ten dimensions. The classifier estimates whether the prompt resembles the optimized side of a prompt pair. The current score is an 80/20 combination. This is a disclosed product heuristic; it should be validated against independent human ratings before being treated as predictive.

## Failure behaviour

- Missing/corrupt model artifact: `/health` reports `degraded`; `/analyze` and `/improve` return a clear 503 detail instead of random or mock scores.
- Provider timeout/failure: `/improve` returns a provider-error response containing original analysis; it does not label fallback output as provider-generated.
- No provider key, or provider disabled: deterministic template-based rewrite is returned with an explicit notice.

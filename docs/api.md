# API Contract

Base URL locally: `http://127.0.0.1:8000`. Interactive schema: `/docs`.

## `GET /health`

Example response:

```json
{"status":"ok","api_version":"1.0.0","model_available":true}
```

`status` is `degraded` when the artifact cannot load. This endpoint does not claim the model is scientifically validated.

## `GET /model-info`

Returns model version, training data, label semantics, evaluation metadata, model limitations, rubric version and ten rubric weights. The checked-in local model is labeled `pqa-seed-synthetic-v0.1` and `production_ready: false`. After running Colab and replacing artifacts, this endpoint should show `pqa-bpo-pair-preference-v1.0`.

## `POST /analyze`

Request schema:

```json
{"prompt":"Explain photosynthesis in simple terms, with one example."}
```

Main response fields:

- `score`: 0–100 overall estimate.
- `category`, `assessment`: qualitative interpretation of the combined score.
- `rubric_score`: independent transparent-rubric result.
- `model_preference_score`: learned estimate that the text resembles the optimized side of a training pair.
- `scoring_method`: explicit formula label.
- `dimensions`: ten numeric dimension scores; `dimension_labels` maps internal keys to display names.
- `strengths`, `weaknesses`, `recommendations`, `features`: explainable findings based on detected cues.
- `model`: version and training-data metadata.

Errors: `422` for invalid, empty, overlong or extra-field payloads; `413` for a request body exceeding the middleware limit; `429` if the process-local rate limit is exceeded; `503` if the model artifact is unavailable.

## `POST /improve`

Request:

```json
{"prompt":"help me with my Python code","use_provider":false}
```

Response fields include `status`, `provider`, `original_prompt`, `rewritten_prompt`, `original_analysis`, `improved_analysis`, `score_delta`, `changes`, and `notice`. `use_provider=false` explicitly selects the deterministic fallback. If enabled and `OPENAI_API_KEY` is configured on the server, the input can be sent to OpenAI's Responses API. Provider failures return `status: provider_error` with the original analysis preserved.

A positive `score_delta` means the current scoring method estimates a higher score. It is not proof that a downstream model will give a better answer. Negative or flat deltas are not suppressed.

## CORS and limits

`CORS_ORIGINS` is a comma-separated exact allowlist. `AnalysisRequest.prompt` and `ImproveRequest.prompt` accept 1–8,000 non-whitespace characters. Middleware rejects declared request bodies over 100,000 bytes. Default in-process request limit is 30 requests per minute per client IP; production deployments should use a shared limiter/gateway.

# Prompt Quality Analyzer (Promptly)

A student-friendly full-stack app for evaluating prompt clarity and structure, showing ten explainable dimensions, suggesting targeted edits, and producing an intent-preserving rewrite.

> **Research integrity note:** The included local model artifact is a *synthetic bootstrap model* so the project can start and the API can be tested immediately. Its 100% holdout result measures separation of synthetic templates, not real-world prompt quality. It is explicitly marked `production_ready: false`. Run `notebooks/prompt_quality_analyzer_training.ipynb` in Google Colab to train the better BPO pair-preference model, then replace the artifact files before interpreting the learned score beyond a demo. BPO does **not** contain objective 0–100 quality labels, so even that model predicts an optimized-style preference signal, not universal prompt quality.

## What is implemented

- **React + Vite + TypeScript UI:** responsive input workspace, examples, theme toggle, score ring, ten dimension bars, strengths/weaknesses, actionable recommendations, before/after comparison, editable rewrite, copy and `.txt` download.
- **FastAPI backend:** request/response validation, `/health`, `/model-info`, `/analyze`, and `/improve`; CORS allowlist from environment; basic per-process rate limiting and request-size limit.
- **ML pipeline:** saved word/character TF-IDF + logistic-regression artifact, versioned metadata, explicit label semantics, no retraining during API requests.
- **Explainable rubric:** configurable weights for clarity, context, specificity, task definition, constraints, output format, audience/purpose, completeness, actionability, and ambiguity/redundancy. Simple factual questions receive context-aware defaults rather than a penalty for being short.
- **Rewrite engine:** deterministic fallback with optional placeholders; optional OpenAI Responses API call when explicitly enabled and configured. Provider errors are not mislabeled as a successful LLM rewrite.
- **Tests and docs:** backend unit/API/integration tests, frontend tests, CI workflow, dataset research, dataset card, model card, architecture, API and deployment notes.

## Quick start

Prerequisites: Python 3.10+ (3.11 recommended), Node.js 20+ and npm. Commands below were tested for the backend in the creation environment; frontend dependency installation/build needs a network-enabled environment and is **not claimed as verified here**.

### 1. Run the backend

```bash
cd backend
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` for the interactive API docs. The default local artifact is included at `backend/models/`; API startup reports its version and source through `/model-info`.

### 2. Run the frontend

Open a second terminal:

```bash
cd frontend
cp .env.example .env
npm install
npm run dev
```

Vite will print a local URL, usually `http://localhost:5173`. The default frontend API base URL is `http://localhost:8000`.

### 3. Run the backend tests

```bash
cd backend
pytest
```

### 4. Train the BPO-based model in Colab

1. Upload/open `notebooks/prompt_quality_analyzer_training.ipynb` in Google Colab.
2. Run all cells from top to bottom in a fresh runtime. Internet access is required to download BPO.
3. Review the validation model-comparison table, held-out test metrics, error examples and robustness probes. The score estimates an original-vs-optimized pair label, not general ground truth.
4. Download `prompt_quality_model_artifacts.zip` from the last cell and extract `prompt_preference.joblib` and `metadata.json` into `backend/models/`, replacing the bootstrap files.
5. Restart the backend and verify `GET /model-info` reports `pqa-bpo-pair-preference-v1.0` and the BPO data source.
6. Do not claim the model is validated for universal prompt quality until you evaluate it against independently human-rated prompts.

The notebook has been authored but **has not been executed in Colab in this session**; its data download, training metrics, and exported BPO artifact are therefore unverified until you run it.

## API overview

### `GET /health`

Returns the API state and whether a model artifact is loaded. A missing artifact is exposed as `degraded` rather than silently replacing the model with random scores.

### `GET /model-info`

Returns version, training data, label semantics, metrics, limitations and rubric weights.

### `POST /analyze`

Request:

```json
{ "prompt": "Explain photosynthesis in simple terms, with one example." }
```

Response fields include `score` (0–100 estimate), `category`, `rubric_score`, `model_preference_score`, `scoring_method`, ten `dimensions`, `strengths`, `weaknesses`, `recommendations`, and `model` metadata. Numeric example values are not fixed; actual values depend on the current artifact and rubric.

### `POST /improve`

```json
{ "prompt": "help me with my Python code", "use_provider": false }
```

The no-provider path is deterministic and key-free. It returns the revised prompt, original/improved analyses, `score_delta`, and the method label. A score increase is never guaranteed or artificially forced. With `use_provider: true` and a valid server-side `OPENAI_API_KEY`, the configured OpenAI Responses API may be used; submitted text is then sent to that external provider. The UI exposes this as an opt-in control.

## Environment variables

Backend (`backend/.env.example`):

- `CORS_ORIGINS`: comma-separated allowed browser origins.
- `OPENAI_API_KEY`: optional server-side key for rewriting only.
- `OPENAI_MODEL`: optional model identifier for the Responses API.
- `MODEL_DIR`: optional model artifact directory.
- `RATE_LIMIT_PER_MINUTE`: in-process abuse protection. At scale, use a trusted gateway or shared limiter.

Frontend (`frontend/.env.example`):

- `VITE_API_BASE_URL`: public API base URL. This value is bundled into browser code, so never put secrets here.

No submitted prompt history is stored by this implementation, and request bodies are not logged by application code. If the optional provider is enabled, its provider-side processing and retention terms also apply.

## Repository map

```text
prompt-quality-analyzer/
├── backend/
│   ├── app/main.py
│   ├── app/config.py
│   ├── app/schemas.py
│   ├── app/ml/model.py
│   ├── app/services/scoring.py
│   ├── app/services/rewrite.py
│   ├── models/prompt_preference.joblib
│   ├── models/metadata.json
│   ├── tests/
│   ├── train_seed_model.py
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── src/App.tsx
│   ├── src/styles.css
│   ├── src/services/api.ts
│   ├── src/types/index.ts
│   ├── package.json
│   └── .env.example
├── notebooks/prompt_quality_analyzer_training.ipynb
├── data/README.md
├── docs/
└── .github/workflows/ci.yml
```

## Important limitations

1. Prompt quality depends on task, user intent, model, and context. There is no universal objective 0–100 score here.
2. BPO labels derive from whether a sample is an original or optimized prompt. Its documentation states the optimization data were generated with GPT-3.5 and may contain imperfect edits.
3. The rubric uses transparent text cues and task-aware defaults; it cannot fully understand domain truth or predict how a particular model will answer.
4. The local artifact is an explicitly synthetic prototype. Its high seed holdout metrics are not evidence of generalization.
5. The in-memory rate limiter is per process and resets on restart. Production should add an edge/API gateway limiter and monitoring.
6. Human-reviewed prompt-quality scores, multilingual validation, score calibration, semantic duplicate detection, provider error smoke tests and an actual live deployment are still needed for release-level claims.

## Current verification status

- Backend tests: run in the authoring environment; see the handoff report in `docs/verification_report.md`.
- Colab BPO training: not run from this environment.
- Frontend npm install/build/browser test: source and test files are present; not run because this workspace could not resolve external package hosts.
- Vercel/Render deployment and live URL: not performed; hosting credentials are not connected.

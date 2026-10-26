# Prompt Quality Analyzer (Promptly)

link = https://prompt-quality-analyzer-chi.vercel.app

A student-friendly full-stack app for evaluating prompt clarity and structure, showing ten explainable dimensions, suggesting targeted edits, and producing an intent-preserving rewrite.

## What is implemented

- **React + Vite + TypeScript UI:** responsive input workspace, examples, theme toggle, score ring, ten dimension bars, strengths/weaknesses, actionable recommendations, before/after comparison, editable rewrite, copy and `.txt` download.
- **FastAPI backend:** request/response validation, `/health`, `/model-info`, `/analyze`, and `/improve`; CORS allowlist from environment; basic per-process rate limiting and request-size limit.
- **ML pipeline:** saved word/character TF-IDF + logistic-regression artifact, versioned metadata, explicit label semantics, no retraining during API requests.
- **Explainable rubric:** configurable weights for clarity, context, specificity, task definition, constraints, output format, audience/purpose, completeness, actionability, and ambiguity/redundancy. Simple factual questions receive context-aware defaults rather than a penalty for being short.
- **Rewrite engine:** deterministic fallback with optional placeholders; optional OpenAI Responses API call when explicitly enabled and configured. Provider errors are not mislabeled as a successful LLM rewrite.
- **Tests and docs:** backend unit/API/integration tests, frontend tests, CI workflow, dataset research, dataset card, model card, architecture, API and deployment notes.

## Quick start

Prerequisites: Python 3.10+ (3.13.16 recommended), Node.js 20+ and npm. Commands below were tested for the backend in the creation environment; frontend dependency installation/build needs a network-enabled environment and is **not claimed as verified here**.

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

The no-provider path is deterministic and key-free. It returns the revised prompt, original/improved analyses, `score_delta`, and the method label. A score increase is never guaranteed or artificially forced. With `use_provider: true` and a valid server-side `OPENAI_API_KEY`, the configured OpenAI Responses API may be used; submitted text is then sent to that external provider. The UI exposes this as an opt-in control.

## Important limitations

1. Prompt quality depends on task, user intent, model, and context. There is no universal objective 0–100 score here.
2. BPO labels derive from whether a sample is an original or optimized prompt. Its documentation states the optimization data were generated with GPT-3.5 and may contain imperfect edits.
3. The rubric uses transparent text cues and task-aware defaults; it cannot fully understand domain truth or predict how a particular model will answer.
4. The local artifact is an explicitly synthetic prototype. Its high seed holdout metrics are not evidence of generalization.
5. The in-memory rate limiter is per process and resets on restart. Production should add an edge/API gateway limiter and monitoring.
6. Human-reviewed prompt-quality scores, multilingual validation, score calibration, semantic duplicate detection, provider error smoke tests and an actual live deployment are still needed for release-level claims.

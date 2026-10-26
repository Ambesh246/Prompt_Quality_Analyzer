# Local and Production Deployment

**Deployment has not been performed in this workspace.** These instructions prepare the source; a valid deployment requires you to connect your repository and hosting account, configure public origins and then smoke-test the live URL.

## Local development

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate       # Windows PowerShell: .venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Test: `curl http://127.0.0.1:8000/health`, then open `http://127.0.0.1:8000/docs`.

### Frontend

```bash
cd frontend
cp .env.example .env
npm install
npm run dev
```

The default `VITE_API_BASE_URL` is `http://localhost:8000`. The frontend build only includes public `VITE_` values; never add provider credentials to frontend environment variables.

## Render backend

A starter Blueprint is at `/render.yaml` with `rootDir: backend`, `pip install -r requirements.txt`, and `uvicorn app.main:app --host 0.0.0.0 --port $PORT`. Render's current FastAPI guide documents this build/start pattern: https://render.com/docs/deploy-fastapi and the broader web-service requirements at https://render.com/docs/web-services.

Steps:

1. Push this repository to a Git provider and connect it in Render.
2. Create the service from `render.yaml` or create a Python web service with root directory `backend`.
3. Add `CORS_ORIGINS` with the exact deployed frontend origin, for example `https://your-project.vercel.app` (replace with the real URL).
4. Keep `OPENAI_API_KEY` unset if you only want the deterministic fallback; if set, store it as a Render secret.
5. Verify `/health` and `/model-info`. The checked-in artifact is a synthetic demo; replace it with the Colab BPO-trained candidate first.
6. Send a real `POST /analyze` and `POST /improve` request. Verify errors and logs. A `200` health response alone is not full validation.

The joblib artifact is small in this project. For larger future models, use a versioned artifact store or runtime fetch with integrity checks instead of committing large binaries. Confirm your chosen plan's memory/cold-start limits in the current Render dashboard; resource tiers and free-tier restrictions can change.

## Vercel frontend

The frontend contains a Vite configuration and SPA fallback in `frontend/vercel.json`. Current official Vercel guidance for Vite: https://vercel.com/docs/frameworks/frontend/vite.

1. Import the repository to Vercel.
2. Set the project root to `frontend`.
3. Build command: `npm run build`; output directory: `dist`.
4. Add `VITE_API_BASE_URL` as the deployed backend HTTPS origin, without a trailing slash.
5. Redeploy after setting the environment variable; Vite values are provided at build time.
6. Add the deployed frontend origin to the backend's `CORS_ORIGINS`.
7. Open the production site, run analysis and fallback rewrite, inspect the browser console/network panel, and test on a small mobile viewport.

## Optional OpenAI rewrite provider

The service uses the current Responses API endpoint documented at https://developers.openai.com/api/reference/resources/responses/methods/create (`POST /v1/responses`). Keep `OPENAI_API_KEY` server-side, use a model supported by your account, and configure a timeout. The frontend toggle is off by default; when enabled, the user is told that prompt text may be sent to a configured external provider. Follow OpenAI key safety guidance: https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety.

## Before launch

- Run all backend and frontend tests.
- Run the notebook end-to-end in Colab and verify downloaded artifacts can be loaded from a clean runtime.
- Replace the synthetic bootstrap artifact and re-check `/model-info`.
- Add an independently human-reviewed prompt-quality evaluation set.
- Confirm CORS, rate limits, provider timeouts, privacy disclosure and failure recovery.
- Smoke-test both hosts using their live URLs; record timestamp and actual responses.
- Review current vendor docs/pricing/resource tiers rather than assuming free-tier behavior.

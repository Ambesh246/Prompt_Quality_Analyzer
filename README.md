# Prompt Quality Analyzer — final deployment package

A model-agnostic prompt review app with:

- **ML estimate:** saved Ridge Regression pipeline trained on the `transformed_quality` proxy labels from `agentlans/prompt-quality`.
- **Task-aware checklist:** rubric signals for general, coding, education, writing, research, data analysis, and translation prompts.
- **Feedback + offline draft:** deterministic suggestions and a task-specific structured rewrite draft.
- **Universal rewrite meta-prompt:** copy it into ChatGPT, Gemini, Claude, or another LLM.
- **Optional API rewrite:** enable an OpenAI-compatible Chat Completions endpoint with Streamlit secrets and explicit user consent.

## Important interpretation

Scores are estimates, not probabilities and not proof that a prompt will produce a better answer. Checklist detections are heuristic. The 60/40 ML/checklist composite weighting is provisional and should be calibrated using independent human ratings and task-outcome testing.

## Files

- `app.py`: Streamlit app.
- `prompt_quality_core.py`: stable feature extractor and task-aware checks; keep it beside `app.py` because the serialized pipeline references it.
- `artifacts/prompt_quality_analyzer_model.joblib`: model artifact provided after the Colab training run.
- `model_environment.json`: training environment metadata.
- `requirements.txt`: pinned core versions to match the model.
- `training/`: reproducible Colab notebook.
- `tests/smoke_test.py`: verifies model loading and basic predictions.

## Test locally

Use Python 3.13 to match the training runtime:

```bash
python3.13 -m venv .venv
source .venv/bin/activate       # macOS/Linux
# .venv\Scripts\Activate.ps1   # Windows PowerShell
python -m pip install --upgrade pip
pip install -r requirements.txt
python tests/smoke_test.py
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Create a GitHub repository and upload this folder's contents. Ensure the `artifacts/` model file is included.
2. In Streamlit Community Cloud, choose **Create app**, your repository, branch `main`, and entrypoint `app.py`.
3. In Advanced settings, select **Python 3.13** to match the saved model's training environment.
4. Deploy and open the app logs if anything fails.
5. Do not upload `.streamlit/secrets.toml` or API keys to GitHub. This package's `.gitignore` excludes the local secrets file.

Official deployment guide: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy

## Optional AI rewrite

Offline mode works without API credentials. To enable AI rewriting, add secrets in the Streamlit app settings:

```toml
LLM_API_KEY = "YOUR_PRIVATE_API_KEY"
LLM_MODEL = "YOUR_PROVIDER_MODEL_ID"
# Optional if the provider uses a custom OpenAI-compatible endpoint:
# LLM_BASE_URL = "https://provider.example/v1"
```

The app asks the user for consent before sending a prompt to the configured provider. Do not put API credentials in GitHub.

## Known limitations / future work

1. Target labels are automated evaluator-derived proxy scores, not universal human ground truth.
2. `np.clip` only bounds display to 0–100; it does not improve the model prediction.
3. Checklist/keyword signals can miss meaning or flag irrelevant dimensions.
4. The improved offline prompt is a structured draft, not a guaranteed semantic improvement.
5. Evaluate separately on human-rated prompts, each task category, and multiple languages. Compare answer quality from original vs. revised prompts using the same target LLM/task.

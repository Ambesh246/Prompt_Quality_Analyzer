# FastAPI backend

See the root README for complete setup. From this folder:

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
pytest
```

API routes: `GET /health`, `GET /model-info`, `POST /analyze`, and `POST /improve`.

The included model file is a synthetic bootstrap artifact only. Run the Colab notebook and replace `models/prompt_preference.joblib` plus `models/metadata.json` before using the learned signal as anything more than a demo.

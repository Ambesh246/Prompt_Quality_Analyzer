"""Deployment smoke test for a trusted model artifact.
Run after installing requirements.txt: python tests/smoke_test.py
"""
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import sklearn

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "prompt_quality_analyzer_model.joblib"
ENV_PATH = ROOT / "model_environment.json"
sys.path.insert(0, str(ROOT))

from prompt_quality_core import (  # noqa: E402
    build_improved_prompt,
    detect_criteria,
    extract_structural_features,
    rubric_score,
)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model artifact not found: {MODEL_PATH}")
if not ENV_PATH.exists():
    raise FileNotFoundError(f"Training environment metadata not found: {ENV_PATH}")

training_env = json.loads(ENV_PATH.read_text())
expected_sklearn = training_env.get("scikit_learn_version")
if expected_sklearn and sklearn.__version__ != expected_sklearn:
    raise RuntimeError(
        f"scikit-learn version mismatch: model trained with {expected_sklearn}, "
        f"currently installed {sklearn.__version__}. Install requirements.txt exactly."
    )

# Only load model artifacts you created or otherwise trust.
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    artifact = joblib.load(MODEL_PATH)

if not isinstance(artifact, dict) or "model" not in artifact:
    raise TypeError("Expected a dictionary artifact containing the fitted 'model'.")
model = artifact["model"]

prompts = [
    "Explain machine learning.",
    "Write a Python function to test whether a number is prime. Include edge cases and three tests.",
    "Translate this sentence into Hindi and preserve its meaning: Hello world.",
]
features = extract_structural_features(prompts)
predictions = np.asarray(model.predict(prompts), dtype=float).reshape(-1)
assert features.shape[0] == len(prompts), "Feature extractor row count mismatch."
assert np.isfinite(features).all(), "Feature extractor returned non-finite values."
assert len(predictions) == len(prompts), "Model returned unexpected prediction count."
assert np.isfinite(predictions).all(), "Model returned non-finite predictions."

checks = detect_criteria(prompts[1], "coding")
score = rubric_score(checks)
draft = build_improved_prompt(prompts[1], "coding", checks)
assert 0 <= score <= 100, "Checklist score is outside 0-100."
assert prompts[1] in draft, "Improved draft did not preserve original prompt text."

print("PASS: model artifact loaded")
print("PASS: training metadata matches scikit-learn", sklearn.__version__)
print("PASS: structural features shape", features.shape)
print("PASS: finite predictions", np.round(predictions, 2).tolist())
print("PASS: coding rubric score", round(score, 2))
print("PASS: improved draft preserves original prompt")
if caught:
    print("WARNING: messages while loading model:")
    for item in caught:
        print(" -", str(item.message)[:250])

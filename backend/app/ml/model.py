
"""Load the Colab regression model or a legacy preference classifier."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib

DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "models"


class PreferenceModel:
    def __init__(self, model_dir: str | Path | None = None):
        configured_dir = model_dir or os.getenv("MODEL_DIR")
        self.model_dir = Path(configured_dir or DEFAULT_MODEL_DIR)

        self.pipeline: Any | None = None
        self.metadata: dict[str, Any] = {}
        self.load_error: str | None = None
        self.scoring_mode = "unavailable"
        self.artifact_filename: str | None = None

        self.load()

    @staticmethod
    def _read_metadata(path: Path) -> dict[str, Any]:
        if not path.is_file():
            return {}

        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Invalid metadata file: {path.name}")

        return data

    def load(self) -> None:
        self.pipeline = None
        self.metadata = {}
        self.load_error = None
        self.scoring_mode = "unavailable"
        self.artifact_filename = None

        errors: list[str] = []

        # First priority: the regression artifact from your Colab notebook.
        regression_path = (
            self.model_dir / "prompt_quality_analyzer_model.joblib"
        )
        regression_metadata = (
            self.model_dir / "prompt_quality_analyzer_metadata.json"
        )

        if regression_path.is_file():
            try:
                artifact = joblib.load(regression_path)

                if not isinstance(artifact, dict):
                    raise TypeError(
                        "Expected a dictionary containing the trained model."
                    )

                pipeline = artifact.get("model")

                if pipeline is None or not callable(
                    getattr(pipeline, "predict", None)
                ):
                    raise TypeError(
                        "The artifact must contain a model with predict()."
                    )

                embedded_metadata = {
                    key: value
                    for key, value in artifact.items()
                    if key != "model"
                }

                saved_metadata = self._read_metadata(
                    regression_metadata
                )

                self.pipeline = pipeline
                self.metadata = {
                    **embedded_metadata,
                    **saved_metadata,
                }
                self.scoring_mode = "colab_regression"
                self.artifact_filename = regression_path.name
                return

            except Exception as exc:
                errors.append(
                    f"{regression_path.name}: "
                    f"{type(exc).__name__}: {exc}"
                )

        # Optional backward compatibility with the older classifier.
        classifier_path = (
            self.model_dir / "prompt_preference.joblib"
        )

        if classifier_path.is_file():
            try:
                pipeline = joblib.load(classifier_path)

                if not callable(
                    getattr(pipeline, "predict_proba", None)
                ):
                    raise TypeError(
                        "The classifier must expose predict_proba()."
                    )

                self.pipeline = pipeline
                self.metadata = self._read_metadata(
                    self.model_dir / "metadata.json"
                )
                self.scoring_mode = "bpo_preference_classifier"
                self.artifact_filename = classifier_path.name
                return

            except Exception as exc:
                errors.append(
                    f"{classifier_path.name}: "
                    f"{type(exc).__name__}: {exc}"
                )

        self.load_error = "; ".join(errors) or (
            "No compatible model found. Expected "
            "prompt_quality_analyzer_model.joblib or "
            "prompt_preference.joblib in the model directory."
        )

    @property
    def available(self) -> bool:
        return self.pipeline is not None

    def predict_score(self, prompt: str) -> float | None:
        if not self.available:
            return None

        try:
            if self.scoring_mode == "colab_regression":
                # Ridge regression predicts a numerical score directly.
                raw_score = float(
                    self.pipeline.predict([prompt])[0]
                )
            else:
                # Legacy classifier: class 1 means optimized-style.
                probabilities = self.pipeline.predict_proba([prompt])[0]
                classes = list(
                    getattr(self.pipeline, "classes_", [])
                )

                index = (
                    classes.index(1)
                    if 1 in classes
                    else min(1, len(probabilities) - 1)
                )

                raw_score = float(probabilities[index]) * 100.0

            # Keep the displayed score inside the promised 0–100 range.
            bounded_score = max(0.0, min(100.0, raw_score))
            return round(bounded_score, 1)

        except Exception:
            # The API will still be able to use the rubric fallback.
            return None

    # Backward compatibility with older code and tests.
    def preference_score(self, prompt: str) -> float | None:
        return self.predict_score(prompt)

    def public_info(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "scoring_mode": self.scoring_mode,
            "artifact_filename": self.artifact_filename,
            "version": self.metadata.get(
                "model_version", "unavailable"
            ),
            "model_type": self.metadata.get(
                "model_type", "unknown"
            ),
            "training_data": self.metadata.get(
                "training_data", "unknown"
            ),
            "label_semantics": self.metadata.get(
                "label_semantics", "unknown"
            ),
            "metrics": self.metadata.get("metrics", {}),
            "limitations": self.metadata.get("limitations", []),
            "load_error": (
                self.load_error if not self.available else None
            ),
        }

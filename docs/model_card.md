# Model Card: Prompt Quality Analyzer

## Model variants

### Included bootstrap artifact — `pqa-seed-synthetic-v0.1`

- **Type:** word + character TF-IDF features, logistic regression classifier.
- **Training data:** 120 manually specified synthetic contrastive task variants, expanded to 240 prompt texts. These examples are authored in `backend/train_seed_model.py`.
- **Target:** distinguish a weak/original-style seed prompt (`0`) from an explicitly structured seed prompt (`1`).
- **Reported holdout metrics:** the training script performs a pair-grouped split; the resulting current metadata contains 1.0 accuracy, precision, recall, F1 and ROC-AUC. This result reflects the cleanly templated synthetic examples and should be treated as a warning about an easy artificial test—not as a real-world quality metric.
- **Status:** development bootstrap only; `production_ready` is `false`.

### Colab-trained candidate — `pqa-bpo-pair-preference-v1.0`

The notebook fits a word-TF-IDF baseline and word+character TF-IDF model on BPO pairs, selects with validation metrics, refits on train+validation, tests once on the held-out test set, and exports a `joblib` pipeline plus metadata. This artifact is not included as an executed output because this environment cannot download the dataset. Its metrics are available only after the notebook is actually run.

## Intended use

Provide a weak auxiliary signal to a prompt-analysis product, highlighting that a prompt's clarity, constraints, task definition and other features may be improved. The model is not a substitute for human evaluation and does not predict the quality or factuality of any particular LLM's response.

## Not intended for

- Claiming a universal objective 0–100 prompt-quality truth.
- Hiring, grading, admissions or high-stakes decisions.
- Automated enforcement of a user's intentions or policy compliance.
- Predicting how a specific commercial model will answer without direct downstream evaluation.
- Non-English scoring until separately evaluated.

## Score composition in the API

The service computes a documented ten-dimension heuristic rubric and combines it with the model's optimized-style preference signal: 80% rubric + 20% learned signal. These weights are engineering choices, not validated optimal weights. The rubric's dimension weights are configurable in `backend/app/services/scoring.py` and documented at `/model-info`.

## Metrics and evaluation scope

The seed artifact's classification report only reflects synthetic split discrimination. The BPO notebook's test metrics reflect original-vs-optimized labels in the BPO distribution. Neither establishes calibration against human prompt-quality ratings. A final release should add an independent human-reviewed dataset and evaluate inter-rater agreement, calibration, task slices, language slices, and stability to superficial paraphrases/filler.

## Ethical considerations and risks

- The model may favor formatting or verbosity patterns rather than intent-aware quality.
- Short factual questions can be effective without explicit output format, audience or detailed constraints; the rubric includes context-aware defaults to mitigate but not eliminate this issue.
- A rewrite could change the intended scope or introduce assumptions. Review before use.
- Optional hosted rewriting sends prompt text to a configured external provider; this is opt-in in the UI and the API key remains backend-side.
- The rule-based fallback must be labeled honestly as deterministic, not an LLM output.

## Limitations

Prompt quality depends on task, user, context, language, model and desired answer style. There is no universal score; numeric values are an estimate. A rewrite score increase is not proof of better real-world output, and the app does not force every rewrite to score higher.

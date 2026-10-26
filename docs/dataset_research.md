# Dataset Research and Selection Report

**Research snapshot:** 24 October 2026. Dataset viewer counts and licenses can change; always re-check the linked dataset card before redistributing data or model artifacts.

## Decision summary

**Primary dataset: [zai-org/BPO — Black-box Prompt Optimization](https://huggingface.co/datasets/zai-org/BPO).** It is the closest direct fit because it contains original prompts paired with optimized prompts. The current Hugging Face card lists 14,395 rows (train 13,895, validation 300, test 200), fields `prompt`, `optimized_prompt`, `good_res`, and `bad_res`, English language, and Apache-2.0 license. The original paper is [Black-Box Prompt Optimization: Aligning Large Language Models without Model Training](https://arxiv.org/abs/2311.04155).

**Crucial limitation:** BPO does not provide expert-calibrated 0–100 prompt-quality labels. The optimizer used a research pipeline that generated optimized prompts from feedback data, and the authors note that optimization quality is not guaranteed. We therefore use the original/optimized pair side as a **relative preference proxy** (`0 = original`, `1 = optimized`), not as universal ground truth. The separately implemented rubric supplies the ten human-readable dimensions. The final score is explicitly described as an estimate.

## Candidate comparison

| Dataset | Publisher / size | Relevant fields and labels | License as listed by source | Suitability and limitation |
|---|---|---|---|---|
| **BPO** | Z.ai; 14,395 prompt-optimization pairs | Original `prompt`, `optimized_prompt`, `good_res`, `bad_res`; pair status, not an absolute quality score | Apache-2.0 | Best direct fit for training a lightweight model to identify BPO-style optimized prompts and for learning rewrite patterns. English-only; optimization outputs may contain imperfections. |
| [OpenAssistant OASST1](https://huggingface.co/datasets/OpenAssistant/oasst1) | OpenAssistant; dataset paper reports 161,443 messages and 461,292 quality ratings; current Hugging Face viewer exposes a train split around 84.4k rows for its current export | Conversation message text, `role`, `lang`, `rank`, `review_result`, `labels`, and conversation-tree IDs. Quality annotations primarily concern conversation/message quality, not direct prompt-quality scores | Apache-2.0 | Helpful for later multilingual conversational studies and user/assistant message context, but not suitable as the sole source of prompt-quality labels. Count differences reflect the release/card/export representation; inspect the exact snapshot used. Paper: [2304.07327](https://arxiv.org/abs/2304.07327). |
| [Anthropic HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf) | Anthropic; current Hugging Face card describes 169,352 rows, about 94.7 MB | `chosen` and `rejected` response pairs tied to a conversation prefix; red-team subset includes fields such as `task_description_harmlessness_score`, which is a harmlessness measure, not overall prompt quality | MIT listed on the Hugging Face card; verify per subset and terms | Strong research source for response preferences and safety-oriented studies, but response preference is not direct prompt quality. Contains potentially upsetting content; unnecessary for this first model. Paper: [2204.05862](https://arxiv.org/abs/2204.05862). |
| [Microsoft PromptBench](https://github.com/microsoftarchive/promptbench) | Microsoft Research archive; benchmark framework rather than one single prompt-quality-labelled dataset | Tests prompt engineering, task robustness and adversarial prompt variations on datasets such as GLUE, MMLU and others | Framework source has a license file; constituent datasets have their own licenses and terms | Useful for robustness/benchmark methodology, not for direct 0–100 quality targets. Do not assume all bundled dataset licenses are the same. |
| [OpenAI Model Spec Evals](https://alignment.openai.com/model-spec-evals) | OpenAI; described as 596 prompts across 225 focus areas | Policy-oriented scenarios with rubrics defining expected model behavior | Check each artifact's reuse terms before incorporating it | Good example of rubric-based evaluation, but designed for model-spec compliance rather than general prompt quality. Not used for model training here. |

## Why BPO was chosen

1. It supplies prompt pairs that already express an intended relationship between an input and a rewritten version.
2. Its schema lets the training notebook retain official train/validation/test splits without manufacturing quality scores.
3. The pair data can train a compact TF-IDF + logistic-regression classifier that is cheap to serve and easy to explain.
4. The same pair data can inform the rewrite-service design without treating `good_res` or `bad_res` as prompt-quality labels.

## Proposed training label and evaluation unit

Each row is expanded into two text examples: original prompt with label `0`, optimized prompt with label `1`. The notebook compares word TF-IDF against combined word/character TF-IDF, chooses using validation F1 (ROC-AUC as tie-breaker), then evaluates once on the held-out test set. Both prompt sides from a pair remain in that pair's original split. It additionally excludes validation/test pairs containing an exact normalized prompt string found in training. This does not fully remove semantic near-duplicates.

Metrics therefore answer: *How well does this model identify which side of a known BPO pair a prompt came from?* They do **not** answer: *How accurate is the model's 0–100 score against human judgments?* That second question requires a separate expert/human-rated test set.

## Download and storage

Recommended acquisition in Colab:

```python
from datasets import load_dataset
bpo = load_dataset("zai-org/BPO")
```

The source currently lists approximately 24.4 MB for the dataset files and 14,395 rows. A loader may cache the source locally; allow additional space for temporary files, decoded data, vectorizers and model artifacts. The notebook does not commit the original dataset into this Git repository.

## Licensing and provenance rules

- Preserve the source URL, publisher, paper and license label in metadata.
- Re-check current license terms at time of use, especially before commercial redistribution.
- Do not mix original BPO records with generated or hand-authored examples without a source/provenance field.
- Do not present BPO pair status as human-certified quality.
- Keep any future manually rated evaluation set separately versioned and document its rating rubric, raters, agreement and splits.

## Sources

- BPO card: https://huggingface.co/datasets/zai-org/BPO
- BPO paper: https://arxiv.org/abs/2311.04155
- OASST1 card: https://huggingface.co/datasets/OpenAssistant/oasst1
- OASST1 paper: https://arxiv.org/abs/2304.07327
- HH-RLHF card: https://huggingface.co/datasets/Anthropic/hh-rlhf
- HH-RLHF paper: https://arxiv.org/abs/2204.05862
- PromptBench repository: https://github.com/microsoftarchive/promptbench
- Model Spec Evals: https://alignment.openai.com/model-spec-evals

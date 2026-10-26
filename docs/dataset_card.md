# Dataset Card: Prompt Quality Analyzer Training Data

## Primary dataset

- **Name:** Black-box Prompt Optimization (BPO)
- **Publisher:** Z.ai (dataset is hosted at Hugging Face)
- **Source:** https://huggingface.co/datasets/zai-org/BPO
- **Paper:** https://arxiv.org/abs/2311.04155
- **License as listed by the current card:** Apache-2.0
- **Language:** English
- **Current listed size:** 14,395 rows: 13,895 train, 300 validation, 200 test
- **Fields:** `prompt`, `optimized_prompt`, `good_res`, `bad_res`
- **Use in this project:** only `prompt` and `optimized_prompt` are converted into original-vs-optimized classifier examples. Response fields are not used as prompt-quality labels.

## Intended use

Training and evaluating a compact classifier that estimates whether a prompt resembles the optimized side of a BPO pair. It is used as a modest auxiliary signal alongside a separate explainable heuristic rubric.

## Labels and semantics

- `0`: original prompt side from a dataset pair.
- `1`: optimized prompt side from a dataset pair.

These labels are dataset roles, not objective prompt-quality judgments. They are not interchangeable with ratings on a universal 0–100 scale.

## Processing

1. Download with `datasets.load_dataset("zai-org/BPO")`.
2. Validate required columns and record the split schema.
3. Remove null/empty prompts, exact identical sides of a pair, and exact duplicate pairs.
4. Preserve the dataset's named train/validation/test split.
5. Remove evaluation pairs that contain exact normalized prompt strings found in the training split.
6. Expand pairs into separate prompt texts with labels 0 and 1.
7. Train only on the training split; select the model using validation F1 (ROC-AUC as a tie-breaker); evaluate on test once.

## Known limitations and biases

- Optimized prompts were generated in a research procedure, not independently expert-rated on a universal prompt-quality rubric.
- Some optimized variants may be imperfect or overly verbose.
- English-only data may fail on Hindi, Hinglish, multilingual or domain-specific prompts.
- Common optimization patterns and length/format cues may act as shortcuts.
- Exact overlap filtering does not detect all paraphrases or near-duplicates.
- A high test score on original-vs-optimized classification does not validate general prompt-quality scoring.

## Privacy and safety

The project does not include prompt-history persistence. Dataset contents are downloaded from the public source. Follow source license and institutional data-use requirements; inspect records before producing any public demonstration set.

## License handling

The current Hugging Face card lists Apache-2.0. Verify the license from the upstream dataset at the date of use and keep provenance in the exported `metadata.json`. This card is a project note, not legal advice.

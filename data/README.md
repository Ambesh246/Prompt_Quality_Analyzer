# Data folder

No downloaded raw dataset is committed here. The training notebook downloads BPO at runtime:

- Dataset: https://huggingface.co/datasets/zai-org/BPO
- Paper: https://arxiv.org/abs/2311.04155
- License as currently listed: Apache-2.0 (verify before redistribution)

The notebook stores downloaded data in the Colab dataset cache, then exports only the trained `joblib` pipeline, model metadata and a test-prediction CSV. Do not commit raw datasets, private prompt collections, or any dataset with unknown licensing. See `docs/dataset_card.md` and `docs/dataset_research.md` for label semantics and processing.

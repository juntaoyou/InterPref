# InterPref

InterPref is a three-stage pipeline (induction -> causal intervention ->
cross-validation) for preference-related internal states in large language
models, together with its evaluation and analysis utilities.

## Repository layout

| Path | Contents |
|---|---|
| [`code/`](code/README.md) | InterPref pipeline (induction, localization, intervention, generation, judging, perplexity, analysis) and shared libraries |
| [`dataset/`](dataset/README.md) | Psoups and MMLU query pools used for localization and evaluation |

Script names use a two-digit stage prefix (`NN_<stem>.<ext>`); see
[`code/README.md`](code/README.md) for the stage map and runnable examples.
Paths such as `<REPO_ROOT>` and `<MODELS_DIR>` are placeholders for local
directories and must be set before running the scripts.

## Requirements

Python 3, `torch`, `transformers`, `vllm`, `datasets`, `numpy`, `pandas`, and
an LLM-as-a-judge endpoint (see `code/README.md`).
Model checkpoints and judge models are not included in this repository.

## License

MIT — see [LICENSE](LICENSE).

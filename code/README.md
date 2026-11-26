# InterPref pipeline code

Implementation of the InterPref procedure (induction -> causal intervention ->
cross-validation) plus the evaluation, perplexity and analysis utilities used
with it. Scripts are copied from a local working directory (`<REPO_ROOT>/code`).

## Naming convention

Every script carries its pipeline stage as a **two-digit prefix** followed by the
descriptive stem (lowercase snake_case; variant tokens such as `2`, `3` or
`_0818` are preserved where they distinguish different runs):

```
NN_<stem>.<ext>        e.g.  10_search.py,  10_search.sh,  40_deactivate.py
```

A shell driver keeps the stem of the Python script it launches (`40_deactivate.py` ↔
`40_deactivate.sh`), so a stage can be traced from prefix alone.

| Prefix | Stage |
|---|---|
| `01_` | setup / utilities (dependency install, JSON-key repair, missing-file generation, data supply) |
| `10_` | single-preference induction (activation-probability collection) |
| `20_` / `21_` | localization: Δp computation and top-θ selection / θ selection |
| `30_` | multi-preference induction (pair templates) |
| `40_` / `41_` | deactivation (necessity): single-preference / dose–response multi |
| `50_`–`54_` | amplification (sufficiency) and single-preference generation |
| `60_` | joint steering under the co-induction prompt |
| `70_`–`72_` | neutral baseline generation (incl. the instruction-order swapped variant) |
| `80_` | LLM-as-a-judge scoring (primary: `80_llm_score_with_ci.py`) |
| `81_` / `82_` | evaluation drivers / overlap-scoring drivers |
| `85_` | judge robustness (alternative judge) |
| `86_` | perplexity (response-token PPL) |
| `90_` | run orchestration — generation/evaluation chains and special runs |
| `95_` | statistics / consistency analysis |

Shared importable modules carry the `lib_` prefix and no stage number
(`lib_instructions.py`, `lib_data_utils.py`), matching the project's shared
helper modules. `dataset_tools/` keeps its own namespace and clean snake_case;
the two-digit prefixes make the renamed scripts self-describing.

## Pipeline map

| Stage | Script(s) | Notes |
|---|---|---|
| **Preference induction** — preference/neutral prompts, activation collection | `10_search.py` (+ `10_search.sh`) | single prompt template, one `.pt` per condition with token-level gate-activation statistics |
| ″ variant used for the Qwen runs | `10_search2.py` (+ `10_search2.sh`) | output names carry the model suffix (`<pref>.Qwen3-8B.pt`); reads the query pool through `--dataset` |
| **Localization** — Δp = p_P − p_None, top-θ selection | `20_process_with_one.py` (args: `--preference --delta --top_percent`), `20_process_with_one2.py` | Δp against the **neutral (None)** run; emits `_sorted_differences_<δ>.txt`, `_dict_<δ>.json`, `_top_<θ>percent.txt` |
| **θ selection** ("smallest value that preserved generation fluency") | `21_select_theta.py` | reads a dose–response CSV supplied via `--sweep`; finds the smallest rho > 0 with fluency within `--tolerance` of the rho = 0 baseline |
| **Multi-preference induction** (pairs, canonical order) | `30_search_multi.py`, `30_search_multi2.py` | two-preference template, "equally expressed" |
| **Deactivation (necessity)** — clamp under preference prompt | `40_deactivate.py` / `40_deactivate2.py` (+ `.sh`) | `--deactivate preference_neurons` or `random` (matched-size random-neuron control); selected neurons **clamped to zero** |
| **Dose–response** (deactivate 2–20 %) | `41_deactivate_multi.py` / `41_deactivate_multi2.py`, `95_run_t_test_deactivate_vary.sh` | fraction sweep with paired statistics |
| **Amplification (sufficiency)** — scaled up under neutral prompt | `50_generation_with_one.py/2/3`, `51_generation_with_one_overlap.py`, `52_generation_multi.py`, `54_p2_multi.py` (+ wrappers) | `a_j → a_j · (1 + val)` — pure scaling, no gating/weighting/dropout; runs under `###Preference requirement:None` |
| **Joint steering** (co-induction prompt) | `60_with_preferences_generation_multi.py`, `60_with_preferences_generation_multi_multi_single.py`, `53_generation_multi3.py` | amplifies the target sets and deactivates the opposite-pole sets; set construction `union / both / overlap / partial1 / partial2` preserved |
| **Neutral baseline generation** | `70_baseline_generation.py`, `71_baseline_generation_multi.py` | matched neutral reference for every contrast |
| **Instruction-order control** (canonical vs reversed) | `72_baseline_generation_multi_swapped.py`, `psoups_swapped` scoring scripts | ordering-robustness check |
| **Cross-dataset overlap transfer** (amplify the intersection) | `51_generation_with_one_overlap.py`, `82_score_overlap_*.sh`, `95_multi_neuron_analysis.py`, `95_neuron_layer_iou.py` | IoU quantification and the intersection-amplification test |
| **LLM-as-a-judge evaluation** (preference + fluency scores, 3 runs) | `80_llm_score_with_ci.py` (primary), `80_llm_score.py/2/memfix`, `80_eval_with_preferences.py` (+ `80_llm_score*.sh`, `81_evaluate*.sh`) | judge `Qwen2.5-72B-Instruct-AWQ`; rubrics in `lib_instructions.py`; `num_runs=3` |
| **Judge robustness** (alternative judge) | `85_compute_judge_robustness.py` | re-scoring with a Llama judge, agreement statistics |
| **Perplexity** | `86_ppl2.py` + `86_compute_ppl_collected.py` (+ `86_ppl.py`, `86_ppl.sh`) | response-token PPL = exp(mean NLL) per query; base models same family + scale as each generator (Qwen3-8B-Base / Llama-3.1-8B / gemma-2-9b); no outlier filtering by default |
| **Statistics / consistency analysis** | `95_t_test_analysis.py`, `95_analyze_t_test_results.py`, `95_neuron_cosine_similarity.py`, `95_compare_delta_vs_occurence.py`, `95_nlp_analysis.py`, `95_plot_iou.py` | paired t-tests, cross-model correlation, activation-probability similarity, NLP diagnostics |
| **Run orchestration** | `90_run_multimodel_0818.sh` → `90_gen_llm_0818.sh`, `90_eval_llm_0818.sh`; `90_run_all_models_datasets.sh`; `81_run_qwen_eval_0818.sh`; `90_gen_*/90_eval_*` wrappers | model -> generation -> judging chains with their hyper-parameter values |
| **Dataset construction** | `dataset_tools/build_mmlu_eval_570.py`; verification in `../dataset/verify_datasets.py` | `--mode reproduce` rebuilds the shipped 570-query file byte-for-byte; `--mode random --seed S` implements seeded stratified random sampling |

Shared modules: `lib_instructions.py` (instruction templates and judge rubrics) and
`lib_data_utils.py` (query-pool loader for the packaged JSONL/CSV *and* the original HF
`save_to_disk` directories).

## Environment and inputs

> Real paths are masked (see the top-level README): `<MODELS_DIR>`, `<REPO_ROOT>`
> etc. stand for local directories; `../dataset/*.jsonl` ships with this
> package and needs no substitution.

```bash
source <REPO_ROOT>/.venv/bin/activate   # vllm + transformers + torch + datasets
# models:                    <MODELS_DIR>/{Qwen3-8B, Llama-3.1-8B-Instruct, gemma-2-9b-it}
# judge:                     <MODELS_DIR>/Qwen2.5-72B-Instruct-AWQ
# query pools:               ../dataset/*.jsonl   (defaults wired into 10_search*.py)
```

Example — induction + localization on the packaged Psoups pool:

```bash
python 10_search.py --model <MODELS_DIR>/Qwen3-8B \
    --answer_path out/answer --neuron_path out/neuron --dataset ../dataset/psoups_localization_2525.jsonl
python 20_process_with_one.py --neuron_dir out/neuron --out_dir out/with_one \
    --preference Expertise_high --delta 0.1 --top_percent 10
```

Example — judging with the primary judge (3 runs):

```bash
python 80_llm_score_with_ci.py --model <MODELS_DIR>/Qwen2.5-72B-Instruct-AWQ \
    --input_path <generation_dir> --out_path <results_dir> --principle expert --batch_size 4
```

## Generation / evaluation parameters

| Stage | max_tokens | temperature | repetition_penalty | source script |
|---|---|---|---|---|
| Induction | 400 | 0 | 1.1 | `10_search*.py`, `30_search_multi*.py` |
| Deactivation / amplification / baseline generation | 400 | 0 | 1.15 | `40_deactivate*.py`, `41_deactivate_multi*.py`, `50_generation_with_one*.py`, `70_baseline_generation*.py`, `71_baseline_generation_multi*.py` |
| Steering (with_preferences) | 400–500 | 0 | 1.15 | `60_with_preferences_generation_multi.py` |
| PPL | — | — | — | `86_ppl2.py` (greedy generations re-scored) |

These values are tabulated here for reproducibility.

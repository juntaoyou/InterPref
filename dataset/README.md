# Query datasets

The four query sets used by the InterPref pipeline. Every file ships as JSONL
(one `{"query": ...}` object per line, the format the pipeline scripts read) and
as CSV (`query_id`, `query`).

`verify_datasets.py` re-checks sizes, uniqueness, disjointness, split membership
and per-category stratification against the upstream public sources.

| File | Role | Size | Format |
|---|---|---|---|
| `psoups_localization_2525.jsonl` / `.csv` | Psoups, Train Set — Localization ("Deduplication of raw queries to ensure uniqueness") | 2,525 rows, 2,525 unique | JSONL / CSV |
| `psoups_eval_koala_49.jsonl` / `.csv` | Psoups, Koala Eval Set — Evaluation (In-Dist.) ("Real users' open-ended queries from the Koala Evaluation Set") | 49 rows, 49 unique | JSONL / CSV |
| `mmlu_localization_2525.jsonl` / `.csv` | MMLU, Train Set — Localization (OOD) ("Random sampling to match the Psoups pool size") | 2,525 rows, 2,523 unique | JSONL / CSV |
| `mmlu_eval_570_stratified.jsonl` / `.csv` | MMLU, Test Set — Evaluation (OOD) ("10 queries per category across all 57 categories") | 570 rows, 569 unique | JSONL / CSV |

## Provenance

* **Psoups (Personalized Soups)** — the open-source `lFelix/p_soups` training set
  (291,415 prompt–response pairs). The localization pool is the deduplicated set of its
  2,525 unique user queries; the 49 evaluation queries are the Koala-evaluation queries kept
  by the original Personalized Soups benchmark. The two sets are disjoint (no leakage).
* **MMLU (Hendrycks et al.)** — the localization pool is 2,525 queries drawn from the MMLU
  *training* set (`auxiliary_train`, 99,842 questions), sized to match the Psoups pool;
  none of its queries appear in the dev, validation or test splits. The evaluation set is
  570 test-set queries, **10 per subject across all 57 subjects**. Localization and
  evaluation therefore use different subsets.

## Reproducing

`verify_datasets.py` re-derives all of the above (sizes, uniqueness, disjointness, split
membership, per-category stratification) from the upstream sources:

```bash
python verify_datasets.py                       # default upstream locations
python verify_datasets.py --mmlu_dir /path/to/mmlu --psoups_dir /path/to/pSoups
```

The upstream sources themselves are public and are not redistributed here:
`hendrycks/test` (MMLU) and `lFelix/p_soups` (Personalized Soups).

## Field notes (read before citing a count)

1. **Row count vs. unique text.** `mmlu_localization_2525` contains 2,525 rows but 2,523
   unique query strings: two questions ("Which is true?" and one biology ordering question)
   occur twice *in the MMLU auxiliary-train source itself*, and both copies were sampled.
   `mmlu_eval_570_stratified` contains 570 rows but 569 unique strings because one question
   text appears in the first-10 block of two different subjects. Counts below are row counts.
2. **MMLU evaluation sampling.** The 570-query evaluation set was built by taking the
   **first 10 rows of each of the 57 subject test files** (deterministic, in the original
   MMLU file order). The construction script (`../code/dataset_tools/build_mmlu_eval_570.py`)
   implements both: `--mode reproduce` rebuilds the shipped file byte-for-byte (the queries
   every MMLU experiment ran on), and `--mode random --seed S` implements seeded stratified
   random sampling for fresh re-runs.
3. **MMLU localization sampling.** The 2,525-query pool is a random draw from
   `auxiliary_train`, but the generating script/seed was not preserved in the repository,
   so the pool ships as data and is verified by membership rather than regenerated.
4. **Template note.** All pipeline stages read these files' `q`/`query` field verbatim into
   the prompt template `###Question:{question}`; the identical template is used for both
   datasets.

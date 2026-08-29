# Spark reference pipeline (maintained Python baseline)

Maintained copy of the collaborative-filtering part of the original PySpark
pipeline (`original_code/python/`, archived and frozen — see
`archive/original_code.sha256`). This is the source of candidate pairs, test
fixtures, and golden Pearson similarities for the native C++/CUDA backends.

Only the CF scripts are maintained here. The MinHash/LSH pair finder
(`task1.py`) and the TF-IDF content model (`task2train.py` /
`task2predict.py`) are application context and stay archive-only: LSH
candidate generation for user-based CF is already built into `cf_train.py`,
and the exported fixtures freeze its output, so no native backend ever needs
that code.

The scripts here are byte-identical to the originals except for:

- restored `sys.argv` command-line arguments (the originals ran with
  hardcoded local paths),
- neutral Spark app names,
- neutral file names.

No algorithmic behavior was changed. Known defects are documented below and
intentionally left in place until they are covered by tests.

## File mapping

| Maintained name | Original          | Purpose                                                        |
| --------------- | ----------------- | -------------------------------------------------------------- |
| `cf_train.py`   | `task3train.py`   | Pearson similarity model, item- or user-based — **the kernel**  |
| `cf_predict.py` | `task3predict.py` | Rating prediction from a CF model                              |

Archive-only (not maintained): `task1.py`, `task2train.py`,
`task2predict.py`, plus the scratch files `test1.py` / `test2.py`.

## Environment

Pinned in `requirements.txt`: Python 3.11 + PySpark 3.5.9 on OpenJDK 17
(the original homework targeted Python 3.6 / Spark 2.3.2; the RDD API used
here is unchanged between the two).

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r spark_pipeline/requirements.txt
export JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home
```

## Run commands

Data lives in `original_code/python/data/` (read-only). Outputs should go to
a working directory outside the archive, e.g. `results/`.

```bash
DATA=original_code/python/data
OUT=results

# The Pearson kernel source (primary workload for the native backends):
spark-submit spark_pipeline/cf_train.py $DATA/train_review.json $OUT/cf_item.model item_based
spark-submit spark_pipeline/cf_train.py $DATA/train_review.json $OUT/cf_user.model user_based

# cf_predict expects user_avg.json / business_avg.json next to the train file.
spark-submit spark_pipeline/cf_predict.py $DATA/train_review.json $DATA/test_review.json \
    $OUT/cf_item.model $OUT/cf_item_predict.json item_based
```

## Known defects (documented, not yet fixed)

1. `cf_predict.py` (user_based) — the denominator uses
   `abs(user_avg)` instead of `abs(sim)`.
2. `cf_predict.py` (user_based) — the output JSON swaps `user_id` and
   `business_id` (test data is keyed `(business_id, user_id)` in this branch).
3. `cf_train.py` — `user_data`/`business_data` come from
   `distinct().collect()`, whose order is not deterministic across runs, so
   `user_idx`/`business_idx` (used by user-based LSH hashing) can vary.

Archive-only scripts have their own defects (`task2train.py` user-profile
slice and hardcoded rare-word threshold); they are out of scope here.

Do not "fix" the Pearson similarity functions to compensate for these; the
similarity contract in `cf_train.py` (`item_pearson_similarity` /
`user_pearson_similarity`) is the frozen reference for all native backends.

## Golden outputs from the original run

The archive already contains model files produced by the original code from
the same dataset:

- `original_code/python/result/task2_model_item` — 557,684 item pairs
- `original_code/python/result/task2_model_user` — 624,898 user pairs

(The `task2_model_*` names are historical; they are Task 3 CF models.)

These serve as golden similarity results until the maintained pipeline
regenerates them reproducibly.

## Reproduction check (2026-08-26)

`cf_train.py ... item_based` was re-run on the pinned environment (37 s,
Apple Silicon, local[*]) and compared against the archived golden model:

- 557,640 pairs identical in both, max abs similarity diff **7.8e-16**.
- 97 pairs disagree on membership (44 only in golden, 53 only in new); all
  of them have |sim| <= 1.2e-16 — mathematically zero correlations that land
  on either side of the `sim > 0` filter depending on floating-point
  summation order (co-rated users are iterated from a Python set, whose
  order depends on randomized string hashing).

Consequences for the native backends: the correctness contract must use a
tolerance (e.g. abs diff <= 1e-12) and must treat the `sim > 0` output
filter as `sim > eps` with a declared eps, or compare pair sets modulo
near-zero similarities.

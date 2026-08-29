# Backend-neutral Pearson workload fixtures

Produced by `spark_pipeline/export_fixture.py`; format documented in each
fixture's `meta.json` and in the exporter docstring. Contract:
`docs/pearson_contract.md`. Validate any fixture without Spark:

```bash
python3 tools/validate_fixture.py data/fixtures/<name>
```

| Fixture       | Source                              | Entities x dims  | Ratings | Candidate pairs | Emitted |
| ------------- | ----------------------------------- | ---------------- | ------- | --------------- | ------- |
| `item_tiny`   | 50 most-reviewed businesses         | 50 x 14,317      | 35,656  | 1,223           | 889     |
| `item_medium` | businesses with sorted idx % 4 == 0 | 2,564 x 24,853   | 123,413 | 74,904          | 35,744  |
| `item_full`   | full train_review.json              | 10,253 x 26,184  | 488,560 | 1,171,857       | 557,478 |
| `user_full`   | full train_review.json (LSH pairs)  | 26,184 x 10,253  | 488,560 | 1,411,864       | 634,993 |

`microcases.json` holds the hand-computed contract cases (not a workload
fixture; consumed by `tools/pearson_reference.py` and native unit tests).

Validation status (2026-08-26): all four fixtures pass full-pair validation
with max abs diff 0.0. `item_full` cross-checked against the archived Spark
model (`tools/crosscheck_archived.py`): all 557,478 emitted pairs match
within 1e-15; the 206 archived-only pairs are near-zero sims below the
contract's EPS filter.

The `golden.bin` similarity is stored for every candidate pair unfiltered,
so backend comparison is by value, never by set membership.

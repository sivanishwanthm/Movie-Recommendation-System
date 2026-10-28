# Phase 5 verification: experimentation infrastructure

Phase 5 is implemented with **live enrollment disabled by default**. No real randomized online experiment was run, and no real engagement improvement is claimed. The delivered comparison is popularity vs live-profile item CF; an ID-based historical neural model is not falsely presented as a valid new-user treatment.

## Verified implementation

- Immutable versioned definitions, pinned artifact/catalog, explicit enrollment schedule and prospective assumptions.
- Deterministic SHA-256 profile assignment with persisted uniqueness and concurrent enrollment protection.
- Durable assignment/request denominators before scoring; recorded failure responses and incomplete-telemetry gates.
- First-visible experiment exposure, idempotent impression tracking and owned-batch click attribution.
- Mature-profile intention-to-treat analysis, fixed follow-up, exact SRM, Wilson/Newcombe intervals and Fisher exact test.
- Power planning; profile-level latency and error guardrails; descriptive catalog coverage.
- Explicit operator-only aggregate analysis, opt-in activation and no automatic winner/promotion.
- File-only synthetic diagnostics with separately labeled UI/API sources.
- Additive transactional PostgreSQL schema v2 migration preserving Phase 4 records.
- Next.js experiment dashboard, production build and TypeScript validation.

## Test results

| Check | Result |
|---|---|
| Full Python suite | 45 passed, no skips; 14 existing Matplotlib/Pyparsing deprecation warnings |
| Database | Real PostgreSQL 18.4, isolated test schemas |
| Migration | v1 to v2 preserved existing movie rows; rerun idempotent |
| Assignment | Repeatable, stable across sessions and concurrent requests; closed windows do not re-enroll |
| Model/definition compatibility | Changed immutable definitions rejected; mismatched model startup rejected |
| Execution | Both control and treatment branches exercised; failed scorer retained assignment and error telemetry |
| Attribution | Impression exposure deduplicated; out-of-window and unrelated-batch clicks excluded |
| Inference | Known numeric references, sparse/zero cells, maturity, SRM, incomplete telemetry and final-review gates tested |
| Synthetic reproduction | Two complete seeded simulation artifacts matched exactly |
| Next.js build | Passed, including `/experiments` |
| TypeScript | Passed |
| HTTP integration | Both variants exercised through Next.js → FastAPI → PostgreSQL, with operator access and immature-window checks |
| Visual browser QA | Not performed; Computer Use permission unavailable |

The HTTP test used generated QA profiles and ratings in a dedicated `phase5_http_qa` schema. It did not use actual human outcomes or mix generated events into the application's public tables. That QA schema was removed after verification. See [HTTP check results](http-smoke.json).

## Synthetic diagnostics — not live results

Every outcome, latency and candidate ID below is generated. The injected-effect scenario deliberately changes its treatment probability; it does not demonstrate effectiveness of any trained recommendation model.

| Generated scenario | Treatment − control conversion | Two-sided Fisher p-value | SRM flag |
|---|---:|---:|---|
| Equal-rate null | +0.001311 | 0.841776 | No |
| Injected +0.03 probability change | +0.036721 | 1.280e-08 | No |
| Deliberately corrupted 80% treatment allocation | -0.011542 | 0.129101 | **Yes** |

Each scenario uses 10,000 generated profiles with seed 42. The allocation-fault scenario is flagged against the expected 50/50 split; it must not be interpreted for treatment effect. All synthetic reports have `review_eligible=false` regardless of p-values.

Repeated null calibration: **19 / 500 rejected (3.8%)**, with Monte Carlo 95% Wilson interval **[2.45%, 5.86%]**. Each repetition generates 1,000 independent profiles per arm with the same 0.1 conversion probability (seed 31415). The interval encompasses the nominal 5% level. This validates only this fixed-sample Bernoulli diagnostic, not live data, repeated peeking, production identity or the entire experimental process. Full scenario intervals and assumptions are in [simulation.json](simulation.json).

## Prospective sample-size example

Assumed control conversion **10%**, absolute detectable change **2 percentage points**, power **80%**, two-sided alpha **5%**, 50/50 allocation: **3,841 mature profiles per arm**, **7,682 total** by normal approximation. These are assumed planning inputs, not observed MovieLens clicks or a live conversion estimate. No traffic-volume estimate or promised completion date is given. See [sample-size-plan.json](sample-size-plan.json).

## Commands verified

```bash
TEST_DATABASE_URL='postgresql+psycopg://recsyslab@127.0.0.1:55432/postgres' \
MPLCONFIGDIR=/tmp/recsyslab-mpl PYTHONPYCACHEPREFIX=/tmp/recsyslab-pycache \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
.venv/bin/python -m pytest -q

.venv/bin/python -m recsyslab.experimentation.simulate --output reports/phase5/simulation.json
.venv/bin/python -m recsyslab.experimentation.simulate --output /tmp/recsyslab-phase5-repeat.json
.venv/bin/python -m recsyslab.experimentation.cli plan --config configs/experiment_popularity_vs_cf.v1.json
npm --prefix frontend run build
npm --prefix frontend run typecheck
# Only against the isolated active QA schema with a QA operator token in the environment:
.venv/bin/python scripts/smoke_phase5.py
```

The simulator refuses to overwrite existing outputs and never opens a database connection. Real database/socket and OpenMP subprocess checks required the appropriate sandbox permissions. No tests were weakened or replaced by SQLite to avoid those checks. No additional dependencies were introduced beyond the locked Phase 4 environment.

## Remaining limits

The primary randomization unit is a browser profile, not an authenticated person. Clearing a cookie can create another unit. Operator changes to active configuration during follow-up can contaminate treatment delivery. Coverage thresholds are descriptive; guardrail confidence intervals are not multiplicity-adjusted primary hypotheses. Telemetry excludes requests rejected before enrollment instrumentation and cannot fully characterize database outages. There is no sequential testing, automatic winner, automatic rollout, neural new-user treatment, or large-scale asynchronous analysis job.

The public local database is migrated to schema v2 but contains no live experiment registration or assignment; QA profiles/events were isolated and removed. Preview services and the local QA PostgreSQL process were stopped after checks. Browser-rendered layout, controls and visibility behavior remain unverified. Phase 6 deployment and broader operational work have not been started.

See [the full protocol and activation guide](../../docs/phase5.md) before registering a real experiment.

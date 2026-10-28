# Phase 6 verification

Phase 6 operations are implemented and locally verified. **Updated Docker status:** local container startup, fresh training, HTTP smoke, tracking, monitoring and rollback now pass with a temporary staged checkout and cached official ZIP. Hosted CI, direct Desktop mounts and unaided container download remain open. Nothing was published and no live experiment was activated.

## Observed checks

| Check | Result |
|---|---|
| Full Python suite, including real PostgreSQL integration | **53 passed**, 14 pre-existing Matplotlib/Pyparsing deprecation warnings |
| Production Next.js standalone build | Passed; all eight app routes built |
| TypeScript check | Passed |
| Python dependency consistency (`pip check`) | Passed |
| Python source/script compilation | Passed using a writable temporary bytecode cache |
| Compose, Prometheus, alerts and CI YAML parsing | Passed; this is not Docker/schema/promtool execution |
| Real MLflow SQLite tracking and registry | Passed; run, configuration, metrics, bundle, registered version and `validated` alias |
| Release and receipt integrity checks | Passed, including rejected model/metric/receipt tampering |
| Quality regression rejection | Passed; lower validation NDCG cannot be prepared |
| Local standalone frontend → FastAPI → PostgreSQL | Passed scripted HTTP QA in a disposable schema |
| Metrics authentication and bounded route labels | Passed, including CSRF rejection and unhandled 500 accounting |
| Startup release pinning and restart activation/rollback | Passed over real HTTP; prior release restored |
| Drift checks | Identical, shifted, insufficient and invalid fixtures passed; actual temporal report generated |
| Docker images, Compose health ordering and Prometheus rule execution | **Passed** in staged local Docker verification; see follow-up below |
| Browser visual QA and public deployment | **Not performed** |

The local HTTP test checked six pages, anonymous session persistence, catalog search, cold fallback, warm item CF, rated-item exclusion, event deduplication, click attribution and artifact-backed metrics. All generated actions were explicitly synthetic QA. Its unique PostgreSQL schema was removed and its API/frontend processes stopped afterward. The public database remains free of QA profiles, events and experiment assignments.

## Actual artifact and tracking results

The release gate used the existing real MovieLens Phase 2 run, not generated evaluation scores:

| Validation input | Value |
|---|---:|
| Item-CF NDCG@10 | 0.2687715604 |
| Popularity NDCG@10 | 0.2604025450 |
| Catalog coverage | 0.0445111961 |
| In-process item-CF p95 | 0.8294497 ms |

These pass the configured baseline/coverage/latency policy. They are not a causal lift estimate or a deployed latency benchmark. Test-set metrics were not used in the gate.

Loaded model fingerprint: `item-cf-5e25ab883c84f961`. The real tracking run registered version 1 of `recsyslab-item-cf` in `artifacts/phase6/mlflow.db`; the exact run ID, release hashes and gate inputs are in [release-verification.json](release-verification.json). Generated bundles are under `artifacts/phase6/releases`. Local tracking artifacts are stored by MLflow under ignored `mlruns/`.

Two independently repeated Phase 2 benchmark bundles were selected and rolled back. They have identical model fingerprints and predictions; their release hashes differ because the benchmark timing/provenance files differ. Separate tests exercise activation/rollback with differing model weights. [http-verification.json](http-verification.json) records startup pinning and restart rollback through the standalone frontend. This is a local-process deployment check, not a container rollout claim.

## Actual drift report

[drift.json](drift.json) compares 800,164 MovieLens training ratings with 100,024 validation ratings:

- Rating Jensen–Shannon divergence: **0.0000371267 bits**.
- Item-bucket divergence: **0.0007000070 bits**.
- Unseen-item fraction: **0.0001699592**.
- Status: **ok**, below the heuristic 0.1-bit threshold.

This is an offline temporal distribution comparison, not monitored human traffic. The threshold is an engineering heuristic; it is not calibrated to recommendation-quality loss. The live command returns insufficient data for an empty observation window. Drift does not trigger automatic retraining or promotion.

## Remaining gate and limitations

Run the clean Compose CI job on a Docker-capable machine before accepting the deployment gate. It downloads the official dataset, trains a fresh baseline, builds images, initializes a new PostgreSQL volume, starts the stack, registers the selected bundle in the HTTP MLflow server, runs synthetic HTTP QA and checks Prometheus rules. Image pulls, Linux dependency installation, container permissions, actual scrape/alert behavior and this complete sequence remain unverified here.

The local stack uses loopback bindings, a single API worker and a single-instance restart, with no zero-downtime rollout, external alert notifications or tested backup restoration. Registry aliases do not deploy models; operators explicitly select and restart verified bundles. No automatic retraining, live A/B activation, public hosting or online lift claim was added. See [the runbook](../../docs/phase6.md) for commands and operational boundaries.

## Follow-up: local database recovery

The subsequent recovery drill passed: all 11 public tables matched the exported snapshot after restoration into a separate database. It preserved 1 profile, 6 ratings, 118 interaction events and 16 recommendation batches. Columns/indexes matched; CHECK constraints matched after PostgreSQL reparse. The restored API readiness and catalog search passed, and the temporary database was removed. The user's running application was left intact. See [recovery-verification.json](recovery-verification.json) and [the recovery runbook](../../docs/recovery.md). This closes the local application-database recovery check; full-stack, off-device and Docker recovery remain open.

## Follow-up: isolated Docker preparation

Docker remains unavailable on this Mac. Prepared private Compose configuration, isolated data/artifact directories, separate ports and an independent project name. Compose YAML parsing passes; image builds, container startup, HTTP smoke and monitoring checks have **not** run. See [the prepared commands](../../docs/docker-verification.md).

## Follow-up: Docker execution verified

Docker Desktop 29.9.2 (Linux ARM64 engine) and Compose v5.6.1 successfully built the images and ran the isolated stack. Fresh PostgreSQL initialization, metadata import and API/frontend readiness passed. The official cached MovieLens archive was extracted and benchmarked inside Docker. The HTTP smoke passed six pages plus profile persistence, recommendations, ratings, impressions/clicks and deduplication. Prometheus reported scrape `up=1`, evaluated all three alert rules with healthy status, and rejected unauthorized API metrics access. MLflow registered and served the release bundle; downloaded contents passed integrity and quality checks.

A second model with shrinkage 12 was trained solely for rollback QA. Its validation NDCG was 0.2678018093 versus popularity 0.2604025450. The API stayed pinned before restart, then loaded the different model and returned to its original model after rollback/restart. This is operational QA, not evidence that the second model improves recommendations. The MLflow explicit-URI artifact-context handling was corrected and its context-restoration paths tested. Full regression: **53 passed**, 14 pre-existing warnings. Original profile database: **1 profile and 6 ratings preserved**.

This run required a temporary checkout because mounting the original Desktop path stalled, and a checksum-matched cached official archive because the container could not resolve the dataset host. The unassisted download path and hosted CI have not passed. See [docker-verification.json](docker-verification.json), [docker-rollback.json](docker-rollback.json), [docker-mlflow.json](docker-mlflow.json), and [reproduction/stop commands](../../docs/docker-verification.md). Earlier “not run” notes above describe the original Phase 6 checkpoint; this follow-up supersedes them for the checks actually executed.

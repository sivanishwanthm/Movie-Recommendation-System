# Phase 6: operations and local deployment

Phase 6 adds MLflow run tracking and model registration, verified releases, explicit activation and rollback, JSON request logs, Prometheus metrics/alert rules, drift reports, Docker Compose and CI. It remains a **local development deployment**. Local Docker execution, model restart/rollback, MLflow and Prometheus are now verified using a staged checkout and cached official dataset archive. Direct Desktop mounts, unaided container download and hosted CI remain unverified. See [observed results](../reports/phase6/results.md).

## Install and prepare a release

From the repository root, using the existing virtual environment:

```bash
python -m pip install -e '.[dev,phase4,phase6]'
python -m recsyslab.operations.releases prepare --run artifacts/phase2-1m
```

The command prints a 64-character `release` identifier. Use that exact identifier for `RELEASE_ID` below. `requirements-phase6-lock.txt` records the tested macOS Python 3.9 environment, including neural training dependencies. `requirements-serving-lock.txt` is the smaller pinned dependency closure used in the Python 3.11 API image. The pinned serving dependencies installed successfully in the Linux ARM64 image. Hosted CI on Linux x86_64 remains a separate check.

```bash
export RELEASE_ID='PASTE_RELEASE_IDENTIFIER'
python -m recsyslab.operations.releases verify --release "$RELEASE_ID"
python -m recsyslab.operations.releases track --release "$RELEASE_ID"
python -m recsyslab.operations.releases activate --release "$RELEASE_ID"
export RELEASE_ROOT=artifacts/releases
# Configure DATABASE_URL, then start the existing local launcher:
python scripts/run_phase4.py
```

A release copies the item-CF files, source manifest and benchmark metrics into a directory named by the SHA-256 of its file-hash mapping. It checks the copied snapshot before publishing it. The receipt records the checks and model fingerprint. Verification rehashes all files, reruns the gate and checks receipt consistency. Existing release directories are never overwritten. Only trusted operators should have write access; these fingerprints detect changes, not a malicious operator replacing both data and provenance.

The gate requires validation-selected item CF, equal nonempty validation cohort sizes, finite NDCG at least as high as the same-run popularity baseline, catalog coverage ≥1%, and in-process recommendation p95 ≤50ms. No test-set value enters the gate. These thresholds are a documented engineering policy, not evidence of statistically significant or causal lift. They do not establish superiority over an incumbent from a different protocol. Compare compatible validation runs and review A/B results separately before any real rollout.

## Tracking and registry

`track` creates a real MLflow run with configuration, validation metrics, manifest and the complete release bundle, registers `recsyslab-item-cf`, and updates its `validated` alias. Runs that fail during logging are marked failed. Local tracking defaults to `sqlite:///artifacts/mlflow.db`; artifacts use MLflow's local artifact store. Keep the SQLite file and artifact directory together when backing up or moving them. Run these commands from the repository root to keep relative locations stable.

The bundle is a **custom RecSysLab artifact**, loaded by `LiveRecommender`, not an MLflow pyfunc model. No pickle deserialization is used. `validated` is informational and does not change the running service. Repeated tracking creates additional auditable run/version entries. The serving source of truth is the verified `current.json` selection, not a registry alias. Downloaded bundles must go through `prepare`, `verify`, and explicit activation before use.

For the Compose tracking server:

```bash
python -m recsyslab.operations.releases track --release "$RELEASE_ID" \
  --tracking-uri http://127.0.0.1:5000
```

The MLflow UI binds to loopback only and has no application authentication. Do not expose it publicly. SQLite tracking is intended for a single local operator, not a high-concurrency production registry.

## Activation and rollback

Activation locks the release root and atomically replaces a document containing both the selected and previous release. The API resolves and verifies it **once at startup**. `/api/health` reports `model_version` and `release_id`; legacy `MODEL_RUN` startup reports a null release ID. An invalid bundle, missing catalog metadata or a model/catalog mismatch with the active experiment stops startup. Keep one Uvicorn worker per container for the process-local metrics registry.

```bash
python -m recsyslab.operations.releases activate --release "$RELEASE_ID"
docker compose --env-file .env.compose restart api
curl --fail http://localhost:3000/api/health
# If the new deployment fails health or operational checks:
python -m recsyslab.operations.releases rollback
docker compose --env-file .env.compose restart api
curl --fail http://localhost:3000/api/health
```

For the local launcher, stop it and restart it instead of using Compose commands. Selection alone cannot hot-swap a running model. Repeated rollback toggles the two most recently selected releases. Keep both bundles until the release is accepted. This single-instance deployment has restart downtime; it does not implement a rolling or canary rollout. Coordinate model changes with experiment windows: a different pinned model fails startup with the existing active experiment, and disabling enrollment is not a substitute for completing its follow-up and analysis. Model rollback never downgrades a database schema.

## Clean Docker Compose deployment

For this workspace, an isolated test configuration has been prepared on ports 3001/5001/9091. Follow [the Mac Docker verification guide](docker-verification.md) to preserve the running localhost:3000 session. The defaults below apply to a fresh checkout.

Requires Docker Engine/Desktop and Compose v2 with `--wait` support. No existing local PostgreSQL or Node installation is needed. The bootstrap downloads official MovieLens 1M, trains the Phase 2 benchmark and selects the verified baseline. Its initial download requires network access and is subject to the dataset's terms. It retains an existing verified selection; it will not replace an active release. If a training job was interrupted, use a fresh output directory or remove only its incomplete `artifacts/compose-baseline` output after inspection.

```bash
python scripts/init_compose.py
# Generates private .env.compose and .secrets/metrics_token; refuses to overwrite.
docker compose --env-file .env.compose config --quiet
docker compose --env-file .env.compose run --build --rm bootstrap
docker compose --env-file .env.compose up --build --wait --wait-timeout 180
```

Open http://localhost:3000, MLflow at http://localhost:5000, and Prometheus at http://localhost:9090. PostgreSQL and the API have no host ports. The frontend's same-origin proxy targets `http://api:8000` baked into the container build; local builds default to `http://127.0.0.1:8000`. A successful PostgreSQL health check precedes the one-shot schema migration/catalog import. The API starts only after that succeeds; the frontend waits for API readiness. Only movie metadata is imported into the application database.

API and web processes run without root. Secrets, data, training artifacts and local toolchains are excluded from image build context. Images are version-tagged and application dependencies are pinned; registry image digests and supply-chain attestation are not implemented. This is not a public production security configuration. Add TLS, a production identity model, authenticated/private monitoring, managed secrets, resource limits and reviewed infrastructure before public hosting. Experiment enrollment is off in the default stack.

```bash
# Inspect state and logs; ordinary shutdown retains persistent volumes:
docker compose --env-file .env.compose ps
docker compose --env-file .env.compose logs --tail 100 api
docker compose --env-file .env.compose down
```

Back up PostgreSQL (`pg_dump`), the selected and previous release directories plus `current.json`, the MLflow database/artifacts, and secret configuration before upgrades. Stop MLflow before making a filesystem snapshot of its SQLite volume. Restore backups into a disposable stack and verify health/catalog/profile counts before switching traffic. Do not use `down --volumes` on data you want to retain; only the disposable CI job does that. The public application database has now passed a local backup/restore drill; see [recovery evidence and commands](recovery.md). Full-stack/off-device recovery, including MLflow and release files, remains unverified.

## Monitoring and diagnosis

`/api/live` checks process liveness. `/api/health` checks database reachability and reports the loaded model and schema. `/api/metrics` is disabled unless `METRICS_TOKEN` or `METRICS_TOKEN_FILE` is configured; it requires `Authorization: Bearer TOKEN`. Compose mounts a generated token for both the API and Prometheus. Metrics and request logs exclude query strings, profile IDs, tokens and request bodies. Matched route templates bound label cardinality; unmatched paths share one label. Incoming request IDs are not trusted; the API generates its own.

Counters cover completed requests including origin rejection and unhandled errors. Histogram duration runs through response headers, including database work and experiment telemetry, but excludes streamed body transfer. It is distinct from offline generation timing and Phase 5's experiment-attempt measurement. Scrapes and health probes are also counted; the error alert excludes metrics scrapes. Prometheus stores 15 days of samples. The rules detect failed scraping, sustained >5% server errors, and recommendation API p95 >500ms, with request-volume floors for the latter two. Alerts appear in Prometheus; no Alertmanager or external notifications are configured.

Useful PromQL:

```promql
sum by (route, status) (rate(recsys_http_requests_total[5m]))
histogram_quantile(0.95, sum by (le) (rate(recsys_http_request_duration_seconds_bucket{route="/api/recommendations"}[5m])))
```

For failed startup, check schema migration, catalog import, file hashes and experiment/model pinning. For request failures, use the generated request ID and JSON status/route logs; check database availability and pool pressure. For slow recommendations, compare traffic volume, model size and database latency before altering the model. An unavailable scrape can also mean a token mismatch, so check the service health separately.

## Drift checks

```bash
python -m recsyslab.operations.drift \
  --reference artifacts/phase1-1m/train_ratings.csv \
  --current artifacts/phase1-1m/validation_ratings.csv \
  --output artifacts/offline-drift.json
# Read real application rating events without exporting profile identities:
python -m recsyslab.operations.drift \
  --reference artifacts/phase1-1m/train_ratings.csv \
  --live-since 2026-10-01T00:00:00Z --output artifacts/live-drift.json
```

Outputs are new files, never overwritten. Jensen–Shannon divergence in bits compares fixed half-star buckets and a fixed reference top-100 item set plus known-tail/unseen buckets. Either divergence >0.1 produces a heuristic alert, provided both samples have at least 100 observations; otherwise status is `insufficient_data`. The report also includes unseen-item share. Live input uses timestamp-filtered rating events, so repeated rating changes count as events. Queries currently load the chosen window into memory. Schedule this read-only CLI with your scheduler; it does not silently create a background job.

A change can reflect acquisition, catalog, user selection or instrumentation, not degraded recommendations. These uncalibrated heuristic thresholds do not estimate precision, NDCG or conversion drift. There is no automatic retraining or promotion. Investigate, run the existing temporal benchmark on an appropriate dataset, review compatible validation metrics, then prepare and explicitly activate a new release.

## Verification and CI

```bash
# Set TEST_DATABASE_URL to a test-capable local Postgres role first.
python -m pytest -q
# Requires existing repeated Phase 2 runs, built frontend, and free ports 8000/3000:
python scripts/verify_phase6_local.py
```

The second command creates a unique PostgreSQL schema, runs real HTTP QA through the standalone frontend, verifies startup pinning/restart/rollback and metric authorization, then drops its own schema and stops its servers. Generated QA actions never enter the public application schema. The regular test suite covers differing model weights, gate failures, tampered bundles, exact/shifted/sparse drift, and a real SQLite-backed MLflow registry round trip.

[CI](../.github/workflows/ci.yml) installs all model dependencies, runs PostgreSQL-backed tests and frontend checks, then runs a separate clean Compose bootstrap/build/HTTP smoke job with Prometheus rule validation. It exists as configuration; no hosted run has occurred. Local runtime checks have now passed with the documented staging/cache workarounds; require the hosted job to pass before claiming an unattended clean deployment. Browser visual QA remains outstanding from Phase 4.

Implementation references: [MLflow client and aliases](https://www.mlflow.org/docs/latest/api_reference/python_api/mlflow.client.html), [Compose dependency health/completion conditions](https://docs.docker.com/compose/how-tos/startup-order/), and the installed Next.js 16.4.3 `output: standalone` documentation.

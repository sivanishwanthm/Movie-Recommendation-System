# Docker verification on this Mac

**Verified on October 28, 2026:** both images built, a fresh model trained in Linux, database migration/catalog import completed, all application health checks passed, HTTP smoke passed, MLflow registration/artifact retrieval passed, Prometheus scraping/rule evaluation passed, and activation/restart/rollback passed with different model weights. The full Python regression suite also passed (53 tests).

Two environment workarounds were required:

1. Docker rejected the colon-containing project path with short bind syntax. Strict long-form binds fixed that parsing error, but Docker Desktop still stalled mounting the original Desktop directory. The successful test used a temporary copied checkout outside that path.
2. The container could not resolve the MovieLens download host. We copied the previously downloaded official ZIP, verified the copy's SHA-256, and trained fresh models inside Docker. No local profile database or pretrained model was imported.

This verifies the local container runtime path **with those prerequisites**. Unassisted dataset download, direct mounting of the original Desktop directory and hosted GitHub CI remain unverified. See [verification evidence](../reports/phase6/docker-verification.json) and [rollback evidence](../reports/phase6/docker-rollback.json).

## Current running test stack

- Web: http://127.0.0.1:3001
- MLflow: http://127.0.0.1:5001
- Prometheus: http://127.0.0.1:9091
- Compose project: `recsyslab-check`
- Working directory: `/private/tmp/recsyslab-docker-hn_i_9m0`

Use **127.0.0.1**, not localhost, for this test UI. Browser cookies are shared across ports on the same hostname; a different hostname avoids replacing the original localhost session cookie. This stack contains explicitly synthetic QA activity in its own PostgreSQL volume. Your original local database still contains your 1 profile and 6 ratings and was returned to its prior stopped state after regression checks.

The temporary directory must remain present while these containers use its files. It is a verification setup, not durable hosting. Container-trained models have also been copied back to ignored `artifacts/compose-check/`. Named PostgreSQL/MLflow/Prometheus volumes persist when the stack stops.

To inspect or stop this stack:

```bash
export PATH="/Applications/Docker.app/Contents/Resources/bin:$PATH"
cd /private/tmp/recsyslab-docker-hn_i_9m0
docker compose --project-name recsyslab-check --env-file .env.compose ps
docker compose --project-name recsyslab-check --env-file .env.compose down
```

`down` retains named volumes. Do not delete the temporary checkout before stopping its containers. Do not use `down --volumes` if you want to retain test history or registered MLflow runs.

## Repeat using a new staged checkout

Stop the existing `recsyslab-check` stack first. From the original project directory:

```bash
.venv/bin/python scripts/stage_docker_check.py --use-cached-archive
```

The helper prints a new temporary directory and saves its location in ignored `artifacts/docker-check-location.json`. It copies source/configuration and private local test credentials, creates empty training directories, and optionally copies the official ZIP. It excludes profile databases, backups, model artifacts, local toolchains and node_modules. Omit the flag when container DNS/downloads work and you want to verify that path.

Change into the printed directory, then run:

```bash
export PATH="/Applications/Docker.app/Contents/Resources/bin:$PATH"
docker compose --project-name recsyslab-check --env-file .env.compose config --quiet
docker compose --project-name recsyslab-check --env-file .env.compose run --build --rm bootstrap
docker compose --project-name recsyslab-check --env-file .env.compose up --build --wait --wait-timeout 180
docker compose --project-name recsyslab-check --env-file .env.compose run --rm \
  -e MLFLOW_TRACKING_URI=http://mlflow:5000 bootstrap python scripts/track_selected_release.py
docker compose --project-name recsyslab-check --env-file .env.compose exec -T prometheus \
  promtool check rules /etc/prometheus/alerts.yml
```

Reusing the project name retains its existing named volumes; for a completely empty database/registry, choose a new project name consistently, with the previous stack stopped to free ports. API health prevents mismatched catalogs from starting. The default example experiment is not enabled: Linux-trained artifact fingerprints can differ from the earlier macOS run, so any future experiment must pin the actual deployed fingerprint.

From the **original project directory**, run the HTTP smoke check:

```bash
.venv/bin/python scripts/smoke_phase4.py --base-url http://127.0.0.1:3001
```

It creates synthetic QA interactions in the isolated container database. Prometheus Targets should show `recsyslab-api` UP and `up{job="recsyslab-api"}` should return 1. MLflow should show the registered model and `validated` alias. External alert notification delivery, browser visual QA and public deployment are not established by these checks.

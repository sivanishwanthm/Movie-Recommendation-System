# Phase 4: model serving and the recommendation app

**Current version note:** Phase 5 extends this foundation with schema v2 and experiment endpoints. Run `python -m recsyslab.serving.migrations` before starting the updated API. The v1 descriptions below document the original Phase 4 schema; see [Phase 5](phase5.md) for the additive migration and active-experiment configuration.

Phase 4 adds a local FastAPI service, a real PostgreSQL store and a Next.js App Router frontend. The app supports private browser profiles, movie search and details, half-star ratings, recommendations, measured offline model comparisons and profile-scoped interaction analytics. The API serves the Phase 2 validation-selected item-CF artifact. No request trains a model or reads held-out ratings.

## Architecture

```mermaid
flowchart LR
  Browser[Next.js: discovery / movies / profiles / dashboards] --> Proxy[Same-origin /api proxy]
  Proxy --> API[FastAPI]
  API --> Artifact[Verified frozen item-CF artifact]
  API --> DB[(PostgreSQL schema v1)]
  Live[Live positive ratings only] --> Fold[Sum frozen item similarities]
  Artifact --> Fold
  Fold --> TopK[Seen filtering and Top K]
  Reports[Measured offline report JSON] --> API
```

A movie catalog import loads only official titles, IDs and genres. Historical MovieLens users, demographics and ratings are **never imported into live profile tables**. The serving process loads and verifies the artifact once during FastAPI lifespan startup. Missing metadata, incompatible schema or changed model files fail startup. A database outage returns 503; invalid input returns 422; missing/foreign batches return 404; session failures return 401; conflicting idempotency IDs return 409.

## Setup

Requirements: Python environment from earlier phases, Node.js >=20.9 (22.23.4 tested), PostgreSQL (18.4 tested), official MovieLens metadata, and the Phase 2 `artifacts/phase2-1m` run. The checked-in Phase 4 lockfile captures the complete Python environment, including earlier-phase tools. Do not commit credentials or generated data.

```bash
source .venv/bin/activate
python -m pip install -r requirements-phase4-lock.txt
python -m pip install -e '.[dev,phase4]'
# Install/configure PostgreSQL and create your local database/role first.
# Set this to your actual credentials; .env.example is a template, not auto-loaded.
export DATABASE_URL='postgresql+psycopg://recsyslab:YOUR_PASSWORD@127.0.0.1:5432/recsyslab'
python -m recsyslab.serving.catalog --movies data/raw/ml-1m/movies.dat
cd frontend
npm ci
npm run build
cd ..
python scripts/run_phase4.py
```

Open **http://localhost:3000**. API documentation is at http://127.0.0.1:8000/docs. Ctrl+C in the launcher stops both application processes; it does not stop PostgreSQL. The launcher uses a system Node or the existing ignored `.tools/node` runtime. It does not train models, initialize a database, or build the frontend.

To run services separately:

```bash
# Terminal 1, repository root, DATABASE_URL set:
python -m uvicorn recsyslab.serving.app:create_app --factory --host 127.0.0.1 --port 8000
# Terminal 2:
cd frontend
npm run dev
```

The Next.js proxy forwards `/api/*` to `API_URL` (default `http://127.0.0.1:8000`). API_URL is server-side only and is incorporated into rewrites at build time; rebuild after changing it. `APP_ORIGIN` defaults to `http://localhost:3000`; use that browser URL or set APP_ORIGIN to the exact frontend origin you intend to use. There is no permissive cross-origin CORS policy. `MODEL_RUN` and `REPORTS_DIR` default to `artifacts/phase2-1m` and `reports`. Set `COOKIE_SECURE=true` when HTTPS is actually available; this local phase is not a public deployment.

### Tools available in this workspace

This machine originally had no Node or PostgreSQL. Verification downloaded a checksum-verified official Node runtime and integrity-verified `@embedded-postgres/darwin-arm64` binaries into ignored `.tools/`. The PostgreSQL server identifies as 18.4. The cluster uses trust authentication **only on loopback for local QA** and has been stopped after testing; it is not a production configuration. The initialized public schema contains only movie metadata.

To reuse that existing local cluster:

```bash
.tools/postgres/package/native/bin/pg_ctl -D .tools/pgdata -l .tools/postgres.log \
  -o '-h 127.0.0.1 -p 55432 -k /tmp' start
export DATABASE_URL='postgresql+psycopg://recsyslab@127.0.0.1:55432/postgres'
.venv/bin/python scripts/run_phase4.py
# After stopping the app:
.tools/postgres/package/native/bin/pg_ctl -D .tools/pgdata stop
```

This workspace path contains a colon, which cannot be used as a single PATH entry. The launcher invokes the local Node executable directly. For npm in this workspace, use the existing `/tmp/recsyslab-node` symlink or create a new symlink to `.tools/node` under a path without a colon, then prepend its `bin` directory to PATH. Application npm scripts invoke Next and TypeScript by explicit file path for the same reason. On a normal system Node installation, ordinary npm commands work.

## Live recommendation behavior

Every live profile has a random UUID and a separate bearer session cookie. The API accepts no MovieLens user ID to impersonate. For fewer than three positive (>=4-star) ratings on trained-catalog items, recommendations use the frozen popularity fallback. Thereafter, each candidate score is the sum of the saved item similarities for that profile's positive live-rated movies. Ties use training popularity then movie ID. **All** live-rated movies, including dislikes, are excluded. Rating an out-of-training-catalog movie is permitted but does not contribute to model scores; the details page explains this.

Editing a rating changes the live profile on the next recommendation request. It never changes similarities, retrains a network or updates a shared model. This is item-CF profile folding, with the same scoring rule as Phase 2. Historical offline benchmark values are not presented as measured live-profile performance. The Phase 3 user-ID tower and its ranker are not used for new live identities: doing so would require a separately validated new-user representation. Their measured results remain visible in the benchmark dashboard.

The response reports model version, selected strategy, usable positive-rating count, a generated batch ID and model-generation latency. The returned list is at most 50 movies; the app requests 12. No Redis cache is introduced: immutable models are resident in memory, and live-profile invalidation complexity is not justified by the observed local performance.

## Database and sessions

Schema v1 contains `profiles`, `movies`, `ratings`, `recommendation_batches`, `interaction_events` and `schema_version`. The explicit import command creates the initial schema and performs idempotent metadata upserts. Startup only checks schema and catalog compatibility; it does not mutate schema. Future schema changes require real versioned migrations; this initial version is not a general migration framework.

Session tokens use 32 random bytes; only SHA-256 token hashes are stored. The cookie is HttpOnly, SameSite=Strict, scoped to `/` and expires after 30 days. Server-side expiry is checked too. Mutation requests require the app's custom header and reject unexpected Origin headers. Profiles can be renamed, and persistence survives API restarts. Rating/event endpoints never accept a client-supplied profile owner.

This is an anonymous local browser-profile mechanism, not full account authentication. There is no account recovery, password login, logout/revocation UI, multi-device synchronization or automated expired-profile deletion. Losing the cookie loses access to that profile. Production identity, rate limiting, HTTPS, retention controls and deployment hardening remain future work. No demographic or personal contact data is requested.

## Interaction semantics

Recommendation generation creates a batch with its item IDs, model version and strategy. It does **not** create impressions. The frontend uses IntersectionObserver at 50% visibility to report card impressions, deduplicated per profile/batch/movie. A recommendation-card click first confirms its impression, then records the click before navigation. API validation requires the item to belong to the owned batch and an impression to exist. Repeated clicks on the same card/batch count once. Movie exploration visits are not recommendation clicks.

Ratings accept a UUID event ID and half-star value. Replaying the same payload returns the original accepted event without adding an event or replaying a stale rating over a later update. Conflicting payloads return 409. The frontend retains its pending rating event ID across a failed-request retry. PostgreSQL row locks serialize a profile's writes, with uniqueness constraints backing event deduplication. Concurrent duplicate impression tests verify that one event is stored.

Live analytics are scoped to the current profile. They expose visible recommendation cards, unique clicked cards, rating events, distinct rated movies, batch count, descriptive card click-through and mean model-generation time. A zero impression denominator produces null, displayed as an em dash. These are client-reported interaction logs, not independently verified human activity. Card click-through is **not** user-level conversion or randomized A/B evidence. The page explicitly says experiments are not yet implemented.

## Endpoints

| Method | Route | Purpose |
|---|---|---|
| GET | `/api/health` | Database readiness and loaded model version |
| POST | `/api/profiles` | Create private browser profile/session |
| GET / PATCH | `/api/me` | Inspect / rename current profile |
| GET | `/api/movies?q=&offset=&limit=` | Paginated title search |
| GET | `/api/movies/{id}` | Metadata and trained-catalog eligibility |
| GET | `/api/ratings` | Current profile's ratings |
| PUT | `/api/ratings/{id}` | Idempotent rating event and current rating update |
| POST | `/api/recommendations` | Generate and persist a recommendation batch |
| POST | `/api/impressions` | Log visible batch items idempotently |
| POST | `/api/clicks` | Log an attributed recommendation-card click |
| GET | `/api/analytics` | Current profile's descriptive live activity |
| GET | `/api/benchmarks` | Actual checked-in offline evaluation aggregates |

## Verification

```bash
# Each integration test creates/drops only its own uniquely named test schema.
export TEST_DATABASE_URL='postgresql+psycopg://YOUR_TEST_ROLE:YOUR_PASSWORD@127.0.0.1:5432/YOUR_TEST_DATABASE'
MPLCONFIGDIR=/tmp/recsyslab-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 python -m pytest -q
cd frontend
npm run typecheck
npm run build
```

Without TEST_DATABASE_URL, PostgreSQL tests are explicitly skipped, not silently replaced by SQLite. On this machine all 34 Python tests ran and passed. Full-suite OpenMP subprocess checks require shared-memory access beyond the restricted sandbox. The production Next.js build and explicit TypeScript check pass. `scripts/smoke_phase4.py` checks the real HTTP proxy and service flow using synthetic QA interactions; point it **only at an isolated disposable QA database/schema**, not a live user database. Verification used a dedicated QA schema, kept these events out of application tables, and removed that schema afterward.

Browser automation could not run because Computer Use permission was not granted. HTTP tests verify routes and transport but do not establish visual correctness, click handling, IntersectionObserver behavior, mobile rendering or accessibility in an actual browser. Responsive CSS, accessible labels, keyboard focus states and loading/error/empty states are implemented; manual visual QA remains outstanding.

See [the verification report](../reports/phase4/results.md). Phase 5 adds randomized experimentation; Phase 6 adds deployment and broader operational controls. Neither is claimed complete here.

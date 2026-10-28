# Phase 4 verification

Phase 4 implements a functional local FastAPI/PostgreSQL backend and Next.js application. **34 Python tests passed**, including all eight real-PostgreSQL serving tests. The production frontend build, explicit TypeScript check and end-to-end HTTP smoke test passed. Browser visual and interaction QA could not run because Computer Use permission was not granted.

## Delivered behavior

- Private, persistent browser profiles with HttpOnly session cookies and profile ownership checks.
- Official MovieLens metadata search, pagination and movie details; 3,883 movie metadata rows imported, with no historical ratings or users copied into live tables.
- Half-star live ratings, immediate profile updates and seen-movie exclusion.
- Verified item-CF artifact serving with popularity fallback until three usable positive live ratings exist.
- Recommendation batches, visible-card impression ingestion and click attribution; deduplicated writes backed by PostgreSQL transactions and constraints.
- Offline benchmark dashboard reading the actual Phase 1–3 report artifacts.
- Profile-scoped live activity analytics, explicitly distinguished from randomized experiments.
- Local launcher and documented setup/configuration; no training on request paths.

The UI includes For You, Explore, movie details, Profile, Models and Activity pages. Movie card artwork is decorative typography, not official posters. No fake movie descriptions, ratings, model metrics, engagement outcomes or A/B results are displayed.

## Checks performed

| Check | Outcome |
|---|---|
| Full Python suite | 34 passed, no skips, 14 existing Matplotlib/Pyparsing deprecation warnings |
| PostgreSQL integration | 8 tests passed in uniquely named disposable schemas |
| Frozen model integrity | Changed artifact bytes rejected before startup |
| No training during recommendations | Training method patched to raise; requests still succeeded |
| Profile boundaries | Foreign batch access rejected; ratings/analytics isolated |
| Rating replay | Identical event accepted once; conflicting reuse rejected |
| Concurrent impressions | Four simultaneous duplicate submissions stored exactly one event |
| Session persistence | Ratings remained accessible across new API application instances |
| CSRF guards | Unexpected origins and missing custom client headers rejected |
| Next.js production build | Passed; six application routes plus standard not-found route |
| TypeScript | `npm run typecheck` passed |
| HTTP through Next.js proxy | Six pages, readiness, profiles, ratings, recommendations, events and dashboards passed |
| Local launcher | Both child services reached ready state; Ctrl+C teardown checked |
| Browser layout/interaction review | Not performed: Computer Use permission unavailable |

The HTTP smoke test used the actual MovieLens metadata and verified trained item-CF artifact, but its ratings and interactions were **synthetic QA actions**, not human engagement observations. They were isolated in `phase4_http_qa`, separate from the application's public schema, and that schema was dropped after verification. No QA conversion rate or apparent engagement lift is reported as a project result. See [machine-readable HTTP check results](http-smoke.json).

## Commands executed

```bash
TEST_DATABASE_URL='postgresql+psycopg://recsyslab@127.0.0.1:55432/postgres' \
MPLCONFIGDIR=/tmp/recsyslab-mpl PYTHONPYCACHEPREFIX=/tmp/recsyslab-pycache \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
.venv/bin/python -m pytest -q

# Node was provided by the ignored local installation via a colon-free PATH.
npm --prefix frontend run build
npm --prefix frontend run typecheck

# API was configured against the disposable QA schema for this command.
.venv/bin/python scripts/smoke_phase4.py
```

The host sandbox restricted PostgreSQL/socket access and OpenMP subprocess shared memory; these tests ran with the necessary local permissions. This did not change test assertions or substitute a mock database.

## Environment and limitations

Tested with Python 3.9.6, FastAPI 0.115.12, SQLAlchemy 2.0.41, psycopg 3.2.9, PostgreSQL 18.4, Node 22.23.4, Next.js 16.4.3, React 19.3.1 and TypeScript 7.0.2. Exact dependency resolutions are retained in `requirements-phase4-lock.txt` and `frontend/package-lock.json`.

The live serving artifact is the validation-selected Phase 2 item-CF model. Live-profile scoring is driven only by ratings submitted to this app, not a mapped MovieLens user ID. Offline report metrics describe historical frozen profiles and are not measurements of live-profile quality. The ID-based neural user tower is not silently repurposed for new identities.

Browser-driven card visibility, click handling, responsive layout and accessibility remain visually unverified. HTTP success does not prove those behaviors. The code implements IntersectionObserver logging, loading/error/empty states, accessible labels, focus styles and responsive layouts; manual browser review is still needed.

This is a local development app, not a public production deployment. Anonymous 30-day sessions have no account recovery or multi-device login. Schema v1 initialization is explicit; future schema evolution needs versioned migrations. Production authentication, HTTPS, rate limiting, retention controls, automated deployment and monitoring remain later work. Randomized experiment assignment, SRM, power analysis and experiment inference belong to Phase 5 and are not implemented.

The application and QA database processes were stopped after verification. See [the startup guide](../../docs/phase4.md) to use the retained local tools and build, or install standard Node/PostgreSQL dependencies on another machine.

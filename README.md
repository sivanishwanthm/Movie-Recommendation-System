# RecSysLab

A reproducible movie recommendation project. **Phases 1–6 are implemented:** official MovieLens ingestion, validation, descriptive EDA, time-aware splitting, popularity, item-based collaborative filtering, implicit ALS matrix factorization, PyTorch two-tower retrieval, FAISS indexing, LightGBM ranking, paired offline model comparison, FastAPI serving, PostgreSQL interaction storage, a Next.js app, randomized experiment infrastructure and statistical analysis. MLflow tracking, verified releases/rollback, monitoring, drift checks, Compose and CI are included. Local Docker runtime checks passed with a staged checkout and cached dataset; hosted CI remains pending; see [the architecture](docs/architecture.md).

## Setup and reproduce

Python 3.9–3.12 recommended. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
python -m pip install -e '.[dev]'
pytest -q
recsyslab download --dataset 1m
MPLCONFIGDIR=/tmp/recsyslab-mpl recsyslab run --config configs/phase1.json --data data/raw/ml-1m --output artifacts/phase1-1m
```

Use a new output directory for each run; existing directories are rejected. The downloader preserves the archive, records its SHA-256 and validates ZIP CRCs. SHA-256 is a reproducibility fingerprint, not independently authenticated publisher verification. Source file hashes, split cutoffs, configuration and runtime versions are captured in each manifest. No datasets or user-level outputs should be committed.

Run artifacts:

- `manifest.json`: configuration, provenance, cutoffs, row counts, model checksum and environment.
- `popularity.json`: versioned training-only catalog and positive counts.
- `eda.json`, `eda.png`: descriptive summaries and distributions.
- `train_ratings.csv`, `validation_ratings.csv`, `test_ratings.csv`: auditable partitions.
- `metrics.json`: validation/test metrics, user bootstrap intervals, coverage and local timing.
- `validation_per_user.csv`, `test_per_user.csv`: audit-ready user metrics.

Read [the evaluation contract](docs/evaluation.md) before interpreting results. Positives are ratings >= 4. Global temporal boundaries prevent future data crossing into training. The catalog contains only movies seen during training. Both evaluation windows use the frozen training model; test filtering includes prior validation history. Cold-item positives remain in the metric denominator. No negative sampling is used.

## Phase 2 benchmark

After installing dependencies and downloading 1M as above:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
MPLCONFIGDIR=/tmp/recsyslab-mpl \
python -m recsyslab.benchmark --config configs/phase2.json --output artifacts/phase2-1m
```

The benchmark fits popularity, item CF and implicit ALS using identical training data and full-catalog evaluation. It records validation-based model selection, paired user-bootstrap differences, warm/fallback cohort metrics, training time and local recommendation timing. Models persist as JSON/NPZ and are verified after reload. See [model definitions and limitations](docs/phase2.md) and [measured Phase 2 results](reports/phase2/results.md). A new output directory is required for each run.

## Phase 3 retrieval and ranking

```bash
python -m pip install -r requirements-phase3-lock.txt
python -m pip install -e '.[dev,phase3]'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
MPLCONFIGDIR=/tmp/recsyslab-mpl \
python -m recsyslab.stage3 --config configs/phase3.json --output artifacts/phase3-new
```

A forward temporal fold inside training supplies leakage-safe LambdaRank candidates and labels. The final retriever is refitted on outer training; FAISS returns at most 200 unseen candidates for ranking. The benchmark compares retrieval-only and ranked outputs against popularity and item CF, measures candidate recall, and verifies persisted model/index compatibility. See [Phase 3 methodology](docs/phase3.md) and [measured results](reports/phase3/results.md). The current exact FAISS index does not establish approximate-search scalability. Full 32M execution remains unvalidated.

## Phase 4: run the application

The app provides profiles, movie search/details, ratings, personalized recommendations, measured model benchmarks and live activity analytics. It serves the validation-selected item-CF artifact; new profiles never inherit MovieLens identities or held-out ratings.

```bash
python -m pip install -r requirements-phase4-lock.txt
python -m pip install -e '.[dev,phase4]'
# Configure PostgreSQL first; substitute your actual local connection URL.
export DATABASE_URL='postgresql+psycopg://recsyslab:YOUR_PASSWORD@127.0.0.1:5432/recsyslab'
python -m recsyslab.serving.catalog
cd frontend
npm ci
npm run build
cd ..
python scripts/run_phase4.py
```

Open **http://localhost:3000**. The API docs are at http://127.0.0.1:8000/docs. See [the complete Phase 4 run guide](docs/phase4.md), including how to reuse the local Node/PostgreSQL tools already installed in this workspace, and [verification results](reports/phase4/results.md). Browser visual QA remains unverified because Computer Use permission was unavailable. The API, database integration and production frontend build were verified.

## Phase 5: experiments and analysis

Phase 5 adds immutable definitions, deterministic profile assignment, visible exposure attribution, fixed-window user conversion, SRM checks, confidence intervals, power planning and guardrails. The `/experiments` dashboard separates operator-restricted live analysis from generated simulations. Live enrollment is **off by default**.

```bash
# With DATABASE_URL configured:
python -m recsyslab.serving.migrations
python -m recsyslab.experimentation.cli plan --config configs/experiment_popularity_vs_cf.v1.json
# File-only simulation; it never writes generated events to PostgreSQL:
python -m recsyslab.experimentation.simulate --output artifacts/synthetic-experiments.json
```

The supported live comparison is popularity vs live-profile item CF. Primary conversion means any attributed recommendation click within 24 hours, analyzed only after each profile's full window. No real online lift is claimed. Read [the Phase 5 protocol and activation guide](docs/phase5.md) before registering or enabling an experiment, and [the verification report](reports/phase5/results.md) for test and simulation evidence. Phase 6 operations are now implemented; local Docker runtime verification passed with documented workarounds.

## Phase 6: operations and deployment

Phase 6 adds MLflow tracking/registry, verified release bundles with validation gates and rollback, structured request logs, Prometheus metrics/alerts, distribution drift checks, Docker Compose and CI.

```bash
python -m pip install -e '.[dev,phase4,phase6]'
python -m recsyslab.operations.releases prepare --run artifacts/phase2-1m
# Use the printed release ID to track and activate it; see the runbook below.
```

Read [the Phase 6 runbook](docs/phase6.md) for release activation, restart/rollback, clean Compose bootstrap, monitoring and backups. [Verification evidence](reports/phase6/results.md) includes PostgreSQL tests, real MLflow registration and local standalone HTTP rollback checks. Docker startup, HTTP behavior, MLflow, monitoring and rollback now pass using a staged checkout and cached official archive. Hosted CI and unattended dataset download remain open; see [Docker verification](docs/docker-verification.md). No public deployment or live experiment was activated.

Local database recovery has also been verified; see [backup/restore commands and scope](docs/recovery.md).

## MovieLens 32M

```bash
recsyslab download --dataset 32m
python -c 'import json; c=json.load(open("configs/phase1.json")); c["dataset"]="32m"; json.dump(c,open("configs/phase1-32m.json","w"),indent=2)'
recsyslab run --config configs/phase1-32m.json --data data/raw/ml-32m --output artifacts/phase1-32m
```

The 32M CSV adapter is fixture-tested, but a full 32M benchmark has not been run. It loads data in memory; plan substantial RAM and disk for copies and output partitions. It does not fabricate the demographics absent from this release. Chunked ingestion and efficient columnar artifacts are future scaling work.

## Sources and limitations

Official sources: [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/), [MovieLens 32M](https://grouplens.org/datasets/movielens/32m/), and each downloaded README containing usage terms. Retain attribution and review those terms before use or redistribution. Citation: F. Maxwell Harper and Joseph A. Konstan (2015), *The MovieLens Datasets: History and Context*, ACM TiiS, DOI: 10.1145/2827872.

These are observational explicit ratings, not clicks or conversions. Offline recall does not establish causal engagement lift. No online A/B experiment has run. Global splitting introduces cold-start users, and rating selection bias limits generalization. Popularity is an intentionally simple non-personalized baseline. Bootstrap intervals are conditional on a single temporal split. Timings cover local recommendation calls only, not deployed latency. The local app is functional; public deployment, production identity and further operational hardening remain outstanding.

See [architecture and phased gates](docs/architecture.md), [metric definitions](docs/evaluation.md), and [verified run results](reports/phase1.md).

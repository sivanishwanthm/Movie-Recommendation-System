# Local database backup and restore drill

The recovery drill passed on PostgreSQL 18.4. A consistent snapshot of the application's `public` schema was dumped, restored into a new randomly named database, and verified. The running application and its source database were not modified or restarted.

Observed snapshot: 1 profile, 6 ratings, 118 interaction events, 16 recommendation batches and 3,883 movies. All 11 table hashes matched. Columns and indexes matched exactly. PostgreSQL rendered two CHECK constraints with equivalent but different cast syntax after restoration; the verifier reparses CHECK definitions in temporary tables and compares the resulting definitions, rather than ignoring constraints. API readiness and movie search passed against the restored database. The temporary database was removed.

See [machine-readable evidence](../reports/phase6/recovery-verification.json). Row counts reflect that snapshot, not future activity. `schema_version: 1` in the table-count map means one version-table row; the application schema remains version 2.

## Repeat the drill

Run from the repository root while the local database is running:

```bash
DATABASE_URL='postgresql+psycopg://recsyslab@127.0.0.1:55432/postgres' \
.venv/bin/python scripts/verify_backup_restore.py --bin-dir .tools/pg-client/bin
```

The script exports one read-only repeatable-read snapshot and shares it with `pg_dump`, so continuing user activity does not create false comparison failures. It requires a database role with `CREATEDB`. It cannot accept an existing restore-target name: the destination is generated internally and only that database is dropped during cleanup. Restore executes in one transaction and stops on errors. A failed comparison retains the backup and records failure in the report. Each run makes a new backup directory; it updates the latest recovery report.

Backups are in ignored `artifacts/backups/<UTC timestamp>-<suffix>/`. Each directory contains `application.dump`, `manifest.json`, `dump.log` and `restore.log`. The directory is accessible only to its owner; the dump and manifest have mode 0600. The manifest includes the dump SHA-256 and per-table row counts/content hashes. It contains no plaintext row values, but the dump contains private profile data and session-token hashes: keep backups private. These files are local and unencrypted; an encrypted off-device backup and a retention schedule are still needed for actual disaster recovery.

Matching PostgreSQL 18.4 `pg_dump` and `pg_restore` were built from the official source archive after checking its published SHA-256. They and their local libpq are under ignored `.tools/pg-client/`. This build is intended for the existing loopback database and does not enable TLS. Use a supported packaged PostgreSQL client with TLS when connecting to hosted databases.

## Recovery scope

This verifies recovery of the public application schema and its data, indexes and constraints into a database on the **same local PostgreSQL instance**. It does not verify recovery from loss of this computer, PostgreSQL roles/permissions, cross-version restoration, model/release files, MLflow storage or secret configuration. Ownership and ACLs are intentionally not restored in this local drill. The restored API uses the existing local model artifact. Keep model releases, `current.json`, MLflow database/artifacts and secrets in a separate private backup plan.

Do not restore a dump over the running application's database. For real recovery, first restore into a fresh database, run this comparison and application checks, then explicitly configure the app to use the verified replacement. Keep the old database until the replacement is accepted.

Reference: [PostgreSQL pg_dump snapshot and archive documentation](https://www.postgresql.org/docs/18/app-pgdump.html).

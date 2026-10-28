"""Back up public application tables and restore into a disposable PostgreSQL database.

Source remains online. Exported snapshot makes source fingerprints and pg_dump
observe the same committed state. Requires CREATEDB and matching pg_dump/restore.
No existing database can be selected as the restore target.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
from uuid import uuid4

import psycopg
from psycopg import sql
from sqlalchemy.engine import make_url


def fingerprint(conn):
    result = {}
    tables = conn.execute("SELECT tablename FROM pg_tables WHERE schemaname='public' ORDER BY tablename").fetchall()
    for (name,) in tables:
        digest = hashlib.sha256()
        count = 0
        statement = sql.SQL('SELECT to_jsonb(t)::text FROM public.{} t ORDER BY to_jsonb(t)::text').format(sql.Identifier(name))
        with conn.cursor(name='fingerprint_' + uuid4().hex) as cursor:
            cursor.execute(statement)
            for (row,) in cursor:
                digest.update(row.encode('utf-8') + b'\n')
                count += 1
        result[name] = {'rows':count, 'sha256':digest.hexdigest()}
    return result


def schema_signature(conn, detail=False):
    columns = conn.execute("""SELECT table_name,column_name,data_type,udt_name,is_nullable,column_default
        FROM information_schema.columns WHERE table_schema='public' ORDER BY table_name,ordinal_position""").fetchall()
    indexes = conn.execute("SELECT tablename,indexname,indexdef FROM pg_indexes WHERE schemaname='public' ORDER BY tablename,indexname").fetchall()
    constraints = conn.execute("""SELECT c.relname,co.conname,pg_get_constraintdef(co.oid)
        FROM pg_constraint co JOIN pg_class c ON c.oid=co.conrelid
        JOIN pg_namespace n ON n.oid=c.relnamespace WHERE n.nspname='public'
        ORDER BY c.relname,co.conname""").fetchall()
    definition = [columns,indexes,constraints]
    return definition if detail else hashlib.sha256(json.dumps(definition,sort_keys=True).encode()).hexdigest()



def canonical_schema(conn, definition):
    """Reparse CHECK expressions in disposable tables; preserve all other definitions."""
    columns, indexes, constraints = definition
    normalized = []
    for table, name, expression in constraints:
        if expression.startswith('CHECK '):
            temporary = 'restore_check_' + uuid4().hex
            conn.execute(sql.SQL('CREATE TEMP TABLE {} (LIKE public.{}) ON COMMIT DROP').format(
                sql.Identifier(temporary), sql.Identifier(table)))
            conn.execute(sql.SQL('ALTER TABLE {} ADD CONSTRAINT audit_check ').format(sql.Identifier(temporary))
                         + sql.SQL(expression))
            expression = conn.execute("SELECT pg_get_constraintdef(oid) FROM pg_constraint WHERE conrelid=%s::regclass AND conname='audit_check'", (temporary,)).fetchone()[0]
            conn.execute(sql.SQL('DROP TABLE {}').format(sql.Identifier(temporary)))
        normalized.append((table, name, expression))
    return [columns, indexes, normalized]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bin-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('artifacts/backups'))
    parser.add_argument('--report', type=Path, default=Path('reports/phase6/recovery-verification.json'))
    args = parser.parse_args()
    url = make_url(os.environ['DATABASE_URL'])
    if url.drivername != 'postgresql+psycopg' or url.query:
        raise ValueError('Use a postgresql+psycopg URL without query options')
    connection = {'host':url.host or '127.0.0.1','port':url.port or 5432,
                  'user':url.username,'password':url.password,'dbname':url.database}
    env = dict(os.environ)
    for key,value in [('PGHOST',connection['host']),('PGPORT',connection['port']),
                      ('PGUSER',connection['user']),('PGPASSWORD',connection['password'])]:
        env.pop(key, None)
        if value is not None:
            env[key] = str(value)
    # Never allow ambient target/service settings to override explicit CLI arguments.
    for key in ['PGDATABASE','PGSERVICE','PGSERVICEFILE','PGOPTIONS']:
        env.pop(key,None)
    for name in ['pg_dump','pg_restore']:
        if not (args.bin_dir / name).is_file():
            raise ValueError(f'Missing {name} in bin-dir')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid4().hex[:8]
    directory = args.output / stamp
    directory.mkdir(mode=0o700, parents=True, exist_ok=False)
    backup = directory / 'application.dump'
    scratch = 'recsys_restore_' + uuid4().hex
    created = False
    report = {'checked_at':datetime.now(timezone.utc).isoformat(), 'scope':'public application schema; not cluster roles, model files, or MLflow',
              'backup':str(backup), 'status':'failed', 'source_modified':False}
    try:
        with psycopg.connect(**connection, autocommit=True) as admin:
            try:
                with psycopg.connect(**connection) as source:
                    source.execute('BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY')
                    snapshot = source.execute('SELECT pg_export_snapshot()').fetchone()[0]
                    before = fingerprint(source)
                    schema_before = schema_signature(source)
                    definition_before = schema_signature(source, detail=True)
                    report['postgres_version'] = source.execute('SHOW server_version').fetchone()[0]
                    with (directory / 'dump.log').open('w') as log:
                        subprocess.run([str(args.bin_dir/'pg_dump'), '--dbname',url.database,
                            '--format=custom','--schema=public','--no-owner','--no-acl',
                            '--snapshot',snapshot,'--file',str(backup)], env=env, check=True, stdout=log, stderr=log)
                backup.chmod(0o600)
                manifest = {'tables':before,'schema_sha256':schema_before,
                            'dump_sha256':hashlib.sha256(backup.read_bytes()).hexdigest()}
                (directory / 'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
                (directory / 'manifest.json').chmod(0o600)
                admin.execute(sql.SQL('CREATE DATABASE {} TEMPLATE template0').format(sql.Identifier(scratch)))
                created = True
                with (directory / 'restore.log').open('w') as log:
                    subprocess.run([str(args.bin_dir/'pg_restore'),'--dbname',scratch,
                        '--no-owner','--no-acl','--clean','--if-exists','--exit-on-error',
                        '--single-transaction',str(backup)],env=env,check=True,stdout=log,stderr=log)
                with psycopg.connect(**dict(connection,dbname=scratch)) as restored:
                    after = fingerprint(restored)
                    schema_after = schema_signature(restored)
                    definitions_match = canonical_schema(restored, definition_before) == canonical_schema(
                        restored, schema_signature(restored, detail=True))
                    if before != after or not definitions_match:
                        report['different_tables'] = [k for k in set(before) | set(after) if before.get(k) != after.get(k)]
                        report['schema_matches'] = definitions_match
                        raise RuntimeError('Restored contents or schema differ from the backup snapshot')
                # Verify the actual application can read the recovered database/model.
                from fastapi.testclient import TestClient
                from recsyslab.serving.app import create_app,Settings
                restored_url = url.set(database=scratch).render_as_string(hide_password=False)
                settings = Settings(restored_url,Path(os.getenv('MODEL_RUN','artifacts/phase2-1m')),Path('reports'))
                with TestClient(create_app(settings)) as client:
                    health = client.get('/api/health')
                    assert health.status_code == 200, 'Restored API health failed'
                    catalog = client.get('/api/movies',params={'q':'Toy Story'})
                    assert catalog.status_code == 200 and catalog.json()['total'] > 0, 'Restored catalog check failed'
                report.update(status='passed',table_counts={k:v['rows'] for k,v in before.items()},
                              all_table_hashes_match=True,columns_indexes_constraints_match=True,
                              schema_comparison='Columns/indexes exact; CHECK constraints compared after PostgreSQL reparse in temporary tables.',
                              restored_api_health='passed',restored_catalog_search='passed',
                              backup_sha256=manifest['dump_sha256'])
            finally:
                if created:
                    # Only our randomly named database is dropped; never accept a target argument.
                    admin.execute(sql.SQL('DROP DATABASE {}').format(sql.Identifier(scratch)))
                    report['temporary_database_removed'] = True
    finally:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()

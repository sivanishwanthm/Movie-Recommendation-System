"""Disposable real-Postgres HTTP and startup/rollback check; requires free ports 8000/3000.

Uses existing built frontend and benchmark artifacts. Never writes QA events to public.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from uuid import uuid4
import httpx
from sqlalchemy import create_engine, text
from recsyslab.serving.database import initialize
from recsyslab.serving.catalog import import_movies
from recsyslab.operations.releases import prepare, select_release

ROOT = Path(__file__).resolve().parents[1]


def ready(process, url, expected=None):
    for _ in range(100):
        if process.poll() is not None:
            raise RuntimeError('Server exited; inspect artifacts/phase6/http-qa.log')
        try:
            response = httpx.get(url, timeout=2)
            if response.status_code == 200 and (expected is None or response.json()['release_id'] == expected):
                return
        except httpx.HTTPError:
            pass
        time.sleep(.1)
    raise RuntimeError('Server did not become ready')


def stop(process):
    if process and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def main():
    url = os.environ['TEST_DATABASE_URL']
    admin = create_engine(url)
    schema = 'recsys_phase6_qa_' + uuid4().hex
    scoped = url + ('&' if '?' in url else '?') + 'options=-csearch_path%3D' + schema
    with admin.begin() as conn:
        conn.execute(text(f'CREATE SCHEMA {schema}'))
    api = web = None
    try:
        engine = create_engine(scoped)
        initialize(engine)
        import_movies(engine, ROOT / 'data/raw/ml-1m/movies.dat')
        engine.dispose()
        output = ROOT / 'artifacts/phase6'
        output.mkdir(exist_ok=True, parents=True)
        with tempfile.TemporaryDirectory(dir=output) as temp, (output / 'http-qa.log').open('w') as log:
            releases = Path(temp) / 'releases'
            a = prepare(ROOT / 'artifacts/phase2-1m', releases)['release']
            b = prepare(ROOT / 'artifacts/phase2-repeat', releases)['release']
            select_release(releases, a)
            env = dict(os.environ, DATABASE_URL=scoped, RELEASE_ROOT=str(releases),
                       METRICS_TOKEN='disposable-qa-metrics-token', ACTIVE_EXPERIMENT='', APP_ORIGIN='http://localhost:3000')
            def start_api():
                process = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'recsyslab.serving.app:create_app',
                                            '--factory','--host','127.0.0.1','--port','8000','--no-access-log'],
                                           cwd=ROOT, env=env, stdout=log, stderr=log)
                return process
            api = start_api()
            ready(api, 'http://127.0.0.1:8000/api/health', a)
            standalone = ROOT / 'frontend/.next/standalone'
            shutil.copytree(ROOT / 'frontend/.next/static', standalone / '.next/static', dirs_exist_ok=True)
            node = shutil.which('node') or str(ROOT / '.tools/node/bin/node')
            web = subprocess.Popen([node, 'server.js'], cwd=standalone,
                                   env=dict(env, HOSTNAME='127.0.0.1', PORT='3000'), stdout=log, stderr=log)
            ready(web, 'http://localhost:3000/api/health', a)
            smoke = subprocess.run([sys.executable, 'scripts/smoke_phase4.py'], cwd=ROOT,
                                   check=True, capture_output=True, text=True)
            assert httpx.get('http://localhost:3000/api/metrics').status_code == 403
            metrics = httpx.get('http://localhost:3000/api/metrics', headers={'Authorization':'Bearer disposable-qa-metrics-token'})
            assert metrics.status_code == 200 and 'recsys_http_request_duration_seconds_bucket' in metrics.text
            select_release(releases, b)
            # A live process stays pinned until explicitly restarted.
            assert httpx.get('http://localhost:3000/api/health').json()['release_id'] == a
            stop(api)
            api = start_api()
            ready(api, 'http://localhost:3000/api/health', b)
            select_release(releases, rollback=True)
            stop(api)
            api = start_api()
            ready(api, 'http://localhost:3000/api/health', a)
            result = {'http_smoke':json.loads(smoke.stdout), 'metrics_authorization':'passed',
                      'standalone_frontend':'passed', 'startup_pin_and_restart':'passed',
                      'restart_rollback':'passed', 'initial_release':a, 'second_release':b, 'restored_release':a,
                      'scope':'Local processes, real PostgreSQL, disposable schema; not Docker execution.'}
            (ROOT / 'reports/phase6/http-verification.json').write_text(json.dumps(result, indent=2) + '\n')
            print(json.dumps(result, indent=2))
    finally:
        stop(web)
        stop(api)
        with admin.begin() as conn:
            conn.execute(text(f'DROP SCHEMA {schema} CASCADE'))
        admin.dispose()


if __name__ == '__main__':
    main()

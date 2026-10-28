"""Run the already-built local app and API; never train or initialize a database.

Set DATABASE_URL first. Ctrl+C stops both child processes. PostgreSQL is managed
separately. Requires an existing Next.js build and Phase 2 model artifacts.
"""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    if not os.getenv('DATABASE_URL'):
        raise SystemExit('Set DATABASE_URL to your PostgreSQL psycopg connection URL.')
    node=shutil.which('node')
    if node is None:
        local=ROOT/'.tools/node/bin/node'
        if local.exists():node=str(local)
    if node is None:raise SystemExit('Install Node.js >=20.9 (Node 22 was tested).')
    if not (ROOT/'frontend/.next/BUILD_ID').exists():
        raise SystemExit('Build frontend first: cd frontend && npm ci && npm run build')
    children=[]
    environment=dict(os.environ,NEXT_TELEMETRY_DISABLED='1')
    try:
        children.append(subprocess.Popen([sys.executable,'-m','uvicorn','recsyslab.serving.app:create_app',
                         '--factory','--host','127.0.0.1','--port','8000','--no-access-log'],cwd=ROOT,env=environment))
        standalone=ROOT/'frontend/.next/standalone'
        if (standalone/'server.js').exists():
            shutil.copytree(ROOT/'frontend/.next/static',standalone/'.next/static',dirs_exist_ok=True)
            children.append(subprocess.Popen([node,'server.js'],cwd=standalone,
                                             env=dict(environment,HOSTNAME='127.0.0.1',PORT='3000')))
        else:
            children.append(subprocess.Popen([node,'node_modules/next/dist/bin/next','start','--hostname','127.0.0.1'],
                                             cwd=ROOT/'frontend',env=environment))
        print('Local app: http://localhost:3000 | API docs: http://127.0.0.1:8000/docs',flush=True)
        while all(child.poll() is None for child in children):time.sleep(1)
        raise SystemExit('A service exited; inspect its log above. Both services have been stopped.')
    except KeyboardInterrupt:
        print('\nStopping local services…')
    finally:
        for child in children:
            if child.poll() is None:child.terminate()
        for child in children:
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:child.kill();child.wait()

if __name__=='__main__':main()

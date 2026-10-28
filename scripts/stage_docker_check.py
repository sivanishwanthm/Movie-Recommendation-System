"""Stage an isolated Compose verification checkout outside colon-containing Desktop paths.

Does not launch Docker or copy local profiles, database files or model artifacts.
The resulting directory is temporary; retain it while its containers are running.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--use-cached-archive', action='store_true', help='Copy the existing official MovieLens ZIP when container download is unavailable')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if not (root / '.env.compose').is_file() or not (root / '.secrets/metrics_token').is_file():
        raise SystemExit('Run scripts/init_compose.py first.')
    stage = Path(tempfile.mkdtemp(prefix='recsyslab-docker-'))
    ignore = shutil.ignore_patterns('.venv','.tools','data','artifacts','node_modules','.next','.git',
                                   '.secrets','.env*','mlruns','__pycache__','.pytest_cache','*.log')
    shutil.copytree(root, stage, dirs_exist_ok=True, ignore=ignore)
    (stage / '.secrets').mkdir(mode=0o700)
    shutil.copyfile(root / '.secrets/metrics_token', stage / '.secrets/metrics_token')
    overrides = {'WEB_PORT':'3001','MLFLOW_PORT':'5001','PROMETHEUS_PORT':'9091',
                 'APP_HOST':'127.0.0.1','DATA_DIR':'./data','ARTIFACTS_DIR':'./artifacts'}
    lines = [line for line in (root / '.env.compose').read_text().splitlines()
             if line.partition('=')[0] not in overrides]
    (stage / '.env.compose').write_text('\n'.join(lines+[f'{k}={v}' for k,v in overrides.items()])+'\n')
    (stage / '.env.compose').chmod(0o600)
    (stage / 'data').mkdir()
    (stage / 'artifacts/releases').mkdir(parents=True)
    receipt = {'directory':str(stage), 'cached_archive':args.use_cached_archive, 'project':'recsyslab-check'}
    if args.use_cached_archive:
        source = root / 'data/raw/ml-1m.zip'
        target = stage / 'data/raw/ml-1m.zip'
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise RuntimeError('Archive copy checksum mismatch')
        receipt['archive_sha256'] = digest
    (root / 'artifacts').mkdir(exist_ok=True)
    (root / 'artifacts/docker-check-location.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))
    print('Run Compose from that directory with --project-name recsyslab-check --env-file .env.compose.')
    print('Stop any earlier recsyslab-check stack before starting a new staged checkout.')


if __name__ == '__main__':
    main()

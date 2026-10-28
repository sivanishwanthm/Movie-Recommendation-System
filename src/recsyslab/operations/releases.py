"""Validation-gated item-CF releases; atomic startup selection, explicit rollback.

Bundles are custom RecSysLab artifacts, not pickled MLflow pyfunc models.
Only trusted operators may write the release directory or MLflow registry.
"""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from datetime import datetime, timezone
from ..data import digest
from ..serving.model import LiveRecommender

FILES = ['manifest.json', 'metrics.json', 'item_cf/model.json', 'item_cf/arrays.npz',
         'item_cf/profiles.npz', 'item_cf/similarity.npz']


def quality_gate(run):
    manifest = json.loads((run / 'manifest.json').read_text())
    if digest(run / 'metrics.json') != manifest['artifact_sha256']['metrics.json']:
        raise ValueError('metrics integrity check failed')
    LiveRecommender(run)
    metrics = json.loads((run / 'metrics.json').read_text())['validation']
    candidate, baseline = metrics['item_cf'], metrics['popularity']
    ndcg, reference = (m['metrics']['ndcg']['mean'] for m in (candidate, baseline))
    coverage, latency = candidate['catalog_coverage'], candidate['latency_ms_p95']
    numbers = [ndcg, reference, coverage, latency]
    if not all(math.isfinite(v) for v in numbers):
        raise ValueError('non-finite gate inputs')
    checks = {
        'validation_selected_item_cf': manifest['selected_on_validation'] == 'item_cf',
        'same_nonempty_cohort': candidate['users_evaluated'] == baseline['users_evaluated'] > 0,
        'ndcg_at_least_popularity': 0 <= reference <= ndcg <= 1,
        'catalog_coverage_at_least_1_percent': .01 <= coverage <= 1,
        'in_process_p95_below_50ms': 0 <= latency <= 50,
    }
    if not all(checks.values()):
        raise ValueError(f'quality gate rejected candidate: {checks}')
    return {'checks': checks, 'validation_ndcg': ndcg, 'baseline_ndcg': reference,
            'coverage': coverage, 'in_process_p95_ms': latency,
            'scope': 'Offline validation only; not a significance test or online promotion decision.'}


def verify(root, release):
    if not re.fullmatch(r'[0-9a-f]{64}', release):
        raise ValueError('invalid release identifier')
    path = root / release
    receipt = json.loads((path / 'release.json').read_text())
    hashes = {name: digest(path / name) for name in FILES}
    actual = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    if actual != release or receipt['sha256'] != hashes:
        raise ValueError('release integrity check failed')
    gate = quality_gate(path)
    if (receipt.get('release') != release or receipt.get('gate') != gate
            or receipt.get('model_version') != LiveRecommender(path).version):
        raise ValueError('release receipt integrity check failed')
    return path


def prepare(run, root):
    root.mkdir(parents=True, exist_ok=True)
    # Copy first; validate the immutable snapshot rather than a changing source run.
    with tempfile.TemporaryDirectory(dir=root, prefix='.prepare-') as tmp:
        stage = Path(tmp) / 'bundle'
        stage.mkdir()
        for name in FILES:
            target = stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(run / name, target)
        gate = quality_gate(stage)
        hashes = {name: digest(stage / name) for name in FILES}
        release = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
        receipt = {'release': release, 'sha256': hashes, 'gate': gate,
                   'model_version': LiveRecommender(stage).version}
        (stage / 'release.json').write_text(json.dumps(receipt, indent=2) + '\n')
        with lock(root):
            if not (root / release).exists():
                stage.rename(root / release)
            verify(root, release)
    return receipt


@contextmanager
def lock(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def select_release(root, release=None, rollback=False):
    with lock(root):
        pointer = root / 'current.json'
        old = json.loads(pointer.read_text()) if pointer.exists() else {}
        target = old.get('previous') if rollback else release
        if not target:
            raise ValueError('no rollback target or release supplied')
        path = verify(root, target)
        if old.get('release') == target:
            return old
        value = {'release': target, 'previous': old.get('release'),
                 'model_version': LiveRecommender(path).version,
                 'selected_at': datetime.now(timezone.utc).isoformat()}
        # One atomic document contains both selection and rollback target.
        with tempfile.NamedTemporaryFile(mode='w', dir=root, delete=False) as handle:
            json.dump(value, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(handle.name, pointer)
        return value


def selected_path(root):
    return verify(root, json.loads((root / 'current.json').read_text())['release'])


def track(root, release, uri):
    # MLflow 2.x artifact proxy resolution also consults the process tracking URI.
    import mlflow
    previous = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(uri)
    try:
        return _track(root, release, uri)
    finally:
        mlflow.set_tracking_uri(previous)


def _track(root, release, uri):
    from mlflow import MlflowClient
    from mlflow.exceptions import MlflowException
    path = verify(root, release)
    receipt = json.loads((path / 'release.json').read_text())
    client = MlflowClient(tracking_uri=uri, registry_uri=uri)
    experiment = client.get_experiment_by_name('recsyslab-offline')
    experiment_id = experiment.experiment_id if experiment else client.create_experiment('recsyslab-offline')
    run = client.create_run(experiment_id, tags={'release': release, 'model_version': receipt['model_version'],
                                               'evaluation_split': 'validation', 'bundle_format': 'recsyslab-item-cf-v1'})
    try:
        for key in ['validation_ndcg', 'baseline_ndcg', 'coverage', 'in_process_p95_ms']:
            client.log_metric(run.info.run_id, key, receipt['gate'][key])
        config = json.loads((path / 'manifest.json').read_text())['config']
        for key, value in config.items():
            client.log_param(run.info.run_id, key, json.dumps(value, sort_keys=True))
        client.log_artifacts(run.info.run_id, str(path), artifact_path='bundle')
        name = 'recsyslab-item-cf'
        try:
            client.create_registered_model(name)
        except MlflowException as exc:
            if exc.error_code != 'RESOURCE_ALREADY_EXISTS':
                raise
        version = client.create_model_version(name, run.info.artifact_uri + '/bundle', run.info.run_id,
                                               tags={'release': release, 'validation_gate': 'passed'})
        client.set_registered_model_alias(name, 'validated', version.version)
        client.set_terminated(run.info.run_id, 'FINISHED')
    except Exception:
        client.set_terminated(run.info.run_id, 'FAILED')
        raise
    return {'run_id': run.info.run_id, 'registered_model': name, 'version': version.version,
            'alias': 'validated', 'release': release, 'tracking_uri': uri}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'activate', 'rollback', 'verify', 'track'])
    parser.add_argument('--root', type=Path, default=Path('artifacts/releases'))
    parser.add_argument('--run', type=Path, default=Path('artifacts/phase2-1m'))
    parser.add_argument('--release')
    parser.add_argument('--tracking-uri', default=os.getenv('MLFLOW_TRACKING_URI', 'sqlite:///artifacts/mlflow.db'))
    args = parser.parse_args()
    if args.action == 'prepare':
        result = prepare(args.run, args.root)
    elif args.action in {'activate', 'rollback'}:
        result = select_release(args.root, args.release, args.action == 'rollback')
    elif args.action == 'track':
        result = track(args.root, args.release, args.tracking_uri)
    else:
        result = {'path': str(verify(args.root, args.release))}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

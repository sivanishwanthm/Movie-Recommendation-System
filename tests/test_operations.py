"""Release failures, real MLflow round trips, drift and HTTP instrumentation."""
import hashlib
import json
from pathlib import Path
import pandas as pd
import pytest
pytest.importorskip('fastapi')
pytest.importorskip('sqlalchemy')
pytest.importorskip('prometheus_client')
from recsyslab.operations.drift import compare
from recsyslab.operations.releases import prepare, verify, select_release, selected_path, track
from recsyslab.classical import ItemCF


def benchmark(tmp_path, variant=0):
    run = tmp_path / f'run{variant}'
    run.mkdir()
    frame = pd.DataFrame([(u, i, 5, 100) for u, items in [(1,[1,2,3]), (2,[2,3,4])] for i in items],
                         columns=['user_id','movie_id','rating','timestamp'])
    ItemCF.fit(frame, neighbors=3, shrinkage=variant, min_positives=3).save(run / 'item_cf')
    def metric(ndcg):
        return {'metrics':{'ndcg':{'mean':ndcg}}, 'users_evaluated':10,
                'catalog_coverage':.5, 'latency_ms_p95':1}
    (run / 'metrics.json').write_text(json.dumps({'validation':{'item_cf':metric(.3), 'popularity':metric(.2)}}))
    manifest = {'artifact_sha256':{str(p.relative_to(run)):hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in run.rglob('*') if p.is_file()},
                'selected_on_validation':'item_cf', 'config':{'seed':42}}
    (run / 'manifest.json').write_text(json.dumps(manifest))
    return run


def test_atomic_release_and_rollback(tmp_path):
    root = tmp_path / 'releases'
    a = prepare(benchmark(tmp_path), root)['release']
    b = prepare(benchmark(tmp_path, 2), root)['release']
    assert a != b
    select_release(root, a)
    select_release(root, b)
    assert selected_path(root).name == b
    assert select_release(root, rollback=True)['release'] == a
    assert selected_path(root).name == a
    (root / b / 'item_cf/model.json').write_text('{}')
    with pytest.raises(ValueError, match='integrity'):
        select_release(root, b)
    assert selected_path(root).name == a
    with pytest.raises(ValueError):
        verify(root, '../escape')


def test_gate_rejects_tampered_metrics_and_regression(tmp_path):
    run = benchmark(tmp_path)
    path = run / 'metrics.json'
    metrics = json.loads(path.read_text())
    metrics['validation']['item_cf']['metrics']['ndcg']['mean'] = .1
    path.write_text(json.dumps(metrics))
    with pytest.raises(ValueError, match='metrics integrity'):
        prepare(run, tmp_path / 'releases')
    manifest = json.loads((run / 'manifest.json').read_text())
    manifest['artifact_sha256']['metrics.json'] = hashlib.sha256(path.read_bytes()).hexdigest()
    (run / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='quality gate'):
        prepare(run, tmp_path / 'releases')


def test_real_mlflow_registry_round_trip(tmp_path, monkeypatch):
    pytest.importorskip('mlflow')
    monkeypatch.chdir(tmp_path)
    root = tmp_path / 'releases'
    receipt = prepare(benchmark(tmp_path), root)
    uri = 'sqlite:///' + str(tmp_path / 'tracking.db')
    import mlflow
    original_uri = mlflow.get_tracking_uri()
    result = track(root, receipt['release'], uri)
    assert mlflow.get_tracking_uri() == original_uri
    with pytest.raises(ValueError):
        track(root, 'invalid-release', uri)
    assert mlflow.get_tracking_uri() == original_uri
    from mlflow import MlflowClient
    client = MlflowClient(tracking_uri=uri, registry_uri=uri)
    version = client.get_model_version_by_alias(result['registered_model'], 'validated')
    assert version.run_id == result['run_id']
    run = client.get_run(version.run_id)
    assert run.info.status == 'FINISHED'
    assert run.data.metrics['validation_ndcg'] == .3
    downloaded = Path(client.download_artifacts(version.run_id, 'bundle'))
    assert json.loads((downloaded / 'release.json').read_text())['release'] == receipt['release']


def test_drift_identical_shifted_sparse_and_invalid():
    a = pd.DataFrame({'rating':[4]*200, 'movie_id':[1]*200})
    assert compare(a,a)['rating_js_divergence_bits'] == 0
    assert compare(a,a)['status'] == 'ok'
    b = pd.DataFrame({'rating':[1]*200, 'movie_id':[2]*200})
    result = compare(a,b)
    assert result['status'] == 'alert' and result['unseen_item_fraction'] == 1
    assert result['rating_js_divergence_bits'] == pytest.approx(1)
    assert compare(a,b.head(1))['status'] == 'insufficient_data'
    with pytest.raises(ValueError):
        compare(a, pd.DataFrame({'rating':[float('nan')], 'movie_id':[1]}))


from test_serving import setup, HEADERS


def test_metrics_auth_bounded_routes_and_csrf(setup, caplog):
    from fastapi.testclient import TestClient
    from recsyslab.serving.app import create_app
    settings, _ = setup
    settings.metrics_token = 'test-secret'
    with TestClient(create_app(settings)) as client:
        assert client.get('/api/live').status_code == 200
        assert client.get('/api/metrics').status_code == 403
        assert client.get('/api/movies/123456789').status_code == 404
        assert client.post('/api/profiles', json={'name':'Private name'}).status_code == 403
        response = client.get('/api/metrics', headers={'Authorization':'Bearer test-secret'})
        assert response.status_code == 200
        assert 'route="/api/movies/{movie_id}"' in response.text
        assert 'status="403"' in response.text
        assert '123456789' not in response.text and 'Private name' not in response.text
        assert len(response.headers['x-request-id']) == 32


def test_metrics_disabled_by_default(setup):
    from fastapi.testclient import TestClient
    from recsyslab.serving.app import create_app
    settings, _ = setup
    with TestClient(create_app(settings)) as client:
        assert client.get('/api/metrics').status_code == 503


def test_receipt_tampering_is_rejected(tmp_path):
    root = tmp_path / 'releases'
    release = prepare(benchmark(tmp_path), root)['release']
    path = root / release / 'release.json'
    receipt = json.loads(path.read_text())
    receipt['gate']['validation_ndcg'] = 1.0
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='receipt integrity'):
        verify(root, release)


def test_release_startup_and_unhandled_error_metrics(setup, tmp_path):
    from fastapi.testclient import TestClient
    from recsyslab.serving.app import create_app
    settings, _ = setup
    settings.release_root = tmp_path / 'releases'
    release = prepare(benchmark(tmp_path), settings.release_root)['release']
    select_release(settings.release_root, release)
    app = create_app(settings)
    @app.get('/api/test-crash')
    def crash():
        raise RuntimeError('synthetic test failure')
    with TestClient(app, raise_server_exceptions=False) as client:
        assert client.get('/api/health').json()['release_id'] == release
        assert client.get('/api/test-crash').status_code == 500
        assert app.state.monitor.registry.get_sample_value('recsys_http_requests_total',
            {'method':'GET','route':'/api/test-crash','status':'500'}) == 1
    (settings.release_root / release / 'item_cf/model.json').write_text('{}')
    with pytest.raises(ValueError, match='integrity'):
        with TestClient(create_app(settings)):
            pass

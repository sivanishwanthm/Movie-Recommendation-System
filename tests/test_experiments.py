"""Synthetic statistical fixtures and real PostgreSQL experiment integration."""
from datetime import datetime, timedelta, timezone
from uuid import uuid4
import json
import numpy as np
import pytest
pytest.importorskip('fastapi')
pytest.importorskip('sqlalchemy')
from fastapi.testclient import TestClient
from sqlalchemy import select, text, inspect
from sqlalchemy.orm import Session
from recsyslab.experimentation.design import ExperimentSpec, assign
from recsyslab.experimentation.statistics import wilson, conversion_test, sample_size, srm, mean_difference
from recsyslab.experimentation.service import register, analyze, summarize, enroll
from recsyslab.experimentation.simulate import simulate, null_calibration
from recsyslab.serving.app import create_app
from recsyslab.serving.model import LiveRecommender
from recsyslab.serving.database import Assignment, Experiment, ExperimentAttempt, ExperimentBatch, Exposure, Batch, Event, Profile, now
from recsyslab.serving.migrations import upgrade
from test_serving import setup, HEADERS, create


def spec_for(settings=None, **overrides):
    at=now()
    values=dict(name='popularity-vs-cf',version=1,salt='fixed-independent-experiment-salt',
                starts_at=at-timedelta(days=2),ends_at=at+timedelta(days=14),
                model_version=LiveRecommender(settings.model_run).version if settings else 'test-model-v1',
                catalog_size=6 if settings else 100)
    values.update(overrides)
    return ExperimentSpec(**values)


def test_design_assignment_stability_and_validation():
    spec=spec_for()
    before=[assign(spec,str(i)) for i in range(5000)]
    assert before==[assign(ExperimentSpec.model_validate(spec.canonical()),str(i)) for i in range(5000)]
    assert .45<before.count('treatment')/len(before)<.55
    second=spec.model_copy(update={'version':2})
    assert any(assign(second,str(i))!=before[i] for i in range(100))
    with pytest.raises(ValueError):spec_for(starts_at=datetime(2026,1,1))
    with pytest.raises(ValueError):spec_for(minimum_detectable_effect=.95)
    with pytest.raises(ValueError):spec_for(control='item_cf')


def test_statistics_edges_and_reference_values():
    assert wilson(0,0) is None
    assert wilson(0,10)[1]==pytest.approx(.2775328,abs=1e-6)
    assert conversion_test(0,0,1,10)['p_value'] is None
    assert conversion_test(0,10,0,10)['p_value']==1
    assert conversion_test(0,10,0,10)['difference_ci'][1]>0
    strong=conversion_test(10,100,40,100)
    assert strong['absolute_difference']==pytest.approx(.3)
    assert strong['p_value']<.00001 and strong['difference_ci'][0]>0
    assert srm(1000,1000)['mismatch'] is False
    assert srm(100,900)['mismatch'] is True
    assert srm(300,700,probability=.7)['p_value']==1
    plan=sample_size(.1,.02)
    assert 3800<plan['control_users']<3900
    assert plan['control_users']==plan['treatment_users']
    assert sample_size(.1,.01)['total_users']>plan['total_users']
    assert mean_difference([1,1],[2,2],samples=20)['ci95']==[1.,1.]


def record(identifier,variant,at,converted=False):
    return {'id':identifier,'variant':variant,'assigned_at':at,'window_ends_at':at+timedelta(hours=24),
            'converted':converted,'exposed':converted,'attempts':[{'status_code':200,'latency_ms':10.,'items':[1,2]}]}


def test_maturity_itt_and_review_gate():
    spec=spec_for();at=now()
    records=[record('c','control',at-timedelta(hours=25)),record('t','treatment',at-timedelta(hours=25),True),
             record('pending','treatment',at-timedelta(hours=2),True)]
    result=summarize(spec,records,at,samples=20)
    assert result['primary']['control']['users']==1  # Unexposed, unclicked users remain.
    assert result['primary']['treatment']['users']==1
    assert result['primary']['treatment']['converted']==1
    assert result['srm']['treatment']==2  # SRM includes immature enrollments.
    assert result['review_eligible'] is False
    assert 'fixed_horizon_not_complete' in result['review_blockers']
    records[0]['attempts'][0]['status_code']=None
    assert 'incomplete_request_telemetry' in summarize(spec,records,at,samples=20)['review_blockers']


def test_simulation_namespace_and_reproducibility():
    spec=spec_for()
    a=simulate(spec,users=200,seed=4)
    assert a==simulate(spec,users=200,seed=4)
    assert a['source']=='synthetic_simulation' and a['review_eligible'] is False
    assert 'synthetic_data_cannot_support_online_decisions' in a['review_blockers']
    assert simulate(spec,users=1000,seed=4,allocation_fault=True)['srm']['mismatch']
    assert null_calibration(20,100,seed=5)==null_calibration(20,100,seed=5)


def activate(settings,engine):
    spec=spec_for(settings)
    with Session(engine) as db:register(db,spec);db.commit()
    settings.active_experiment=spec.key
    settings.experiment_admin_token='test-only-operator-token'
    return spec


def make_eligible(client):
    create(client)
    for i in [1,2,3]:assert client.put(f'/api/ratings/{i}',json={'event_id':str(uuid4()),'value':5}).status_code==200


def test_definition_immutability_and_additive_migration(setup):
    settings,engine=setup
    with engine.begin() as c:
        c.execute(text('UPDATE schema_version SET version=1'))
        # Emulate a v1 database by dropping only new, empty experiment tables.
        for table in ['experiment_exposures','experiment_batches','experiment_attempts','experiment_assignments','experiments']:
            c.execute(text(f'DROP TABLE {table}'))
    assert upgrade(engine)==2 and upgrade(engine)==2
    with engine.connect() as c:assert c.execute(text('SELECT count(*) FROM movies')).scalar()==7
    spec=activate(settings,engine)
    with Session(engine) as db:
        register(db,spec)
        with pytest.raises(ValueError,match='immutable'):register(db,spec.model_copy(update={'salt':'a-new-salt-but-same-version'}))
    bad=spec.model_copy(update={'model_version':'different-model'})
    with Session(engine) as db:
        row=db.get(Experiment,spec.key);row.specification=bad.canonical();row.fingerprint=bad.fingerprint();db.commit()
    with pytest.raises(ValueError,match='does not match'):
        with TestClient(create_app(settings)):pass


def test_enrollment_is_stable_concurrent_and_exposure_is_visible(setup):
    from concurrent.futures import ThreadPoolExecutor
    settings,engine=setup;spec=activate(settings,engine)
    with TestClient(create_app(settings),headers=HEADERS) as client:
        create(client)
        assert client.post('/api/recommendations',json={}).json()['experiment'] is None
        assert client.get('/api/experiments/assignment').json()['assignment'] is None
        for i in [1,2,3]:client.put(f'/api/ratings/{i}',json={'event_id':str(uuid4()),'value':5})
        with ThreadPoolExecutor(max_workers=3) as pool:
            responses=list(pool.map(lambda _:client.post('/api/recommendations',json={'k':2}),range(3)))
        assert all(r.status_code==200 for r in responses)
        batches=[r.json() for r in responses]
        assert len({b['experiment']['variant'] for b in batches})==1
        with Session(engine) as db:
            assert len(db.scalars(select(Assignment)).all())==1
            assert db.scalars(select(Exposure)).all()==[]
            assert len(db.scalars(select(ExperimentAttempt)).all())==3
        batch=batches[0]
        for _ in range(2):assert client.post('/api/impressions',json={'batch_id':batch['batch_id'],'movie_ids':[batch['items'][0]['id']]}).status_code==200
        with Session(engine) as db:assert len(db.scalars(select(Exposure)).all())==1
        assert client.get(f'/api/experiments/{spec.key}/analysis').status_code==403
        report=client.get(f'/api/experiments/{spec.key}/analysis',headers={'X-Experiment-Admin':'test-only-operator-token'}).json()
        assert report['source']=='live_randomized'
        assert report['primary']['control']['users']+report['primary']['treatment']['users']==0
        cookies=client.cookies
    with TestClient(create_app(settings),headers=HEADERS) as resumed:
        resumed.cookies.update(cookies)
        again=resumed.post('/api/recommendations',json={}).json()
        assert again['experiment']==batch['experiment']


def test_failed_variant_keeps_assignment_and_error_telemetry(setup,monkeypatch):
    import recsyslab.experimentation.service as service
    settings,engine=setup;activate(settings,engine)
    monkeypatch.setattr(service,'assign',lambda spec,profile:'treatment')
    with TestClient(create_app(settings),headers=HEADERS) as client:
        make_eligible(client)
        monkeypatch.setattr(LiveRecommender,'recommend',lambda *args: (_ for _ in ()).throw(RuntimeError('scorer failure')))
        assert client.post('/api/recommendations',json={}).status_code==503
    with Session(engine) as db:
        assert len(db.scalars(select(Assignment)).all())==1
        attempt=db.scalar(select(ExperimentAttempt))
        assert attempt.status_code==503 and attempt.api_latency_ms is not None
        assert attempt.batch_id is None


def test_attribution_window_and_nonexperiment_clicks_excluded(setup):
    settings,engine=setup;spec=activate(settings,engine)
    with TestClient(create_app(settings),headers=HEADERS) as client:
        make_eligible(client)
        batch=client.post('/api/recommendations',json={'k':2}).json()
        movie=batch['items'][0]['id']
        client.post('/api/impressions',json={'batch_id':batch['batch_id'],'movie_ids':[movie]})
        eid=str(uuid4())
        assert client.post('/api/clicks',json={'event_id':eid,'batch_id':batch['batch_id'],'movie_id':movie}).status_code==200
    at=now()
    with Session(engine) as db:
        assignment=db.scalar(select(Assignment));assignment.assigned_at=at-timedelta(hours=25);assignment.window_ends_at=at-timedelta(hours=1)
        event=db.get(Event,eid);event.created_at=at-timedelta(hours=2)
        attempt=db.scalar(select(ExperimentAttempt));attempt.created_at=at-timedelta(hours=2)
        db.commit()
        report=analyze(db,spec,at)
        arm=assignment.variant
        assert report['primary'][arm]['converted']==1
        event.created_at=at # Click after the fixed follow-up must not count.
        db.commit()
        assert analyze(db,spec,at)['primary'][arm]['converted']==0
        extra=Batch(id=str(uuid4()),profile_id=assignment.profile_id,movie_ids=[movie],model_version='test',strategy='nonexperimental',latency_ms=1)
        db.add(extra);db.flush()
        db.add(Event(id=str(uuid4()),profile_id=assignment.profile_id,movie_id=movie,batch_id=extra.id,kind='click',created_at=at-timedelta(hours=2)))
        db.commit()
        assert analyze(db,spec,at)['primary'][arm]['converted']==0


def test_closed_enrollment_and_no_reassignment(setup):
    settings,engine=setup
    spec=spec_for(settings,ends_at=now()-timedelta(hours=1))
    with Session(engine) as db:register(db,spec);db.commit()
    settings.active_experiment=spec.key
    with TestClient(create_app(settings),headers=HEADERS) as client:
        make_eligible(client)
        assert client.post('/api/recommendations',json={}).json()['experiment'] is None
    with Session(engine) as db:assert db.scalars(select(Assignment)).all()==[]


def test_control_routes_to_popularity_and_expired_assignment_is_not_reenrolled(setup,monkeypatch):
    import recsyslab.experimentation.service as service
    settings,engine=setup;activate(settings,engine)
    monkeypatch.setattr(service,'assign',lambda spec,profile:'control')
    with TestClient(create_app(settings),headers=HEADERS) as client:
        make_eligible(client)
        original=LiveRecommender.recommend
        monkeypatch.setattr(LiveRecommender,'recommend',lambda *args: (_ for _ in ()).throw(AssertionError('control executed treatment')))
        response=client.post('/api/recommendations',json={'k':2})
        assert response.status_code==200 and response.json()['strategy']=='experiment_popularity_control'
        with Session(engine) as db:
            assignment=db.scalar(select(Assignment));assignment.window_ends_at=now()-timedelta(seconds=1);db.commit()
        monkeypatch.setattr(LiveRecommender,'recommend',original)
        assert client.post('/api/recommendations',json={}).json()['experiment'] is None
        with Session(engine) as db:assert len(db.scalars(select(Assignment)).all())==1


def test_complete_horizon_srm_and_missing_telemetry_block_review():
    at=now()
    spec=spec_for(ends_at=at-timedelta(hours=25),baseline_rate=.1,minimum_detectable_effect=.3)
    assigned=at-timedelta(hours=26)
    rows=[record(str(i),'control' if i<1000 else 'treatment',assigned,i%5==0) for i in range(2000)]
    clean=summarize(spec,rows,at,samples=20)
    assert clean['review_eligible'] is True
    assert 'winner' not in clean
    skewed=[dict(r,variant='control' if i<10 else 'treatment') for i,r in enumerate(rows)]
    skewed_report=summarize(spec,skewed,at,samples=20)
    assert skewed_report['srm']['mismatch'] is True and not skewed_report['review_eligible']
    rows[0]['attempts'][0]['latency_ms']=None
    assert 'incomplete_request_telemetry' in summarize(spec,rows,at,samples=20)['review_blockers']

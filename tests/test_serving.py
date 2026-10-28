"""Real PostgreSQL integration tests; synthetic fixtures are isolated per test schema."""
import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4
import pandas as pd
import pytest
pytest.importorskip('fastapi')
pytest.importorskip('sqlalchemy')
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
from recsyslab.classical import ItemCF
from recsyslab.serving.app import create_app, Settings
from recsyslab.serving.database import initialize, Movie, Rating, Profile, Event
from recsyslab.serving.model import LiveRecommender

HEADERS={'X-Recsys-Client':'web','Origin':'http://localhost:3000'}

@pytest.fixture
def setup(tmp_path):
    url=os.getenv('TEST_DATABASE_URL')
    if not url:pytest.skip('set TEST_DATABASE_URL for real PostgreSQL integration tests')
    admin=create_engine(url)
    schema='recsys_test_'+uuid4().hex
    with admin.begin() as conn:conn.execute(text(f'CREATE SCHEMA {schema}'))
    scoped=url+('&' if '?' in url else '?')+'options=-csearch_path%3D'+schema
    # Explicit test schema, never drop or truncate an existing application database.
    engine=create_engine(scoped)
    initialize(engine)
    ratings=pd.DataFrame([(u,i,5,100+u*10+i) for u,items in [(1,[1,2,3]),(2,[1,4,5]),(3,[2,4,6]),(4,[3,5,6])] for i in items],columns=['user_id','movie_id','rating','timestamp'])
    model=ItemCF.fit(ratings,neighbors=5,shrinkage=0,min_positives=3)
    run=tmp_path/'run';run.mkdir();model.save(run/'item_cf')
    hashes={str(p.relative_to(run)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (run/'item_cf').iterdir()}
    (run/'manifest.json').write_text(json.dumps({'artifact_sha256':hashes}))
    with Session(engine) as db:
        db.add_all([Movie(id=i,title=f'Test movie {i}',genres=['Drama']) for i in range(1,8)]);db.commit()
    settings=Settings(scoped,run,tmp_path/'reports')
    yield settings,engine
    engine.dispose()
    with admin.begin() as conn:conn.execute(text(f'DROP SCHEMA {schema} CASCADE'))
    admin.dispose()

@pytest.fixture
def client(setup):
    settings,_=setup
    with TestClient(create_app(settings),headers=HEADERS) as c:
        yield c


def create(c,name='Test profile'):
    response=c.post('/api/profiles',json={'name':name})
    assert response.status_code==201,response.text
    return response


def test_end_to_end_fallback_ratings_and_no_request_training(client,monkeypatch):
    monkeypatch.setattr(ItemCF,'fit',lambda *args,**kwargs: (_ for _ in ()).throw(AssertionError('request attempted training')))
    assert client.get('/api/me').status_code==401
    response=create(client)
    assert 'HttpOnly' in response.headers['set-cookie'] and 'SameSite=strict' in response.headers['set-cookie']
    cold=client.post('/api/recommendations',json={'k':4}).json()
    assert cold['strategy']=='popularity_fallback'
    for i in [1,2,3]:assert client.put(f'/api/ratings/{i}',json={'event_id':str(uuid4()),'value':5}).status_code==200
    warm=client.post('/api/recommendations',json={'k':4}).json()
    assert warm['strategy']=='item_cf_live_profile'
    assert {m['id'] for m in warm['items']}.isdisjoint({1,2,3})
    assert warm['positive_count']==3
    assert client.patch('/api/me',json={'name':'Updated'}).json()['name']=='Updated'
    assert len(client.get('/api/ratings').json())==3


def test_rating_validation_and_replay(client):
    create(client);eid=str(uuid4())
    assert client.put('/api/ratings/1',json={'event_id':eid,'value':4.2}).status_code==422
    payload={'event_id':eid,'value':4.5}
    assert client.put('/api/ratings/1',json=payload).json()['replayed'] is False
    assert client.put('/api/ratings/1',json=payload).json()['replayed'] is True
    assert client.put('/api/ratings/2',json=payload).status_code==409
    assert client.get('/api/analytics').json()['rating_events']==1
    assert client.put('/api/ratings/999',json={'event_id':str(uuid4()),'value':5}).status_code==404
    assert client.post('/api/recommendations',json={'k':999}).status_code==422


def test_impressions_clicks_and_empty_analytics(client):
    create(client)
    assert client.get('/api/analytics').json()['card_click_through_rate'] is None
    batch=client.post('/api/recommendations',json={'k':2}).json()
    movie=batch['items'][0]['id'];bid=batch['batch_id']
    click={'event_id':str(uuid4()),'batch_id':bid,'movie_id':movie}
    assert client.post('/api/clicks',json=click).status_code==409
    payload={'batch_id':bid,'movie_ids':[movie,movie]}
    assert client.post('/api/impressions',json=payload).json()['recorded']==1
    assert client.post('/api/impressions',json=payload).json()['recorded']==0
    assert client.post('/api/impressions',json={'batch_id':bid,'movie_ids':[999]}).status_code==422
    assert client.post('/api/clicks',json=click).json()['recorded'] is True
    assert client.post('/api/clicks',json=click).json()['recorded'] is False
    stats=client.get('/api/analytics').json()
    assert stats['impressions']==stats['clicks']==1
    assert stats['card_click_through_rate']==1
    assert stats['experiment_status']=='available_not_active'


def test_profile_isolation_csrf_and_reconnect(client,setup):
    settings,_=setup
    create(client,'First')
    batch=client.post('/api/recommendations',json={}).json()
    client.put('/api/ratings/1',json={'event_id':str(uuid4()),'value':5})
    with TestClient(create_app(settings),headers=HEADERS) as second:
        create(second,'Second')
        assert second.get('/api/ratings').json()==[]
        assert second.post('/api/impressions',json={'batch_id':batch['batch_id'],'movie_ids':[1]}).status_code==404
        assert second.get('/api/analytics').json()['rating_events']==0
    with TestClient(create_app(settings),headers=HEADERS) as reconnected:
        reconnected.cookies.update(client.cookies)
        assert len(reconnected.get('/api/ratings').json())==1
    assert client.patch('/api/me',json={'name':'Attack'},headers={'Origin':'http://other.invalid'}).status_code==403
    assert client.patch('/api/me',json={'name':'Attack'},headers={'X-Recsys-Client':''}).status_code==403


def test_catalog_search_and_no_fabricated_benchmarks(client):
    result=client.get('/api/movies?q=movie%202').json()
    assert result['total']==1 and result['items'][0]['id']==2
    assert client.get('/api/movies?offset=999').json()['items']==[]
    assert client.get('/api/movies/999').status_code==404
    assert client.get('/api/benchmarks').json()['phases']=={}
    assert client.get('/api/health').json()['status']=='ready'


def test_model_artifact_tampering_fails_startup(setup):
    settings,_=setup
    with (settings.model_run/'item_cf/similarity.npz').open('ab') as stream:stream.write(b'corruption')
    with pytest.raises(ValueError,match='integrity'):
        with TestClient(create_app(settings)):pass


def test_live_fold_in_uses_live_ratings_only(setup):
    settings,_=setup
    model=LiveRecommender(settings.model_run)
    # No MovieLens user ID is accepted; positive movie IDs drive similarity scores.
    values=[Rating(movie_id=i,value=5) for i in [1,2,3]]
    result,strategy,count=model.recommend(values,3)
    assert strategy=='item_cf_live_profile' and count==3
    assert set(result)=={4,5,6}
    values[0].value=1
    assert model.recommend(values,3)[1]=='popularity_fallback'


def test_concurrent_impressions_are_recorded_once(client):
    from concurrent.futures import ThreadPoolExecutor
    create(client)
    batch=client.post('/api/recommendations',json={'k':2}).json()
    payload={'batch_id':batch['batch_id'],'movie_ids':[batch['items'][0]['id']]}
    with ThreadPoolExecutor(max_workers=4) as pool:
        responses=list(pool.map(lambda _:client.post('/api/impressions',json=payload),range(4)))
    assert all(r.status_code==200 for r in responses)
    assert sum(r.json()['recorded'] for r in responses)==1
    assert client.get('/api/analytics').json()['impressions']==1

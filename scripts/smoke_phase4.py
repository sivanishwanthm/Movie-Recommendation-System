"""Scripted QA interactions. Run only against a disposable QA database/schema.

These are synthetic test actions, not real human engagement observations.
"""
import argparse
import json
from uuid import uuid4
import httpx


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url',default='http://localhost:3000')
    args=parser.parse_args()
    headers={'X-Recsys-Client':'web','Origin':args.base_url}
    with httpx.Client(base_url=args.base_url,headers=headers,timeout=30) as client:
        for page in ['/','/explore','/profile','/models','/activity','/movies/1']:
            response=client.get(page)
            assert response.status_code==200,(page,response.text)
            assert '<html' in response.text
        assert client.get('/api/health').json()['status']=='ready'
        created=client.post('/api/profiles',json={'name':'Synthetic HTTP QA profile'})
        assert created.status_code==201,created.text
        assert 'httponly' in created.headers.get('set-cookie','').lower()
        cold=client.post('/api/recommendations',json={'k':12}).json()
        assert cold['strategy']=='popularity_fallback'
        movie=cold['items'][0]['id'];bid=cold['batch_id']
        impression={'batch_id':bid,'movie_ids':[movie]}
        assert client.post('/api/impressions',json=impression).json()['recorded']==1
        assert client.post('/api/impressions',json=impression).json()['recorded']==0
        click={'event_id':str(uuid4()),'batch_id':bid,'movie_id':movie}
        assert client.post('/api/clicks',json=click).json()['recorded'] is True
        seen=[]
        for item in cold['items'][:3]:
            body={'event_id':str(uuid4()),'value':5}
            assert client.put(f"/api/ratings/{item['id']}",json=body).status_code==200
            assert client.put(f"/api/ratings/{item['id']}",json=body).json()['replayed'] is True
            seen.append(item['id'])
        warm=client.post('/api/recommendations',json={'k':12}).json()
        assert warm['strategy']=='item_cf_live_profile'
        assert set(seen).isdisjoint({m['id'] for m in warm['items']})
        stats=client.get('/api/analytics').json()
        assert stats['rating_events']==3 and stats['impressions']==1 and stats['clicks']==1
        benchmarks=client.get('/api/benchmarks').json()
        assert benchmarks['phases']['phase3']['test']['item_cf']['metrics']['ndcg']['mean']>0
        assert client.get('/api/movies?q=Toy%20Story').json()['total']>0
        cookies=client.cookies
    with httpx.Client(base_url=args.base_url,headers=headers,cookies=cookies) as resumed:
        assert len(resumed.get('/api/ratings').json())==3
    print(json.dumps({'result':'passed','pages_checked':6,'transport':'Next.js same-origin proxy → FastAPI → PostgreSQL',
                      'data_kind':'synthetic QA actions in isolated schema, not human engagement',
                      'checks':['HttpOnly session','profile persistence','catalog search','popularity fallback','live item-CF',
                                'rated-item exclusion','impression deduplication','rating replay','click attribution','artifact-backed metrics']},indent=2))

if __name__=='__main__':main()

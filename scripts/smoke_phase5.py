"""Synthetic HTTP QA only: run against a disposable QA schema with an active definition."""
import argparse
import json
import os
from uuid import uuid4
import httpx


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url',default='http://localhost:3000')
    args=parser.parse_args()
    token=os.environ['EXPERIMENT_ADMIN_TOKEN']
    headers={'X-Recsys-Client':'web','Origin':args.base_url}
    observed=set(); profiles=0
    with httpx.Client(base_url=args.base_url,headers=headers,timeout=30) as admin:
        assert admin.get('/experiments').status_code==200
        listing=admin.get('/api/experiments').json();key=listing['active_key']
        assert key,'QA server must explicitly activate its isolated definition'
        assert admin.get(f'/api/experiments/{key}/analysis').status_code==403
        synthetic=admin.get('/api/experiments/synthetic').json()
        assert synthetic['source']=='synthetic_simulation' and synthetic['available']
        for i in range(30):
            with httpx.Client(base_url=args.base_url,headers=headers,timeout=30) as client:
                assert client.post('/api/profiles',json={'name':f'Synthetic experiment QA {i}'}).status_code==201
                assert client.post('/api/recommendations',json={}).json()['experiment'] is None
                for movie in [1,2,3]:
                    assert client.put(f'/api/ratings/{movie}',json={'event_id':str(uuid4()),'value':5}).status_code==200
                batch=client.post('/api/recommendations',json={'k':2}).json()
                variant=batch['experiment']['variant'];observed.add(variant);profiles+=1
                assert client.get('/api/experiments/assignment').json()['assignment']['variant']==variant
                again=client.post('/api/recommendations',json={'k':2}).json()
                assert again['experiment']==batch['experiment']
                movie=batch['items'][0]['id']
                payload={'batch_id':batch['batch_id'],'movie_ids':[movie]}
                assert client.post('/api/impressions',json=payload).json()['recorded']==1
                assert client.post('/api/impressions',json=payload).json()['recorded']==0
                assert client.post('/api/clicks',json={'event_id':str(uuid4()),'batch_id':batch['batch_id'],'movie_id':movie}).status_code==200
            if len(observed)==2:break
        assert observed=={'control','treatment'},'both branches must be exercised'
        report=admin.get(f'/api/experiments/{key}/analysis',headers={'X-Experiment-Admin':token}).json()
        assert report['source']=='live_randomized' # This endpoint's QA database is isolated from live data.
        assert report['srm']['control']+report['srm']['treatment']==profiles
        assert report['primary']['control']['users']+report['primary']['treatment']['users']==0
        assert report['review_eligible'] is False
    print(json.dumps({'result':'passed','data_kind':'synthetic QA interactions in disposable database schema',
                      'not_live_engagement':True,'variants_exercised':sorted(observed),
                      'checks':['Next.js experiment route','protected operator report','separate synthetic source',
                                'pre-assignment eligibility','deterministic repeated assignment','both variant branches',
                                'idempotent impressions','attributed clicks','immature-outcome exclusion','no premature decision']},indent=2))

if __name__=='__main__':main()
